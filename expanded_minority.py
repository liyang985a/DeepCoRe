"""Expanded paired CE/sqrt-weighted CE study across source protocols."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import controlled_image_encoding as base
import numpy as np
import torch
from torch.utils.data import DataLoader
from deepcore.models import Branch, LOSS_C, LOSS_K, ModifiedResNet50
from deepcore.models.losses import FocalLoss, clip_gradient


def run(source, dest, variant):
    meta = json.loads((source / "protocol.json").read_text())
    seed = meta["training_seed"]
    target = dest / meta["dataset"] / variant / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "result.json").exists():
        print(f"SKIP {target}", flush=True)
        return
    started = time.time()
    saved = np.load(source / "pixels.npz")
    arrays = {s: (saved[f"{s}_x"], saved[f"{s}_y"]) for s in ("train", "val", "test")}
    for s, (x, y) in arrays.items():
        assert base.digest(x) == meta["splits"][s]["pixel_sha256"]
        assert base.digest(y) == meta["splits"][s]["labels_sha256"]
    datasets = {s: base.CachedGray(x, y, "compact_zero", np.arange(x.shape[1]))
                for s, (x, y) in arrays.items()}
    base.seed_all(seed)
    device = torch.device("cuda:0")
    model = ModifiedResNet50(str(base.ROOT / "weights/pytorch_model.bin")).to(device)
    branch = Branch(meta["num_classes"]).to(device)
    initial = hashlib.sha256()
    for module in (model, branch):
        for key, tensor in module.state_dict().items():
            initial.update(key.encode())
            initial.update(tensor.detach().cpu().numpy().tobytes())
    params = [p for p in list(model.parameters())+list(branch.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.00025, weight_decay=.01)
    counts = np.bincount(arrays["train"][1], minlength=meta["num_classes"])
    assert (counts > 0).all()
    raw_weights = counts.astype(float) ** -.5
    weights = raw_weights / np.average(raw_weights, weights=counts)
    tensor_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=0, weight=tensor_weights if variant == "sqrt_ce" else None)
    lossk, lossc = LOSS_K().to(device), LOSS_C().to(device)
    protocol = {
        "dataset": meta["dataset"], "variant": variant, "seed": seed,
        "source_protocol_sha256": hashlib.sha256((source/"protocol.json").read_bytes()).hexdigest(),
        "input": str(source.resolve()), "split_seed": meta["split_seed"],
        "epochs": 8, "batch_size": 128, "classification_reduction": "arithmetic batch mean",
        "class_weight_rule": "n_c^(-1/2), normalized to training sample mean one",
        "class_weights": weights.tolist() if variant == "sqrt_ce" else None,
        "checkpoint_selection": "first maximum validation macro F1, all training classes",
        "optimizer": "AdamW lr .00025 wd .01; two-epoch warmup then cosine to 1%",
        "auxiliary_weights": {"activation_matching": .4, "cosine_alignment": 2.5},
        "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "precision": "BF16 autocast; FP32 losses",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), Path(base.__file__),
                           base.ROOT/"deepcore/models/losses.py")},
        "initialization_sha256": initial.hexdigest(), "exploratory": True,
    }
    base.write_json(target / "protocol.json", protocol)
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ds, batch_size=128, shuffle=s == "train",
                            generator=generator if s == "train" else None,
                            num_workers=0, pin_memory=True) for s,ds in datasets.items()}
    history, best = [], -1.
    for epoch in range(8):
        multiplier = ((epoch+1)/2 if epoch < 2 else
                      .01+.99*.5*(1+math.cos(math.pi*(epoch-2)/5)))
        for group in optimizer.param_groups:
            group["lr"] = .00025*multiplier
        model.train()
        branch.train()
        tic, total, seen = time.time(), 0., 0
        for images, labels in loaders["train"]:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, p1, d, p2, att = branch(model(base.normalized_batch(images, device)))
            loss = criterion(p1.float(), labels) + criterion(p2.float(), labels)
            loss = loss+.4*lossk([a.float() for a in att])+2.5*lossc(x.float(), d.float())
            assert torch.isfinite(loss), (target, epoch)
            loss.backward()
            clip_gradient(optimizer)
            optimizer.step()
            total += loss.item()*len(labels)
            seen += len(labels)
        vp = base.predict(model, branch, loaders["val"], device)
        val = base.metrics(arrays["val"][1], vp, meta["classes"])
        history.append({"epoch": epoch+1, "lr": optimizer.param_groups[0]["lr"],
                        "loss": total/seen, "val": val, "seconds": time.time()-tic})
        np.savez_compressed(target/f"val_epoch{epoch+1}.npz", y=arrays["val"][1], pred=vp)
        if val["macro_f1_all_train"] > best:
            best = val["macro_f1_all_train"]
            torch.save({"model": model.state_dict(), "branch": branch.state_dict(),
                        "epoch": epoch+1}, target/"best.pt")
        base.write_json(target/"history.json", history)
        print(json.dumps({"dataset": meta["dataset"], "variant": variant,
                          "seed": seed, "epoch": epoch+1,
                          "val_macro": val["macro_f1_all_train"],
                          "seconds": time.time()-tic}), flush=True)
    state = torch.load(target/"best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    branch.load_state_dict(state["branch"])
    tp = base.predict(model, branch, loaders["test"], device)
    np.savez_compressed(target/"test_predictions.npz", y=arrays["test"][1], pred=tp)
    result = {"dataset": meta["dataset"], "variant": variant, "seed": seed,
              "split_seed": meta["split_seed"], "classes": meta["classes"],
              "initialization_sha256": initial.hexdigest(), "selected_epoch": state["epoch"],
              "test": base.metrics(arrays["test"][1], tp, meta["classes"]),
              "total_seconds": time.time()-started}
    base.write_json(target/"result.json", result)
    print("RESULT "+json.dumps(result), flush=True)
    del state, optimizer, model, branch, datasets, loaders
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["unsw", "kdd", "cic"])
    args = p.parse_args()
    torch.set_num_threads(6)
    assert torch.cuda.is_available()
    for dataset in args.datasets:
        for source in sorted((args.inputs/dataset).glob("split*")):
            for variant in ("ce", "sqrt_ce"):
                run(source, args.output, variant)
