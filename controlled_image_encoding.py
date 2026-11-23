"""Fixed-budget, paired image-encoding study for reviewer R2-3.

This supplements, rather than reproduces, historical manuscript scores.
The supplied official UNSW splits remain separate. All learned preprocessing
uses the fixed training subset, and the test set never selects checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deepcore.data.dataset import default_transforms
from deepcore.data.tabular_to_image import arrange_features, normalize_features
from deepcore.models import Branch, LOSS_C, LOSS_K, ModifiedResNet50
from deepcore.models.losses import clip_gradient

VARIANTS = ("compact_zero", "compact_mean", "compact_column", "compact_permuted", "direct128_zero")
NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_ftp_login", "is_sm_ips_ports",
]
CATEGORICAL = ["proto", "service", "state"]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False))


def prepare(out, train_cap, val_cap):
    raw = ROOT / "data/unsw_nb15"
    def read(name):
        frame = pd.read_csv(raw / name)
        frame = frame.loc[frame.service.ne("-")].copy()
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=NUMERIC + CATEGORICAL + ["attack_cat"])
        return frame
    supplied_train = read("UNSW_NB15_training-set.csv")
    test = read("UNSW_NB15_testing-set.csv")
    # Row IDs are from the original supplied CSV, not the locally pooled files.
    candidate, validation = train_test_split(
        np.arange(len(supplied_train)), test_size=0.2, random_state=20261111,
        stratify=supplied_train.attack_cat)
    selected, _ = train_test_split(candidate, train_size=train_cap,
                                   random_state=20261112,
                                   stratify=supplied_train.iloc[candidate].attack_cat)
    validation, _ = train_test_split(validation, train_size=val_cap,
                                     random_state=20261113,
                                     stratify=supplied_train.iloc[validation].attack_cat)
    train = supplied_train.iloc[selected].copy()
    val = supplied_train.iloc[validation].copy()
    assert set(train.id).isdisjoint(val.id)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scale = MinMaxScaler()
    enc.fit(train[CATEGORICAL])
    scale.fit(train[NUMERIC])
    le = LabelEncoder().fit(train.attack_cat)
    # Exclude both the binary target `label` and attack_cat, as well as id.
    feature_names = NUMERIC + enc.get_feature_names_out(CATEGORICAL).tolist()
    assert not {"id", "label", "attack_cat"} & set(feature_names)
    split_arrays, split_meta = {}, {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        assert set(frame.attack_cat) <= set(le.classes_)
        x = np.column_stack((scale.transform(frame[NUMERIC]),
                             enc.transform(frame[CATEGORICAL])))
        assert np.isfinite(x).all()
        pixels = np.stack([normalize_features(row) for row in x])
        y = le.transform(frame.attack_cat).astype(np.int64)
        split_arrays[name] = pixels, y
        split_meta[name] = {
            "rows": len(y), "features": x.shape[1],
            "source_ids": frame.id.astype(int).tolist(),
            "pixel_sha256": digest(pixels), "labels_sha256": digest(y),
            "class_counts": dict(zip(le.classes_.tolist(),
                                    np.bincount(y, minlength=len(le.classes_)).tolist())),
            "constant_rows_after_scaling": int(np.sum(np.ptp(x, axis=1) == 0)),
        }
    permutation = np.random.default_rng(20261114).permutation(len(feature_names))
    meta = {
        "description": "New fixed-budget sensitivity experiment; not a reproduction of historical results.",
        "split_protocol": "Preserve supplied train/test boundaries; service='-' excluded separately. "
                          "Fixed stratified validation split and fixed stratified train/validation caps. "
                          "Fit categorical encoder, numeric MinMaxScaler and label encoder on selected train only.",
        "duplicate_policy": "No deduplication; fixed supplied split. This does not resolve historical leakage concerns.",
        "split_seed": 20261111, "train_selection_seed": 20261112, "val_selection_seed": 20261113,
        "features": feature_names, "num_classes": len(le.classes_),
        "classes": le.classes_.tolist(), "permutation_seed": 20261114,
        "permutation": permutation.tolist(), "splits": split_meta,
        "categories": {c: a.tolist() for c, a in zip(CATEGORICAL, enc.categories_)},
        "numeric_min": scale.data_min_.tolist(), "numeric_max": scale.data_max_.tolist(),
        "raw_csv_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv")
        },
    }
    write_json(out / "protocol.json", meta)
    np.savez_compressed(out / "fixed_split_pixels.npz",
                        **{f"{name}_{kind}": arr for name, pair in split_arrays.items()
                           for kind, arr in zip(("x", "y"), pair)})
    return split_arrays, meta, permutation


def canvas(row, variant, permutation):
    side = 128 if variant == "direct128_zero" else math.ceil(math.sqrt(len(row)))
    assert len(row) <= side * side
    values = row[permutation] if variant == "compact_permuted" else row
    padded = np.full(side * side, int(np.mean(row)) if variant == "compact_mean" else 0,
                     dtype=np.uint8)
    padded[:len(row)] = values
    image = padded.reshape(side, side)
    return image.T.copy() if variant == "compact_column" else image


class CachedGray(Dataset):
    def __init__(self, pixels, labels, variant, permutation):
        self.labels = torch.from_numpy(labels)
        self.images = np.stack([
            np.asarray(Image.fromarray(canvas(row, variant, permutation))
                       .resize((128, 128), resample=Image.Resampling.BILINEAR)).copy()
            for row in pixels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]), self.labels[index]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def normalized_batch(images, device):
    values = images.to(device).unsqueeze(1).expand(-1, 3, -1, -1).float().div_(255)
    mean = values.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = values.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (values - mean) / std


@torch.no_grad()
def predict(model, branch, loader, device):
    model.eval()
    branch.eval()
    result = []
    for images, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, p1, _, p2, _ = branch(model(normalized_batch(images, device)))
        result.append((p1.float() + p2.float()).argmax(1).cpu().numpy())
    return np.concatenate(result)


def metrics(y, pred, classes):
    supported = np.unique(y).tolist()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1_present": float(f1_score(y, pred, labels=supported, average="macro", zero_division=0)),
        "macro_f1_all_train": float(f1_score(y, pred, labels=list(range(len(classes))),
                                                 average="macro", zero_division=0)),
        "present_labels": supported,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
    }


def run_one(args, arrays, meta, permutation, variant, seed):
    target = args.output / variant / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "result.json").exists() and not args.pilot:
        print(f"SKIP completed {variant} seed{seed}", flush=True)
        return
    started = time.time()
    datasets = {n: CachedGray(*pair, variant, permutation) for n, pair in arrays.items()}
    # Verify cached grayscale -> RGB numerical path against the actual loader.
    reference = default_transforms()
    for j in range(min(16, len(arrays["train"][0]))):
        native = canvas(arrays["train"][0][j], variant, permutation)
        actual = reference(Image.fromarray(native).convert("RGB"))
        cached = normalized_batch(torch.from_numpy(datasets["train"].images[j:j + 1]), "cpu")[0]
        torch.testing.assert_close(actual, cached, rtol=0, atol=0)
        if variant == "compact_zero":
            expected = arrange_features(arrays["train"][0][j], len(native), len(native))
            np.testing.assert_array_equal(native, expected)
    seed_all(seed)
    device = torch.device("cuda:0")
    model = ModifiedResNet50(str(ROOT / "weights/pytorch_model.bin")).to(device)
    branch = Branch(meta["num_classes"]).to(device)
    initial_hash = hashlib.sha256()
    for module in (model, branch):
        for key, tensor in module.state_dict().items():
            initial_hash.update(key.encode())
            initial_hash.update(tensor.detach().cpu().numpy().tobytes())
    initialization_sha256 = initial_hash.hexdigest()
    params = [p for p in list(model.parameters()) + list(branch.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.00025, weight_decay=0.01)
    lossk, lossc = LOSS_K().to(device), LOSS_C().to(device)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train",
                         generator=generator if name == "train" else None, num_workers=0,
                         pin_memory=True)
        for name, ds in datasets.items()
    }
    history, best = [], -1
    # Pilot only times gradient steps; its results cannot enter the final study.
    if args.pilot:
        model.train()
        branch.train()
        torch.cuda.synchronize()
        step_times = []
        for step, (images, labels) in enumerate(loaders["train"]):
            tic = time.time()
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, p1, d, p2, att = branch(model(normalized_batch(images, device)))
            loss = F.cross_entropy(p1.float(), labels) + F.cross_entropy(p2.float(), labels)
            loss = loss + 0.4 * lossk([a.float() for a in att]) + 2.5 * lossc(x.float(), d.float())
            assert torch.isfinite(loss)
            loss.backward()
            clip_gradient(optimizer)
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.time() - tic)
            if step == 9:
                break
        report = {"variant": variant, "batch_size": args.batch_size,
                  "step_times": step_times, "mean_after_warmup": float(np.mean(step_times[2:])),
                  "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
        write_json(args.output / "pilot.json", report)
        print(json.dumps(report), flush=True)
        return
    for epoch in range(args.epochs):
        # Multiplicative schedule expressed explicitly to avoid the legacy LambdaLR bug.
        warmup = 2
        multiplier = ((epoch + 1) / warmup if epoch < warmup else
                      0.01 + 0.99 * 0.5 * (1 + math.cos(
                          math.pi * (epoch - warmup) / max(args.epochs - warmup - 1, 1))))
        for group in optimizer.param_groups:
            group["lr"] = 0.00025 * multiplier
        model.train()
        branch.train()
        total_loss, seen = 0.0, 0
        tic = time.time()
        for images, labels in loaders["train"]:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, p1, d, p2, att = branch(model(normalized_batch(images, device)))
            loss = F.cross_entropy(p1.float(), labels) + F.cross_entropy(p2.float(), labels)
            loss = loss + 0.4 * lossk([a.float() for a in att]) + 2.5 * lossc(x.float(), d.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} {seed} {epoch}: {loss}")
            loss.backward()
            clip_gradient(optimizer)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        vp = predict(model, branch, loaders["val"], device)
        val = metrics(arrays["val"][1], vp, meta["classes"])
        record = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                  "loss": total_loss / seen, "val": val, "seconds": time.time() - tic}
        history.append(record)
        if val["weighted_f1"] > best:
            best = val["weighted_f1"]
            torch.save({"model": model.state_dict(), "branch": branch.state_dict(),
                        "epoch": epoch + 1}, target / "best.pt")
        write_json(target / "history.json", history)
        print(json.dumps({"variant": variant, "seed": seed, "epoch": epoch + 1,
                          "loss": record["loss"], "val_wf1": best,
                          "seconds": record["seconds"]}), flush=True)
    state = torch.load(target / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    branch.load_state_dict(state["branch"])
    tp = predict(model, branch, loaders["test"], device)
    np.savez_compressed(target / "test_predictions.npz", y=arrays["test"][1], pred=tp)
    result = {
        "variant": variant, "seed": seed, "epochs": args.epochs,
        "initialization_sha256": initialization_sha256,
        "batch_size": args.batch_size, "selected_epoch": state["epoch"],
        "best_val_weighted_f1": best, "test": metrics(arrays["test"][1], tp, meta["classes"]),
        "total_seconds": time.time() - started, "precision": "BF16 autocast; FP32 losses and parameters",
        "transform_equivalence_checks": 16, "optimizer": "AdamW",
        "base_lr": 0.00025, "weight_decay": 0.01, "warmup_epochs": 2,
        "loss_weights": {"classification_each_head": 1, "activation_matching": 0.4, "cosine_alignment": 2.5},
        "class_weighting": False, "checkpoint_selection": "validation weighted F1",
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    write_json(target / "result.json", result)
    print("RESULT " + json.dumps(result), flush=True)
    del state, optimizer, branch, model, datasets, loaders
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--train-cap", type=int, default=20000)
    p.add_argument("--val-cap", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    assert torch.cuda.is_available(), "A GPU is required; no silent CPU fallback."
    arrays, meta, permutation = prepare(args.output, args.train_cap, args.val_cap)
    write_json(args.output / "environment.json", {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "numpy": np.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "weight_sha256": hashlib.sha256((ROOT / "weights/pytorch_model.bin").read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })
    for variant in args.variants:
        for seed in args.seeds:
            run_one(args, arrays, meta, permutation, variant, seed)
            if args.pilot:
                return


if __name__ == "__main__":
    main()
