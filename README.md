# DeepCoRe supplementary evidence

This archive accompanies the marked and clean manuscripts and responses to both reviewers. It contains existing evidence; no additional model fitting, inference, input preparation or resampling was performed when assembling it.

Start with `manifest.csv` for the exact included files, sizes and SHA-256 checksums. Files retain their original project-relative names below `artifacts/`. Resolve a path printed in a result or registry by adding `artifacts/`; consult the manifest to determine whether that referenced file is included. An internal reference does not by itself imply inclusion.

## Reading order

| Purpose | Project-relative path below `artifacts/` |
| --- | --- |
| All 75 distinct fits; result/history/protocol locators and hashes | `revision/R2_16/evidence/combined_run_registry.csv` |
| Historical retained confusion matrices and corrected aggregate/class metrics | `revision/R2_08/evidence/metric_reconciliation.json` |
| Image conversion and fixed-split encoding controls | `revision/R2_03/evidence/summary.json` |
| Initial fixed-split objective controls | `revision/R2_04/evidence/summary.json` |
| Final three-dataset CE/weighted-CE study | `revision/R2_04_extended/evidence/summary.json` |
| Input provenance, overlap findings and train-only audit | `revision/R2_11/evidence/preprocessing_audit.json` |
| Final UNSW focal controls | `revision/R2_12/evidence/focal_summary.json` |
| Final UNSW mixed/balanced training resampling | `revision/R2_12_sampling/evidence/sampling_summary.json` |
| Literature score/protocol audit, including withdrawn original rows | `revision/R2_13/evidence/literature_protocol_audit.csv` |
| Matched plain-ResNet/random-forest controls | `revision/R2_13/evidence/baseline_summary.json` |
| Resource counts and timing definitions | `revision/R2_14/evidence/resource_summary.json` |
| All 1,040 recorded timing calls | `revision/R2_14/evidence/raw_timings.csv` |
| Repeated-run definitions and the earlier 60-fit registry | `revision/R2_15/evidence/seed_summary.json` and `run_registry.csv` |
| Paired ablations, all outcomes and paired differences | `revision/R2_16/evidence/ablation_summary.json` |
| Historical Tables 11/12 and withdrawn summaries | `revision/R2_16/evidence/historical_audit.json` |
| Dataset partition and paired-baseline protocol check | `revision/R2_18/evidence/protocol_summary.json` |

Every result, validation history and input protocol named in the 75-fit registry is included and its recorded hash checked. The archive additionally includes all 331 files linked by the preceding scientific audit, plus selected plans, resource records and methodological checks. This includes retained prediction files used by those checks, but not every file referenced internally by every JSON. The manifest defines the archive's scope.

## Interpretation and limits

- The 75 rows describe different studies/settings; they are not 75 independent replications of one model. Each reported setting has three runs. Reused full controls count once.
- Early encoding/objective experiments use a fixed historical filtered UNSW population with disclosed predictor overlap. Final UNSW/KDD use training-fitted transformations and exact represented-predictor separation. These checks do not establish host/session/capture independence.
- CIC uses previously processed archive features and combines Heartbleed with Infiltration. Its inherited preprocessing cannot be certified train-only.
- Historical matrix recomputation does not recover original checkpoints or prediction-generating runs. Tables 11/12 retain unverified point estimates, and their validation-versus-test selection partition is unknown.
- Negative results, complete class misses, false-positive costs and interventions that outperform the full model are retained. Descriptive SDs and paired differences are not confidence intervals or significance tests.
- Offline timing excludes live capture, raw flow extraction, queueing and other stated pipeline stages. It is not an end-to-end operational IDS evaluation.
- Split seeds such as `20261121` are random-number identifiers, not dates of data collection or experiment execution.

Raw traffic, full feature/image datasets, model checkpoints, the complete source tree and an independent execution environment are excluded. This is an evidence archive for inspecting retained analyses and reconstructing reported summaries, not a standalone training-reproduction package. It does not supply unseen-attack, cross-dataset, temporal-shift or online experiments.
