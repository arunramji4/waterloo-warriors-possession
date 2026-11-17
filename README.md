# Football possession with two temporal contexts

Submission method: run the frozen original-architecture ±5s and ±10s Transformers, smooth each independently at 2 seconds, then predict **D if either says D; otherwise retain the 5s label**. Do not smooth again after merging.

Both checkpoints trained on 24 external matches (5 DFL, 16 SkillCorner, 3 Metrica), with no Barcelona pseudo-label training examples. External validation selected the checkpoints. Barcelona observation statistics informed ball augmentation; algorithmic pseudo-labels were used only for diagnostic comparisons. See [technical document](docs/technical_document.md).

## Install

Tested runtime: Python 3.14.2, CPU, PyTorch 2.10.0, NumPy 2.4.3, SciPy 1.17.1. Inference fixes CPU threads at four and enables deterministic algorithms. Use this pinned runtime for the reference file; byte-identical output across other runtimes/hardware has not been established.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Provide the licensed assets

Place these files from the supplied data bundle at the following paths:

```text
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/broadcast/BAR-ATH_broadcast_atd.txt
data/broadcast/BAR-ATH_broadcast_atd.xml
data/broadcast/BAR-ATH_broadcast_homography.csv
```

Video, tactical tagging, public possession ground truth and private ground truth are not inputs to submission generation. The published half bounds are fixed metadata: `[11185,90652)` and `[110808,190198)` on the 25 Hz, zero-based tactical timeline. Broadcast positions supplement tactical gaps only, including geometry-accepted out-of-bounds positions. Feed alignment is recomputed from player geometry; no alignment table or tracking cache is shipped.

## Regenerate and validate

```sh
python reproduce.py --output outputs/submission.csv
```

This recomputes alignment, features, full-half predictions, independent smoothing and the final merge, then runs the official format validator. All 199,319 frames appear in `frame,label` order. Both full halves receive H/A/D predictions, including the public five-minute interval; outside the halves is X. No target labels are read. Intermediate observations and predictions stay in ignored `outputs/`.

To rerun inference after preprocessing:

```sh
python src/predict.py --output outputs/submission.csv
python evaluation/validate_submission.py outputs/submission.csv
```

`outputs/submission_manifest.json` records the output SHA-256 and label counts. A complete preprocessing-and-inference rerun produced the same bytes: `8130c7b38d0fe2b4ee0c2f43d9445595867ae5d2db4260160793d7531911f8e2`. See `results/reproduction_audit.json`. `models/manifest.json` records the original training-checkpoint hashes and the stripped inference-checkpoint hashes. Model files include only model state and input-window length, not optimizers or training samples.

## Method and results

The shared player encoder feeds masked mean/max pooling for each team, followed by a two-layer, 64-dimensional, four-head temporal Transformer. Team membership is supplied; player ordering is irrelevant. Both models use 5 Hz observations and 4 Hz prediction centres expanded to the 25 Hz video grid.

| Model | External test spell F1 | Pseudo agreement after 2s smoothing |
|---|---:|---:|
| Original ±5s | 0.5472 | 0.5926 |
| Original ±10s | 0.5544 | 0.5356 |
| Combined dead override | Not measured | 0.5865 |

These are separate evaluations. Pseudo-label agreement is not verified accuracy. The ensemble was chosen to favour extra stoppage calls, not because it beat 5s alone in this diagnostic. Full-half submission predictions are regenerated separately from diagnostic files, which excluded the public interval. Checkpoint selection never used the Barcelona comparisons.

Training records for the two submitted models are in `docs/5s/` and `docs/10s/`. The supplied weights are sufficient to reproduce predictions; retraining from random initialization is not required. External data sources and labels are described in [training data](docs/training_data.md). The technical document explains why ball masking differs between sources and why the goalkeeper rule remains common to all three.

## Submission deliverables

Upload the regenerated CSV and [technical document](docs/technical_document.md), renamed after your registered team. Supply the URL of this repository after publishing it publicly. The organiser requires one response per team. Team and author names are pending.

Do not publish `data/`, `outputs/`, match videos, feature arrays, alignment tables or derived frame-level tracking. Only code, model weights, training records, aggregate results and documentation belong in the public repository.

The official validator, metric implementation and game metadata are from [Metrica Sports](https://github.com/metrica-sports/more-than-a-hack-2027), with its licence retained in `third_party/ORGANISER_LICENSE`. Those files contain no possession ground truth. The competition data has separate redistribution restrictions.
