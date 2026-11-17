# Waterloo Warriors football possession submission

**Authors:** Arun Ramji, Sam Zamir, Ethan Batiuk

Run the frozen ±5s and ±10s temporal Transformers, smooth each independently at 2 seconds, then predict **dead if either says dead; otherwise retain the 5s label**. There is no smoothing after the merge.

The models use shared player encoders, mean/max pooling per team, and a two-layer temporal Transformer. Both trained on external DFL, SkillCorner and Metrica matches without Barcelona pseudo-label training examples. See the [technical write-up](docs/technical_document.md) for the approach, training data, augmentation and limitations.

## Install

Reference runtime: Python 3.14.2 on CPU. Dependencies are pinned in `requirements.txt`; inference uses four CPU threads and deterministic algorithms.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Add the provided data

Place these licensed assets in the following paths:

```text
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/broadcast/BAR-ATH_broadcast_atd.txt
data/broadcast/BAR-ATH_broadcast_atd.xml
data/broadcast/BAR-ATH_broadcast_homography.csv
```

No video, tactical tagging or possession labels are required. Data and generated outputs are ignored by Git and must not be redistributed.

## Reproduce the submission

```sh
python reproduce.py --output outputs/submission.csv
```

This recomputes feed alignment from player geometry, supplements missing tactical ball observations with accepted broadcast observations, builds features, runs both models, smooths and merges predictions, and runs the official format validator. It writes one label for each of the 199,319 video frames, including predictions for the complete halves. It never reads ground-truth or pseudo-label possession files.

The reference CSV SHA-256 is:

```text
8130c7b38d0fe2b4ee0c2f43d9445595867ae5d2db4260160793d7531911f8e2
```

A complete preprocessing-and-inference rerun produced identical bytes in the reference runtime. Byte-identical output across other runtimes or hardware has not been established. The command also writes `outputs/submission_manifest.json` with output and checkpoint provenance. `models/manifest.json` records the frozen checkpoint hashes.

To validate a generated CSV separately:

```sh
python evaluation/validate_submission.py outputs/submission.csv
```

The validator, metric implementation and game metadata come from [Metrica Sports](https://github.com/metrica-sports/more-than-a-hack-2027). Their licence is retained in `third_party/ORGANISER_LICENSE`.
