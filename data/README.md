# Datasets

GuardianEye does not ship data. Public sets do not contain a labelled "poacher" class, and they do not contain time-aligned image+audio incidents. The image branch learns observable classes (`human`, `vehicle`, `animal`, and empty frames with no boxes). The audio branch learns `gunshot` versus `background`. Fusion is decision-level only.

Empty frames and background clips are the majority class in the field. Do not delete them to balance a spreadsheet.

Download URLs are left as `TODO` in `configs/data.yaml`. Fill those in yourself. `scripts/download_data.py` will not guess a link. When a URL and a sha256 are both set, the script checks the checksum.

## Image

| Key | Role | What to place under the dest path |
| --- | --- | --- |
| `birdsai` | Primary. Aerial thermal (BIRDSAI). | Images plus `annotations.jsonl`, `annotations.csv`, or YOLO `.txt` labels. Domain is `aerial_thermal`. |
| `cct` | Caltech Camera Traps, including empty frames. | Same layouts. Domain is `camera_trap_rgb`. |
| `serengeti` | Snapshot Serengeti, supporting camera-trap source. | Same layouts. Domain is `camera_trap_rgb`. |

Doull et al. is supplementary and available on request. Nothing in this repo depends on it.

`annotations.csv` columns: `filename,label,x1,y1,x2,y2`. A row with an empty label is a negative image. Labels are mapped through `image.label_map` in `configs/data.yaml`. Unknown labels are skipped and logged.

Thermal frames are stored two ways, selected by `image.thermal_representation`:

- `replicate` copies the single channel to three identical channels.
- `pseudocolor` is a percentile-normalized iron-style false color.

Evaluation never pools `aerial_thermal` with `camera_trap_rgb`.

## Audio

| Key | Role | What to place under the dest path |
| --- | --- | --- |
| `belize_gunshot` | Primary. Tropical Forest Gunshot dataset (Katsis et al., Belize). | `gunshot/` and `background/` folders of audio files. |
| `vietnam_gunshot` | Chu Mom Ray, used for cross-site testing. | WAV files and/or spectrogram images under `gunshot/` and `background/`. |

Spectrogram images are an audio representation. They are not a third modality. The preparer resizes them to a log-mel grid and does not invent a waveform for them.

Splits:

- Belize → Vietnam: train and validate on Belize only, test on all of Vietnam.
- Vietnam → Belize: the swap.
- In-domain Belize and in-domain Vietnam: train/val/test inside one site.

Background clips are kept in every split.

## Sample data

`make setup` writes a tiny synthetic tree under `data/sample/` so tests and the dashboard run before the real archives arrive. Those files are generated shapes and noises. They are not field recordings. Do not quote scores from that tree as dataset results.

## Prepared layout

`scripts/prepare_image_data.py` writes `data/processed/images/<domain>/` in YOLO format, with an empty label file for every negative image and a `domain` field in `manifest.jsonl`.

`scripts/prepare_audio_data.py` writes fixed-length clips, `.npy` log-mel features, and `data/processed/audio/splits/`.

`scripts/build_paired_events.py` writes `data/processed/paired/events.jsonl`. Every row has `protocol: synthetic_proxy`. Those pairs are constructed from labels. They are not time-aligned incidents.
