# Results template

Copy numbers from the generated files after you run `make eval` on the real splits. Leave a cell blank when the file says the run was not evaluated. Do not type a number that is not in those files.

Fusion tables are the **synthetic proxy protocol**. Say that in the report. They are not field validation.

## Image, by domain

Source files: `reports/image/aerial_thermal.json` and `reports/image/camera_trap_rgb.json`.

Do not average these two domains into one headline number.

| Domain | mAP50 | mAP50-95 | Human precision | Human recall | Vehicle precision | Vehicle recall | Animal precision | Animal recall | Small-object recall | Empty-frame false-positive rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aerial thermal |  |  |  |  |  |  |  |  |  |  |
| Camera trap |  |  |  |  |  |  |  |  |  |  |

Weights kind (`smoke`, `trained`, or not loaded) goes in a caption, from the same JSON.

## Audio, side by side

Source file: `reports/audio/experiments.json`.

| Experiment | Train site | Test site | Precision | Recall | F1 | False-positive rate |
| --- | --- | --- | --- | --- | --- | --- |
| Belize → Vietnam | belize | vietnam |  |  |  |  |
| Vietnam → Belize | vietnam | belize |  |  |  |  |
| In-domain Belize | belize | belize |  |  |  |  |
| In-domain Vietnam | vietnam | vietnam |  |  |  |  |

The two cross-site rows are the generalization result. The in-domain rows are controls. Do not replace them with a single accuracy.

## Fusion proxy protocol

Source file: `reports/fusion/proxy.json`.

State in the caption: synthetic proxy protocol, not field validation.

Operating point `high` (the default alert is HIGH):

| System | Precision | Recall | F1 | False-positive rate |
| --- | --- | --- | --- | --- |
| Image only |  |  |  |  |
| Audio only |  |  |  |  |
| Fused |  |  |  |  |

Operating point `indicator` (HIGH or MEDIUM counts as positive):

| System | Precision | Recall | F1 | False-positive rate |
| --- | --- | --- | --- | --- |
| Image only |  |  |  |  |
| Audio only |  |  |  |  |
| Fused |  |  |  |  |

Failure-analysis rows to discuss, from `failure_analysis` in the same file:

- Impulse-noise clips (thunder, branch snaps) that the audio branch treats like a gunshot.
- Small or canopy-occluded people that the image branch misses, so fusion stays below HIGH.
- True proxy HIGH events that fusion suppresses.

## Latency

Source file: `reports/latency.json`.

| End-to-end seconds | Within the configured target | Device |
| --- | --- | --- |
|  |  | cpu |

The target itself lives in `configs/serve.yaml` (`latency_target_s`). The measured seconds come only from the JSON.

## What not to claim

- There is no field-aligned image+audio incident set, so fusion recall is not a field recall.
- A single human, vehicle, or gunshot detection is an indicator, not a confirmed poaching event.
- Sample-data smoke checkpoints are not the project result. Check `weights_kind` before copying a number.
