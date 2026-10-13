# Hugging Face Spaces

GuardianEye's deploy target is a Docker Space. Training stays on a GPU notebook. The Space only runs inference.

## Space front matter

The Space repository's root `README.md` must start with this block. `app_port` matches `configs/serve.yaml`.

```yaml
---
title: GuardianEye
emoji: 🌲
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8742
pinned: false
---
```

Prepend that block to this project's `README.md` when you push the Space, or let the Space README be this front matter plus a short description. The Dockerfile at the repository root is what the Docker SDK builds.

## Weights on the Hub

1. Create a model repo, for example `your-user/guardianeye-weights`.
2. Upload the checkpoints with this layout:

```text
image/aerial_thermal.pt
image/aerial_thermal.json
image/camera_trap_rgb.pt
image/camera_trap_rgb.json
audio/best.pt
audio/belize_to_vietnam.pt
audio/belize_to_vietnam_calibration.json
```

3. In the Space settings, set the secret or variable `HF_MODEL_REPO` to that repo id.
4. If the model repo is private, set `HF_TOKEN` as a Space secret.

On startup the API calls `huggingface_hub.snapshot_download` into `models/hf/`. Local files under `models/` win when they already exist. If the download fails, the service still boots and `/model-info` reports that weights are not loaded. It does not invent scores.

`notebooks/train_image.ipynb` and `notebooks/train_audio.ipynb` show the upload step after training.

## Space settings

- SDK: Docker
- Port: 8742
- Hardware: CPU is enough for the demo. The latency target is whatever `latency_target_s` is set to in `configs/serve.yaml` (the project requirement is under 5 seconds).
- Health check path: `/health`

The container listens on `0.0.0.0:8742`. The dashboard is at `/dashboard`. `GET /` redirects there.

## Other hosts

Render or Railway can build the same Dockerfile.

- Set the service port to 8742, or override `GUARDIANEYE_PORT`.
- Set `HF_MODEL_REPO` if you want weights pulled at boot.
- Do not put `HF_TOKEN` in the image. Use the host's secret store. `.env.example` lists the names.
