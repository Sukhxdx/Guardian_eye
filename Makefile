PYTHON ?= python3
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup data train-image train-audio eval serve test lint smoke eda

setup:
	test -x $(PY) || $(PYTHON) -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PY) scripts/generate_sample_data.py
	$(PY) scripts/prepare_image_data.py
	$(PY) scripts/prepare_audio_data.py
	$(PY) scripts/build_paired_events.py
	$(PY) scripts/eda.py

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/prepare_image_data.py
	$(PY) scripts/prepare_audio_data.py
	$(PY) scripts/build_paired_events.py
	$(PY) scripts/eda.py

train-image:
	$(PY) -m guardianeye.image.train $(if $(SMOKE),--smoke,)

train-audio:
	$(PY) -m guardianeye.audio.train $(if $(SMOKE),--smoke,)

eval:
	$(PY) -m guardianeye.eval.image_eval
	$(PY) -m guardianeye.eval.audio_eval
	$(PY) -m guardianeye.eval.fusion_eval
	$(PY) -m guardianeye.eval.latency
	$(PY) -c "from guardianeye.eval.reports_index import build_index; build_index()"

serve:
	$(PY) -m guardianeye.serve.api

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests
	$(PY) -m ruff format --check src scripts tests
	$(PY) -m mypy

smoke: 
	$(MAKE) train-audio SMOKE=1
	$(MAKE) train-image SMOKE=1
	$(MAKE) eval

eda:
	$(PY) scripts/eda.py
