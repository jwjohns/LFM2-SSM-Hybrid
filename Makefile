PY=python3

.PHONY: setup lint test train infer export parity

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]

lint:
	ruff check src tests || true
	black --check src tests || true

test:
	pytest -q

train:
	$(PY) scripts/train_torch.py --device cpu --steps 5

infer:
	$(PY) scripts/infer_torch.py --prompt_ids 1,2,3,4

export:
	$(PY) scripts/export_npz.py --out export/model.npz --meta export/meta.json

parity:
	$(PY) scripts/parity_check.py --tokens 1,2,3,4

