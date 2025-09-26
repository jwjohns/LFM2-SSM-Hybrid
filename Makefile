UV?=uv
PY?=python3

.PHONY: setup lint test train infer export parity

setup:
	$(UV) venv
	. .venv/bin/activate && $(UV) pip install -e .[dev]
	. .venv/bin/activate && $(UV) pip install torch transformers

lint:
	$(UV) run ruff check src tests || true
	$(UV) run black --check src tests || true

test:
	$(UV) run pytest -q

train:
	$(UV) run python scripts/train_torch.py --device cpu --steps 5

infer:
	$(UV) run python scripts/infer_torch.py --prompt_ids 1,2,3,4

export:
	$(UV) run python scripts/export_npz.py --out export/model.npz --meta export/meta.json

parity:
	$(UV) run python scripts/parity_check.py --tokens 1,2,3,4
