PYTHON ?= .venv/bin/python

.PHONY: seed
seed:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.seed
