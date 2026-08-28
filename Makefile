PYTHON ?= .venv/bin/python

.PHONY: seed modal-panic
seed:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.seed

modal-panic:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.modal_panic
