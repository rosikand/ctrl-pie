PYTHON ?= .venv/bin/python

.PHONY: seed smoke modal-panic
seed:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.seed

smoke:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.smoke

modal-panic:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.modal_panic
