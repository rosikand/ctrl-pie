PYTHON ?= .venv/bin/python

.PHONY: seed smoke modal-panic yam-probe
seed:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.seed

smoke:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.smoke

modal-panic:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.modal_panic

yam-probe:
	PYTHONPATH=backend/src $(PYTHON) -m ctrl_pi.yam_probe
