.PHONY: install install-serve test preflight

install:
	python -m pip install -e .

install-serve:
	python -m pip install -e '.[serve]'

test:
	python tests/test_core.py
	python -m compileall -q src scripts
	bash -n run_pipeline.sh deploy/*.sh

preflight:
	python scripts/preflight.py --require_gpus 4
