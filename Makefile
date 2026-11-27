PYTHON ?= python

.PHONY: test build-release check-release

test:
	$(PYTHON) -m unittest discover -s tests

build-release:
	rm -f dist/kernelatlas-*.whl dist/kernelatlas-*.tar.gz
	$(PYTHON) -m build

check-release:
	$(PYTHON) -m twine check --strict dist/*
	$(PYTHON) scripts/check_release.py
