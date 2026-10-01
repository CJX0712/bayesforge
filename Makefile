# BayesForge — author: 晨星
PY ?= python

.PHONY: install dev test lint demo check table clean lock

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev,sota]"

lint:
	$(PY) -m ruff check bayesforge tests examples

test:
	$(PY) -m pytest -q

check:
	$(PY) -m bayesforge.cli check --n-evals 16

demo:
	$(PY) examples/run_demo.py

table:
	$(PY) -m bayesforge.cli table --json benchmark.json

lock:
	$(PY) -m pip freeze > requirements.lock.txt

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
