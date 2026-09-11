# FoldQ task runner.
#
# Windows note: GNU make does not ship with Windows. make.ps1 mirrors every target
# below, and tests/test_tooling.py fails if the two ever drift apart. Use
# `.\make.ps1 test` locally on Windows; CI runs this Makefile on ubuntu.

UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help sync lint format typecheck test check clean bench bench-fast figures

help:
	@echo "FoldQ targets:"
	@echo "  sync         Create or refresh the locked virtual environment"
	@echo "  lint         Run ruff's linter and check formatting"
	@echo "  format       Rewrite files with the ruff formatter and apply safe fixes"
	@echo "  typecheck    Run mypy in strict mode"
	@echo "  test         Run the test suite"
	@echo "  check        Run everything CI runs (lint, typecheck, test)"
	@echo "  clean        Remove caches and build artifacts"
	@echo "  bench        Full benchmark sweep, writes a JSON artifact"
	@echo "  bench-fast   Reduced sweep for CI"
	@echo "  figures      Regenerate figures from the committed artifact"

sync:
	$(UV) sync

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# `bench` regenerates every number in README.md and RESULTS.md from scratch.
# `bench-fast` runs the same code path on a reduced sweep, fast enough to gate CI.
bench:
	$(UV) run foldq bench

bench-fast:
	$(UV) run foldq bench --fast

figures:
	$(UV) run foldq figures
