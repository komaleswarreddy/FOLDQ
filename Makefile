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
	@echo "  bench        (M6) Full benchmark sweep - not implemented yet"
	@echo "  bench-fast   (M6) Reduced sweep for CI - not implemented yet"
	@echo "  figures      (M6) Regenerate figures from artifacts - not implemented yet"

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

# The three targets below are declared so the command interface is stable, but they
# are not implemented yet. They exit non-zero on purpose: a task that prints nothing
# and returns success looks like a benchmark that ran, which is what Guardrail 1
# exists to prevent.
bench:
	@echo "make bench is not implemented until milestone M6. See PROJECT_BRIEF.md." >&2
	@exit 1

bench-fast:
	@echo "make bench-fast is not implemented until milestone M6. See PROJECT_BRIEF.md." >&2
	@exit 1

figures:
	@echo "make figures is not implemented until milestone M6. See PROJECT_BRIEF.md." >&2
	@exit 1
