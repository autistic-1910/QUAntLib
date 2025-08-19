# Makefile for QuantLib development

.PHONY: install test lint format clean docs build upload

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml,viz]"

# Testing
test:
	pytest tests/ -v --cov=quantlib --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	flake8 quantlib tests
	mypy quantlib

format:
	black quantlib tests
	isort quantlib tests

format-check:
	black --check quantlib tests
	isort --check-only quantlib tests

# Documentation
docs:
	sphinx-build -b html docs docs/_build/html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Build and distribution
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Development
dev-setup: install-dev
	pre-commit install

run-example:
	python examples/basic_strategy.py

benchmark:
	python benchmarks/performance_test.py