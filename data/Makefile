.PHONY: install install-dev test lint format clean docs docker

# Instalação
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

install-minimal:
	pip install -r requirements-minimal.txt
	pip install -e .

# Testes
test:
	pytest tests/ -v --cov=eat_lab --cov-report=html

test-fast:
	pytest tests/ -v -x --ff

# Qualidade de código
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Limpeza
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentação
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server

# Docker
docker-build:
	docker build -t eat-lab:latest .

docker-run:
	docker run -p 8888:8888 -v $(PWD):/app eat-lab:latest

# Release
build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*

# Utilidades
check-deps:
	pip list --outdated

update-deps:
	pip install --upgrade -r requirements.txt

notebook:
	jupyter notebook notebooks/
