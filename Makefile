# POETRY

download-poetry:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

install:
	poetry lock -n
	poetry install -n

check-safety:
	poetry check

check-style:
	poetry run black --config pyproject.toml --diff --check ./
	poetry run darglint -v 2 **/*.py
	poetry run isort --settings-path pyproject.toml --check-only **/*.py
	poetry run mypy --config-file setup.cfg python-template tests/**/*.py

codestyle:
	poetry run isort --settings-path pyproject.toml **/*.py
	poetry run black --config pyproject.toml ./

tests:
	docker compose up tests

build:
	docker compose build

down:
	docker compose down

api:
	docker compose up api

# PYTHON
run:
	poetry run python main.py


tokenize:
	python src/tokenizer/rules_tokenizer.py