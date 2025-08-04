.PHONY: test lint format

test:
	pytest tests

lint:
	flake8 .

format:
	black .
