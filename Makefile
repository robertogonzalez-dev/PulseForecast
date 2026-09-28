.PHONY: install train serve test lint docker

install:
	pip install -e ".[dev,tracking]"
train:
	pulse-train
serve:
	uvicorn pulse_forecast.api:app --reload
test:
	pytest
lint:
	ruff check . && ruff format --check .
docker:
	docker build -t pulse-forecast . && docker run --rm -p 8000:8000 pulse-forecast
