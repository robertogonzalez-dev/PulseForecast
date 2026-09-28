FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PULSE_ARTIFACTS=/app/artifacts
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train at build time on synthetic data so the image is self-contained.
# For real data, mount artifacts at /app/artifacts instead.
RUN pulse-train --out /app/artifacts

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "pulse_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
