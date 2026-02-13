# Electricity load forecast pipeline - arm64
FROM --platform=linux/arm64 python:3.11-slim

ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY data.py context.py baseline.py metrics.py evaluator.py llm_forecaster.py logger.py main.py .

# Copy dataset
COPY Utility_1_data.csv Utility_2_data.csv Utility_3_data.csv .

# Run full pipeline (pass OPENAI_API_KEY at runtime via -e or --env-file .env)
ENTRYPOINT ["python", "main.py"]
