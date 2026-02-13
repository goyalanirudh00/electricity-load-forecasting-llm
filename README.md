LLM-Based Electricity Load Forecasting
Overview

This project implements a leak-free prototype for 24-hour ahead electricity load forecasting using a Large Language Model (LLM). Performance is evaluated using rolling-origin backtesting and compared against a strong seasonal-naive baseline.

The emphasis is on:

Methodological correctness

No data leakage

Deterministic feature construction

Clean evaluation logic

Reproducibility

Accuracy is reported transparently without hyperparameter tuning or prompt overfitting.

Problem Setup

Each utility dataset contains approximately 90 days of hourly electricity load data (2160 observations).

For selected forecast origin timestamps, the task is to predict the next 24 hourly values.

Evaluation

Evaluation uses:

14 daily rolling origins per utility

24-hour forecast horizon

336 hourly predictions per utility

Metrics

MAE

RMSE

MAPE

Metrics are computed across all hourly predictions and averaged across utilities.

Baseline Model

A seasonal-naive baseline is implemented:

Prediction at time T+h = actual load at time T+h−24

Given strong daily seasonality in the data, this baseline provides a realistic and competitive benchmark.

LLM Forecasting Methodology

At each forecast origin, the LLM receives structured inputs computed strictly from historical data (no future information):

Origin Summary (14-day window)

Recent level statistics

Trend (if significant)

Peak and trough hours

Weekend effect (if present)

Ramp constraints

Non-negativity constraint

Hour-of-Day Profile (14-day average)

24-value representation of typical daily shape

Recent History (7 days)

168 hourly observations provided chronologically

All context is recomputed independently at each origin to prevent leakage.

The LLM is instructed to output exactly 24 non-negative numeric forecasts in strict JSON format.

No external signals such as weather or news are used.

Leakage Prevention

For each origin time T:

All features are computed using data ≤ T

No future values are used in context construction

Forecasts are evaluated only against unseen future observations

Rolling windows are recomputed per origin

This ensures proper backtesting discipline.

Results

Example aggregate results (mean across utilities):

Baseline (Seasonal Naive)
MAPE ≈ 7.43%

LLM Forecaster
MAPE ≈ 7.55%

Interpretation

Seasonal naive is highly competitive due to strong daily seasonality

The LLM performs comparably overall

Performance varies slightly across utilities

Without exogenous context, improvements over seasonal naive are modest

These results are consistent with the dataset characteristics and the short 24-hour forecast horizon

Running the Project
Recommended: Run with Docker

Build the image:

docker build -t electricity-forecast .


Run the container:

docker run -e OPENAI_API_KEY=your_api_key_here electricity-forecast


The container will:

Load all utilities

Perform rolling-origin backtesting

Evaluate both baseline and LLM forecasts

Print formatted metric summaries

Optional model override:

docker run \
  -e OPENAI_API_KEY=your_api_key_here \
  -e LLM_FORECAST_MODEL=gpt-4o-mini \
  electricity-forecast

Optional: Run Locally (Without Docker)
Prerequisites

Python 3.10+

OpenAI API key

Install dependencies
pip install -r requirements.txt

Set API key

macOS / Linux:

export OPENAI_API_KEY=your_api_key_here


Windows (PowerShell):

setx OPENAI_API_KEY "your_api_key_here"

Run evaluation
python main.py
