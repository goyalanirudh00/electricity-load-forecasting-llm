"""LLM-based forecaster: calls LLM API with origin summary and inputs."""

import json
import os
import re
from typing import List, Tuple

SYSTEM_MESSAGE = """You are a forecasting assistant. Produce accurate, conservative 24-hour point forecasts for hourly electricity load. Follow the output format exactly. Do not include explanations."""

USER_MESSAGE_TEMPLATE = """You are forecasting hourly electricity load for the next 24 hours.

You are given:

ORIGIN SUMMARY (deterministic features computed from history up to the forecast origin time):
{origin_summary}

HOUR-OF-DAY MEAN PROFILE (typical day shape, 24 values for hours 0–23, computed from the last 14 days up to the origin):
hod_profile: {hod_profile}

RECENT HISTORY (last 7 days of hourly observations up to and including the origin; each item is (timestamp_utc, load)):
recent_history: {recent_history}

Task:

Produce point forecasts for the next 24 hours immediately after the origin time.

Forecasts must be non-negative.

Use the provided hod_profile, recent_history, and ORIGIN SUMMARY to produce the most plausible 24-hour continuation of the time series.

Ensure hour-to-hour transitions are realistic relative to the historical variability described in the ORIGIN SUMMARY.

Do not use any information beyond the provided inputs.

Output format requirements:

Return JSON only.

The JSON must have exactly one key: "forecast".

"forecast" must be an array of exactly 24 numbers (floats).

Do not include timestamps, comments, or extra keys.

Return JSON now."""


def _format_hod_profile(hod_profile: List[float]) -> str:
    """Format hour-of-day profile for the prompt."""
    return str(hod_profile)


def _format_recent_history(recent_history: List[Tuple[str, float]]) -> str:
    """Format recent history for the prompt."""
    return str(recent_history)


def _build_messages(
    origin_summary: str,
    hod_profile: List[float],
    recent_history: List[Tuple[str, float]],
) -> List[dict]:
    """Build system and user messages for the LLM."""
    user_content = USER_MESSAGE_TEMPLATE.format(
        origin_summary=origin_summary,
        hod_profile=_format_hod_profile(hod_profile),
        recent_history=_format_recent_history(recent_history),
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrapper
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def _call_llm(
    origin_summary: str,
    hod_profile: List[float],
    recent_history: List[Tuple[str, float]],
    model: str,
    api_key: str,
) -> List[float]:
    """Call LLM API and return parsed forecast."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages = _build_messages(origin_summary, hod_profile, recent_history)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")

    data = _extract_json(content)
    if "forecast" not in data:
        raise ValueError(f"LLM response missing 'forecast' key: {data}")

    forecast = data["forecast"]
    if not isinstance(forecast, list):
        raise ValueError(f"'forecast' must be a list, got {type(forecast)}")
    if len(forecast) < 24:
        raise ValueError(f"'forecast' must have at least 24 values, got {len(forecast)}")
    forecast = forecast[:24]

    result: List[float] = []
    for i, v in enumerate(forecast):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"forecast[{i}] is not a number: {v}")
        result.append(max(0.0, f))

    return result


def forecast(
    origin_summary: str,
    hod_profile: List[float],
    recent_history: List[Tuple[str, float]],
) -> List[float]:
    """Produce 24-hour forecast via LLM.

    Calls the configured LLM API with the origin summary, HOD profile, and
    recent history. Parses JSON response and returns 24 non-negative floats.

    Environment variables:
        OPENAI_API_KEY: API key for OpenAI (required).
        LLM_FORECAST_MODEL: Model name (default: gpt-4o-mini).

    Args:
        origin_summary: Textual summary of system state at forecast origin.
        hod_profile: 24-value hour-of-day profile.
        recent_history: 168 (timestamp, load) pairs, chronologically ordered.

    Returns:
        List of 24 floats (hourly predictions), non-negative.

    Raises:
        ValueError: If API key missing, response invalid, or forecast malformed.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for LLM forecaster"
        )

    model = os.environ.get("LLM_FORECAST_MODEL", "gpt-4o-mini")

    if len(hod_profile) != 24:
        raise ValueError(f"hod_profile must have 24 values, got {len(hod_profile)}")

    return _call_llm(origin_summary, hod_profile, recent_history, model, api_key)
