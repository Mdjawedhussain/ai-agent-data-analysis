"""Check that report numbers are supported by the query results.

The deterministic check always runs. Set ``CRITIC_USE_LLM=1`` to also run an
optional LLM consistency check (this costs one additional LLM call per question).

To see the debug messages, configure logging in the application entry point:
``logging.basicConfig(level=logging.DEBUG)``.
"""
import logging
import os

import pandas as pd

from utils import extract_json, close, parse_numbers

logger = logging.getLogger(__name__)

LLM_SYSTEM = """[CRITIC] You check a report against query results. Return ONLY JSON:
{"ok": true|false, "issues": ["..."]}. Mark ok=false only if the report states something the data contradicts
or does not support."""


def _add_numbers_from_dict(value, numbers):
    """Add numeric values nested in dictionaries to ``numbers``."""
    if isinstance(value, dict):
        for child in value.values():
            _add_numbers_from_dict(child, numbers)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers.append(float(value))


def allowed_numbers(df: pd.DataFrame, facts: dict):
    """Collect data values that are allowed to appear in the report."""
    numbers = []
    numeric_columns = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
            numbers.extend(float(value) for value in df[column].dropna().tolist())

    _add_numbers_from_dict(facts or {}, numbers)
    numbers.append(float(len(df)))

    # Reports may express a fraction as a percentage, or a percentage as a fraction.
    numbers_with_percentages = numbers + [number * 100 for number in numbers]
    logger.debug(
        "Built allowed numbers: rows=%d, numeric_columns=%s, values=%d",
        len(df),
        numeric_columns,
        len(numbers_with_percentages),
    )
    return numbers_with_percentages


def check(report: str, df: pd.DataFrame, facts: dict, llm=None) -> dict:
    """Return whether the report's numbers are supported by the data and facts."""
    allowed = allowed_numbers(df, facts)
    issues = []
    report_numbers = parse_numbers(report)
    logger.debug(
        "Checking report: characters=%d, parsed_numbers=%d, llm_check_requested=%s",
        len(report),
        len(report_numbers),
        llm is not None and os.getenv("CRITIC_USE_LLM") == "1",
    )

    for number in report_numbers:
        value = number["value"]
        is_year = (
            float(value).is_integer()
            and 1900 <= abs(value) <= 2100
            and not number["has_suffix"]
            and not number["pct"]
        )
        is_small_integer = (
            float(value).is_integer()
            and abs(value) <= 10
            and not number["has_suffix"]
            and not number["pct"]
        )

        if is_year or is_small_integer:
            logger.debug("Ignoring common year or small integer: %s", number["raw"])
            continue

        is_allowed = any(
            close(value, allowed_value, number["unit"])
            or close(abs(value), abs(allowed_value), number["unit"])
            for allowed_value in allowed
        )
        logger.debug(
            "Report number %s is %s",
            number["raw"],
            "supported" if is_allowed else "not supported",
        )
        if not is_allowed:
            issues.append(
                f"The number '{number['raw']}' does not appear in the data or statistics."
            )

    if llm is not None and os.getenv("CRITIC_USE_LLM") == "1":
        logger.debug("Running optional LLM consistency check")
        try:
            response = extract_json(
                llm.complete(
                    f"Report:\n{report}\n\nData:\n{df.head(30).to_csv(index=False)}",
                    LLM_SYSTEM,
                )
            )
            if not response.get("ok", True):
                issues.extend(str(issue) for issue in response.get("issues", []))
        except Exception:  # noqa: BLE001
            logger.exception("Optional LLM consistency check failed")

    result = {"ok": not issues, "issues": issues}
    logger.debug("Critic result: ok=%s, issues=%d", result["ok"], len(issues))
    return result
