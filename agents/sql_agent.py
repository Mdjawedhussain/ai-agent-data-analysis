
"""Agent 2: SQL agent. Generates, validates and executes read-only SQL."""
from db import run_query
from utils import extract_json

SYSTEM = """You are an expert SQLite analyst.
Return ONLY a JSON object with exactly one key: "sql".

Rules:
- Generate exactly one read-only SQLite SELECT query.
- Use only the table and columns explicitly listed in the supplied schema.
- Never invent columns, tables, or values.
- If the task requests a metric that cannot be calculated from the schema,
  do not substitute a different metric. Return {"sql": ""}.
- Revenue may be calculated as SUM(quantity * unit_price) only when
  both columns exist.
- Do not describe revenue as profit. Profit requires suitable cost data
  or an existing profit column.
- Use clear aliases for calculated columns.
- Do not include Markdown fences or explanations outside the JSON.
"""


def run(task: str, schema: str, llm, db_path: str,
        max_retries: int = 2) -> dict:
    attempts = []
    error = ""
    last_sql = ""

    # Detect requests for unavailable financial metrics.
    schema_lower = schema.lower()
    task_lower = task.lower()

    profit_requested = any(
        word in task_lower
        for word in ("profit", "net income", "margin")
    )
    profit_data_available = (
        "profit" in schema_lower
        or "cost" in schema_lower
        or "expense" in schema_lower
    )

    if profit_requested and not profit_data_available:
        return {
            "ok": False,
            "sql": "",
            "df": None,
            "error": (
                "Cannot calculate profit from the available schema: "
                "no profit, cost, or expense column is available. "
                "Ask for sales revenue instead if that is what you need."
            ),
            "attempts": [],
        }

    for attempt_no in range(max_retries + 1):
        prompt = (
            f"Database schema:\n{schema}\n\n"
            f"Task: {task}\n"
            "Return JSON with a non-empty SQL query using only "
            "the available columns."
        )

        if error:
            prompt += (
                f"\nPrevious SQL: {last_sql!r}\n"
                f"Previous error: {error}\n"
                "Correct the query. Return valid JSON only."
            )

        try:
            raw = llm.complete(prompt, SYSTEM)
            data = extract_json(raw)

            sql_value = data.get("sql")
            if not isinstance(sql_value, str) or not sql_value.strip():
                raise ValueError(
                    "The model returned missing or empty SQL. "
                    f"Response received: {raw[:500]!r}"
                )

            last_sql = sql_value.strip()
            df = run_query(last_sql, db_path)

            attempts.append({"sql": last_sql, "error": None})
            return {
                "ok": True,
                "sql": last_sql,
                "df": df,
                "attempts": attempts,
            }

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({
                "sql": last_sql,
                "error": error,
            })

    return {
        "ok": False,
        "sql": last_sql,
        "df": None,
        "error": error,
        "attempts": attempts,
    }
