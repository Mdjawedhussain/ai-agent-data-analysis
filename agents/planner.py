"""Agent 1 (MAIN): Planner. Decides what the team must do and re-plans after failures."""
from utils import extract_json


SYSTEM = """[PLANNER] You are the lead agent of a data-analysis team.
Given the user's question and database schema, produce a plan.

Return ONLY a JSON object with these keys:
  "sql_task": string - a precise description of one SQLite query,
  "use_stats": boolean,
  "use_viz": boolean,
  "chart_hint": string,
  "reasoning": string - one sentence.

Rules:
- Inspect the schema before planning.
- Only request columns that exist in the schema.
- Never assume a missing column exists.
- Do not treat sales revenue as profit.
- Profit normally requires profit data or suitable cost data.
- If a requested metric cannot be calculated, explain this in reasoning.
- Prefer simple queries that answer the user's actual question.
- Enable visualization only when the query can produce useful chart data.
"""



def _b(v, default=True):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return default if v is None else bool(v)


def make_plan(question: str, schema: str, llm, feedback: str = "") -> dict:
    prompt = f"Database schema:\n{schema}\n\nUser question: {question}\n"
    if feedback:
        prompt += (f"\nThe previous attempt failed: {feedback}\n"
                   "Revise the plan: use a simpler or different approach.\n")
    try:
        d = extract_json(llm.complete(prompt, SYSTEM))
    except Exception:  # noqa: BLE001 - fall back to a safe default plan
        d = {}
    plan = {
        "sql_task": str(d.get("sql_task") or question),
        "use_stats": _b(d.get("use_stats")),
        "use_viz": _b(d.get("use_viz")),
        "chart_hint": str(d.get("chart_hint") or ""),
        "reasoning": str(d.get("reasoning") or ""),
    }
    plan["steps"] = (["sql"] + (["stats"] if plan["use_stats"] else [])
                     + (["viz"] if plan["use_viz"] else []) + ["report", "critic"])
    return plan
