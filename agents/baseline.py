"""Single-agent BASELINE for the evaluation: one generalist prompt, no planner, no retries, no critic."""
from db import get_schema, run_query
from utils import extract_json

SYSTEM = """You are a data analyst. Given a SQLite schema and a question, return ONLY JSON:
{"sql": "<one SELECT query>"}"""
ANSWER_SYSTEM = "You are a data analyst. Answer the question briefly using only the query result."


def run(question: str, llm, db_path: str) -> dict:
    schema = get_schema(db_path)
    try:
        sql = extract_json(llm.complete(f"Schema:\n{schema}\n\nQuestion: {question}", SYSTEM))["sql"]
        df = run_query(sql, db_path)
        answer = llm.complete(f"Question: {question}\nResult:\n{df.head(30).to_csv(index=False)}", ANSWER_SYSTEM)
        return {"status": "done", "sql": sql, "df": df, "report": answer.strip()}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "sql": "", "df": None, "report": f"ERROR: {e}"}
