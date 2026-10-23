"""Agent 4: Visualization agent. The LLM picks a chart spec; code validates it and draws it with Plotly."""
import re

import pandas as pd

from utils import extract_json

SYSTEM = """[VIZ] You choose a chart for a query result. Return ONLY JSON:
{"chart": "bar" | "line" | "pie" | "none", "x": "<column>", "y": "<numeric column>", "title": "<short title>"}
Use line for time series, bar for category comparisons, pie only for shares of a whole with <= 6 slices."""
TIME_LIKE = re.compile(r"year|month|date|quarter|week|day", re.I)


def _fallback(df):
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in df.columns if c not in nums]
    if not nums:
        return {"chart": "none"}
    x = cats[0] if cats else df.columns[0]
    y = next((c for c in nums if c != x), nums[0])
    return {"chart": "line" if TIME_LIKE.search(str(x)) else "bar", "x": x, "y": y, "title": f"{y} by {x}"}


def build_figure(df, spec):
    try:
        import plotly.express as px
    except ImportError:
        return None
    d = df.head(50)
    kind = spec["chart"]
    if kind == "line":
        return px.line(d.sort_values(spec["x"]), x=spec["x"], y=spec["y"], title=spec.get("title"), markers=True)
    if kind == "pie":
        return px.pie(d, names=spec["x"], values=spec["y"], title=spec.get("title"))
    return px.bar(d, x=spec["x"], y=spec["y"], title=spec.get("title"))


def run(df: pd.DataFrame, hint: str, llm) -> dict:
    if df is None or df.empty or len(df) < 2 or len(df.columns) < 2:
        return {"spec": {"chart": "none"}, "fig": None}
    prompt = (f"Columns and types: {dict(df.dtypes.astype(str))}\nFirst rows:\n{df.head(5).to_csv(index=False)}\n"
              f"Hint from the planner: {hint or 'none'}\n")
    try:
        spec = extract_json(llm.complete(prompt, SYSTEM))
        ok = (spec.get("chart") in ("bar", "line", "pie", "none")
              and (spec["chart"] == "none" or (spec.get("x") in df.columns and spec.get("y") in df.columns
                                                and pd.api.types.is_numeric_dtype(df[spec["y"]]))))
        if not ok:
            raise ValueError("invalid chart spec")
    except Exception:  # noqa: BLE001
        spec = _fallback(df)
    if spec["chart"] == "none":
        return {"spec": spec, "fig": None}
    return {"spec": spec, "fig": build_figure(df, spec)}
