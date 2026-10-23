"""Agent 3: Stats agent. Deterministic pandas tool: computes verifiable facts from the SQL result."""
import re

import pandas as pd

TIME_LIKE = re.compile(r"year|month|date|quarter|week|day", re.I)


def _f(x, nd=2):
    return round(float(x), nd)


def run(df: pd.DataFrame) -> dict:
    facts, lines = {"row_count": int(len(df))}, [f"The result has {len(df)} rows."]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    for c in num_cols[:3]:
        s = df[c].dropna()
        if s.empty:
            continue
        facts[c] = {"sum": _f(s.sum()), "mean": _f(s.mean()), "min": _f(s.min()), "max": _f(s.max())}
        lines.append(f"{c}: sum={facts[c]['sum']}, mean={facts[c]['mean']}, "
                     f"min={facts[c]['min']}, max={facts[c]['max']}.")

    if cat_cols and num_cols and len(df) > 1:
        label, val = cat_cols[0], num_cols[0]
        ranked = df[[label, val]].dropna().sort_values(val, ascending=False)
        if len(ranked) > 1:
            top, bottom = ranked.iloc[0], ranked.iloc[-1]
            facts["top"] = {"label": str(top[label]), "value": _f(top[val])}
            facts["bottom"] = {"label": str(bottom[label]), "value": _f(bottom[val])}
            lines.append(f"Highest {val}: {top[label]} ({_f(top[val])}). Lowest: {bottom[label]} ({_f(bottom[val])}).")
            total = ranked[val].sum()
            if total:
                facts["top_share_pct"] = _f(100 * top[val] / total, 1)
                lines.append(f"{top[label]} is {facts['top_share_pct']}% of the total {val}.")
        if TIME_LIKE.search(label):
            ordered = df[[label, val]].dropna().sort_values(label)
            first, last = ordered.iloc[0][val], ordered.iloc[-1][val]
            if first:
                facts["change_pct"] = _f(100 * (last - first) / abs(first), 1)
                lines.append(f"From the first to the last {label} the {val} changed by {facts['change_pct']}%.")
    return {"facts": facts, "summary": lines}
