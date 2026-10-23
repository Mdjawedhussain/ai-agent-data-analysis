"""Evaluate the multi-agent system against the single-agent baseline.

    python -m eval.run_eval                 # all questions, both systems
    python -m eval.run_eval --limit 5       # quick smoke test
    python -m eval.run_eval --sleep 6       # slow down to respect free-tier rate limits

Outputs eval/results.csv and eval/results.md (paste the table into your report).
"""
import argparse
import json
import re
import time

import pandas as pd

from agents import baseline
from db import DB_PATH
from graph import run_pipeline
from llm import get_llm
from utils import contains_number

YEAR = re.compile(r"^(19|20)\d{2}$")


def is_correct(expected: str, text: str) -> bool:
    text = text or ""
    try:
        val = float(expected)
        if YEAR.match(expected):  # years are labels, not magnitudes: need the exact token
            return re.search(rf"(?<!\d){expected}(?!\d)", text) is not None
        decimals = len(expected.split(".")[1]) if "." in expected else 0
        return contains_number(text, val, decimals)
    except ValueError:
        return expected.lower() in text.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between questions (free-tier limits)")
    ap.add_argument("--no-critic", action="store_true")
    args = ap.parse_args()

    qs = json.load(open(args.questions))[: args.limit]
    llm = get_llm()
    rows = []
    for q in qs:
        # multi-agent system (auto-approves the plan)
        t0, c0 = time.time(), llm.calls
        s = run_pipeline(q["question"], llm, DB_PATH, approve=lambda p: True, use_critic=not args.no_critic)
        text = s["report"] + " " + (s["df"].to_csv(index=False) if s.get("df") is not None else "")
        rows.append({"id": q["id"], "system": "multi-agent", "correct": is_correct(q["answer"], s["report"]) or
                     is_correct(q["answer"], text), "seconds": round(time.time() - t0, 1),
                     "llm_calls": llm.calls - c0, "replans": s.get("replans", 0),
                     "critic_rejections": s.get("revisions", 0), "status": s["status"]})
        time.sleep(args.sleep)
        # single-agent baseline
        t0, c0 = time.time(), llm.calls
        b = baseline.run(q["question"], llm, DB_PATH)
        rows.append({"id": q["id"], "system": "single-agent", "correct": is_correct(q["answer"], b["report"]),
                     "seconds": round(time.time() - t0, 1), "llm_calls": llm.calls - c0,
                     "replans": 0, "critic_rejections": 0, "status": b["status"]})
        time.sleep(args.sleep)
        print(f"Q{q['id']}: multi={rows[-2]['correct']} single={rows[-1]['correct']}")

    df = pd.DataFrame(rows)
    df.to_csv("eval/results.csv", index=False)
    summary = df.groupby("system").agg(accuracy=("correct", "mean"), avg_seconds=("seconds", "mean"),
                                       avg_llm_calls=("llm_calls", "mean"), replans=("replans", "sum"),
                                       critic_rejections=("critic_rejections", "sum")).round(2)
    md = summary.to_markdown() if hasattr(summary, "to_markdown") and _has_tabulate() else summary.to_string()
    open("eval/results.md", "w").write(md + "\n")
    print("\n" + md)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
