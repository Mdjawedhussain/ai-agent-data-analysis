"""Command-line demo:  python cli.py "Which region has the highest total profit?" """
import sys

from graph import run_pipeline
from llm import get_llm

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which region has the highest total profit?"
    llm = get_llm()

    def approve(plan):
        print("\nPLAN:", plan["reasoning"], "\n SQL task:", plan["sql_task"], "\n Steps:", plan["steps"])
        return input("Approve? [y/n] ").strip().lower().startswith("y")

    s = run_pipeline(q, llm, approve=approve)
    print("\nSTATUS:", s["status"], "| LLM calls:", s["llm_calls"], "| seconds:", s.get("seconds"))
    print("SQL:", s["sql"])
    print("\nREPORT:\n", s["report"])
    if s.get("critic"):
        print("\nCRITIC:", s["critic"])
