"""Offline tests: a scripted FakeLLM stands in for Gemini, so no API key or internet is needed.
Run:  python -m unittest tests.test_pipeline -v
"""
import json
import os
import tempfile
import unittest

import pandas as pd

import db
from agents import baseline, critic_agent, stats_agent
from eval.run_eval import is_correct
from graph import run_pipeline
from llm import FakeLLM

CSV = "data/sample_sales.csv"
GOOD_SQL = "SELECT region, ROUND(SUM(profit),2) AS total_profit FROM sales GROUP BY region ORDER BY total_profit DESC"


def make_db():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    db.load_csv(CSV, db_path=path)
    return path


def scripted(sql_replies, report_replies):
    sql_i, rep_i = iter(sql_replies), iter(report_replies)

    def responder(prompt, system):
        if "[PLANNER]" in system:
            return json.dumps({"sql_task": "total profit by region, highest first", "use_stats": True,
                               "use_viz": True, "chart_hint": "bar of profit by region", "reasoning": "group by region"})
        if "[SQL]" in system:
            return json.dumps({"sql": next(sql_i)})
        if "[VIZ]" in system:
            return json.dumps({"chart": "bar", "x": "region", "y": "total_profit", "title": "Profit by region"})
        if "[REPORT]" in system:
            return next(rep_i)
        return "{}"
    return FakeLLM(responder)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.path = make_db()
        self.top = pd.read_csv(CSV).groupby("region").profit.sum().round(2)

    def test_happy_path(self):
        label, val = self.top.idxmax(), self.top.max()
        llm = scripted([GOOD_SQL], [f"{label} has the highest total profit at {val:,.2f}."])
        s = run_pipeline("Which region has the highest total profit?", llm, self.path, approve=lambda p: True,
                         log_dir=tempfile.mkdtemp())
        self.assertEqual(s["status"], "done")
        self.assertTrue(s["critic"]["ok"])
        self.assertEqual(s["df"].iloc[0]["region"], label)
        self.assertEqual([t["agent"] for t in s["trace"]][:2], ["planner", "sql"])

    def test_sql_self_correction(self):
        llm = scripted(["SELECT regionn FROM sales", GOOD_SQL], [f"{self.top.idxmax()} leads with {self.top.max():,.2f}."])
        s = run_pipeline("q", llm, self.path, approve=lambda p: True, log_dir=tempfile.mkdtemp())
        self.assertEqual(s["status"], "done")
        sql_trace = [t for t in s["trace"] if t["agent"] == "sql"][0]["output"]
        self.assertEqual(len(sql_trace["attempts"]), 2)
        self.assertIsNotNone(sql_trace["attempts"][0]["error"])

    def test_critic_rejects_hallucinated_number_and_report_is_rewritten(self):
        good = f"{self.top.idxmax()} leads with {self.top.max():,.2f}."
        llm = scripted([GOOD_SQL], [f"{self.top.idxmax()} leads with 999,999.99.", good])
        s = run_pipeline("q", llm, self.path, approve=lambda p: True, log_dir=tempfile.mkdtemp())
        self.assertEqual(s["revisions"], 1)
        self.assertTrue(s["critic"]["ok"])
        self.assertIn(f"{self.top.max():,.2f}", s["report"])

    def test_empty_result_triggers_replan(self):
        empty = "SELECT region, SUM(profit) AS total_profit FROM sales WHERE region='Mars' GROUP BY region"
        llm = scripted([empty, GOOD_SQL], [f"{self.top.idxmax()} leads with {self.top.max():,.2f}."])
        s = run_pipeline("q", llm, self.path, approve=lambda p: True, log_dir=tempfile.mkdtemp())
        self.assertEqual(s["replans"], 1)
        self.assertEqual(s["status"], "done")

    def test_human_can_reject_plan(self):
        llm = scripted([GOOD_SQL], ["x"])
        s = run_pipeline("q", llm, self.path, approve=lambda p: False, log_dir=tempfile.mkdtemp())
        self.assertEqual(s["status"], "rejected")
        self.assertEqual(llm.calls, 1)  # only the planner ran

    def test_baseline(self):
        def r(prompt, system):
            return json.dumps({"sql": GOOD_SQL}) if "SELECT query" in system else "ok answer"
        b = baseline.run("q", FakeLLM(r), self.path)
        self.assertEqual(b["status"], "done")


class SafetyTests(unittest.TestCase):
    def test_sql_validation(self):
        for bad in ["DROP TABLE sales", "DELETE FROM sales", "SELECT 1; DROP TABLE sales", "UPDATE sales SET sales=0", ""]:
            with self.assertRaises(ValueError):
                db.validate_sql(bad)
        self.assertEqual(db.validate_sql("SELECT * FROM sales;"), "SELECT * FROM sales")

    def test_read_only_connection(self):
        import sqlite3
        path = make_db()
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("INSERT INTO sales (order_id) VALUES ('x')")
        con.close()


class HelperTests(unittest.TestCase):
    def test_stats(self):
        df = pd.DataFrame({"region": ["A", "B", "C"], "profit": [100.0, 50.0, 25.0]})
        st = stats_agent.run(df)["facts"]
        self.assertEqual(st["top"]["label"], "A")
        self.assertEqual(st["top_share_pct"], 57.1)

    def test_critic_number_formats(self):
        df = pd.DataFrame({"region": ["West"], "profit": [1234567.0]})
        ok = lambda t: critic_agent.check(t, df, {})["ok"]
        self.assertTrue(ok("West earned $1,234,567 in profit."))
        self.assertTrue(ok("West earned about 1.2M in profit."))
        self.assertTrue(ok("In 2018, West earned $1,234,567."))      # years ignored
        self.assertFalse(ok("West earned $2,000,000 in profit."))     # hallucination

    def test_eval_matching(self):
        self.assertTrue(is_correct("West", "The West region leads."))
        self.assertTrue(is_correct("251945", "Total sales were $251,945.37."))
        self.assertTrue(is_correct("251945", "Total sales were about $251.9K."))
        self.assertFalse(is_correct("750", "There were 747 orders."))
        self.assertTrue(is_correct("0.12", "The average discount is 0.1203."))
        self.assertTrue(is_correct("27.6", "East is 27.6% of sales."))
        self.assertTrue(is_correct("2018", "2018 was the best year."))
        self.assertFalse(is_correct("2018", "2015 was the best year."))


if __name__ == "__main__":
    unittest.main()
