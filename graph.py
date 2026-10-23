r"""Orchestrator: runs the agents as a graph with shared state, retries and a critic loop.

   user question
        |
     PLANNER  <------------- re-plan (if SQL fails / empty)
        |
   [human approval]
        |
      SQL  -> STATS -> VIZ -> REPORT <--(rewrite)-- CRITIC
                                  \_______________/
                                          |
                                     final answer
"""
import json
import os
import time
from datetime import datetime

from agents import critic_agent, planner, report_agent, sql_agent, stats_agent, viz_agent
from db import DB_PATH, get_schema

MAX_REPLANS = 2      # planner re-plans after a failed/empty SQL step
MAX_REVISIONS = 2    # report rewrites after the critic rejects


def _trace(state, agent, inp, out, t0):
    state["trace"].append({"agent": agent, "input": inp, "output": out,
                           "seconds": round(time.time() - t0, 2)})


def create_plan(question: str, llm, db_path: str = DB_PATH) -> dict:
    """Phase 1 - Planner only. The UI shows this to the human for approval."""
    t0 = time.time()
    plan = planner.make_plan(question, get_schema(db_path), llm)
    return plan, [{"agent": "planner", "input": question, "output": plan, "seconds": round(time.time() - t0, 2)}]


def execute_plan(question: str, plan: dict, llm, db_path: str = DB_PATH, use_critic: bool = True,
                 trace=None, log_dir: str = "logs") -> dict:
    """Phase 2 - run the helper agents on an (approved) plan."""
    start, calls0 = time.time(), llm.calls
    schema = get_schema(db_path)
    state = {"question": question, "plan": plan, "trace": list(trace or []), "status": "running",
             "sql": "", "df": None, "stats": None, "viz": None, "report": "", "critic": None,
             "replans": 0, "revisions": 0}

    # --- SQL step with planner re-planning on failure -------------------------------------
    for attempt in range(MAX_REPLANS + 1):
        t0 = time.time()
        res = sql_agent.run(state["plan"]["sql_task"], schema, llm, db_path)
        _trace(state, "sql", state["plan"]["sql_task"],
               {"sql": res["sql"], "ok": res["ok"], "attempts": res["attempts"],
                "rows": None if res["df"] is None else len(res["df"])}, t0)
        if res["ok"] and not res["df"].empty:
            break
        if attempt == MAX_REPLANS:
            state.update(status="failed", report="I could not retrieve data to answer this question.")
            return _finish(state, start, calls0, llm)
        t0 = time.time()
        feedback = res.get("error") or "the query returned no rows"
        state["plan"] = planner.make_plan(question, schema, llm, feedback=feedback)
        state["replans"] += 1
        _trace(state, "planner (re-plan)", feedback, state["plan"], t0)
    state["sql"], state["df"] = res["sql"], res["df"]

    # --- Stats + visualization -----------------------------------------------------------------
    if "stats" in state["plan"]["steps"]:
        t0 = time.time()
        state["stats"] = stats_agent.run(state["df"])
        _trace(state, "stats", f"{len(state['df'])} rows", state["stats"], t0)
    if "viz" in state["plan"]["steps"]:
        t0 = time.time()
        state["viz"] = viz_agent.run(state["df"], state["plan"]["chart_hint"], llm)
        _trace(state, "viz", state["plan"]["chart_hint"], state["viz"]["spec"], t0)

    # --- Report <-> Critic loop ----------------------------------------------------------------
    facts = (state["stats"] or {}).get("facts", {})
    feedback = ""
    for rev in range(MAX_REVISIONS + 1):
        t0 = time.time()
        state["report"] = report_agent.run(question, state["sql"], state["df"], state["stats"], llm, feedback)
        _trace(state, "report", feedback or "first draft", state["report"], t0)
        if not use_critic:
            break
        t0 = time.time()
        state["critic"] = critic_agent.check(state["report"], state["df"], facts, llm)
        _trace(state, "critic", "verify numbers", state["critic"], t0)
        if state["critic"]["ok"]:
            break
        if rev < MAX_REVISIONS:
            state["revisions"] += 1
            feedback = "\n".join(state["critic"]["issues"])
    state["status"] = "done" if (not state["critic"] or state["critic"]["ok"]) else "done_with_warnings"
    return _finish(state, start, calls0, llm)


def _finish(state, start, calls0, llm):
    state["seconds"] = round(time.time() - start, 2)
    state["llm_calls"] = llm.calls - calls0
    return state


def run_pipeline(question: str, llm, db_path: str = DB_PATH, approve=None, use_critic: bool = True,
                 log_dir: str = "logs") -> dict:
    """Convenience wrapper: plan -> (optional human approval) -> execute. approve(plan)->bool."""
    calls0 = llm.calls
    plan, trace = create_plan(question, llm, db_path)
    if approve is not None and not approve(plan):
        return {"question": question, "plan": plan, "trace": trace, "status": "rejected", "report": "",
                "df": None, "stats": None, "viz": None, "critic": None, "sql": "", "llm_calls": llm.calls - calls0}
    state = execute_plan(question, plan, llm, db_path, use_critic, trace, log_dir)
    state["llm_calls"] = llm.calls - calls0
    save_log(state, log_dir)
    return state


def save_log(state: dict, log_dir: str = "logs") -> str:
    """Write the full agent trace as JSON (great screenshots/evidence for the report)."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
    slim = {k: v for k, v in state.items() if k not in ("df", "viz")}
    slim["result_preview"] = None if state.get("df") is None else state["df"].head(20).to_dict("records")
    slim["chart_spec"] = (state.get("viz") or {}).get("spec")
    with open(path, "w") as f:
        json.dump(slim, f, indent=2, default=str)
    return path
