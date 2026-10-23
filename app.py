"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from db import DB_PATH, get_schema
from graph import create_plan, execute_plan, save_log
from llm import get_llm

st.set_page_config(page_title="Agentic Data Analyst", layout="wide")
st.title("Agentic Data Analyst")
st.caption("Planner + SQL, Stats, Visualization, Report and Critic agents. You approve the plan before anything runs.")

if "llm" not in st.session_state:
    st.session_state.llm = get_llm()
llm = st.session_state.llm

with st.sidebar:
    use_critic = st.toggle("Enable Critic agent", value=True)
    with st.expander("Database schema"):
        st.code(get_schema(DB_PATH))

question = st.text_input("Ask a question about the data", "Which region has the highest total profit?")

if st.button("1. Create plan"):
    with st.spinner("Planner is thinking..."):
        plan, trace = create_plan(question, llm)
    st.session_state.update(plan=plan, trace=trace, question=question, result=None)

if st.session_state.get("plan"):
    st.subheader("Proposed plan (human approval step)")
    plan = st.session_state.plan
    st.write(plan["reasoning"])
    plan["sql_task"] = st.text_area("SQL task (you can edit it)", plan["sql_task"])
    st.write("Steps:", " -> ".join(plan["steps"]))
    if st.button("2. Approve and run"):
        with st.spinner("Agents are working..."):
            res = execute_plan(st.session_state.question, plan, llm, use_critic=use_critic,
                               trace=st.session_state.trace)
            save_log(res)
        st.session_state.result = res

res = st.session_state.get("result")
if res:
    st.subheader("Answer")
    st.write(res["report"])
    if res.get("status") == "failed":
        st.error("No data could be retrieved. Open 'Agent trace' below and look at the sql step for the exact error.")
    if res.get("critic"):
        (st.success if res["critic"]["ok"] else st.warning)(
            "Critic: all numbers verified against the data" if res["critic"]["ok"]
            else "Critic warnings: " + "; ".join(res["critic"]["issues"]))
    if res.get("viz") and res["viz"].get("fig") is not None:
        st.plotly_chart(res["viz"]["fig"], use_container_width=True)
    if res.get("df") is not None:
        st.dataframe(res["df"])
        st.code(res["sql"], language="sql")
    if res.get("stats"):
        with st.expander("Statistics"):
            st.write(res["stats"]["summary"])
    with st.expander("Agent trace"):
        for step in res["trace"]:
            st.markdown(f"**{step['agent']}** - {step['seconds']}s")
            out = step["output"]
            if isinstance(out, (dict, list)):
                st.json(out, expanded=False)
            else:
                st.text(str(out))
    st.caption(f"Re-plans: {res['replans']} | Report revisions: {res['revisions']} | Total {res['seconds']}s")