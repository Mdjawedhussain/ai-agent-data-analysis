# Report skeleton (fill the [brackets] after you run the evaluation)

1. **Introduction** - Business users need quick, trustworthy answers from data without writing SQL. [1 paragraph]
2. **Why an agentic approach** - A single prompt mixes planning, SQL, analysis and writing, and it cannot check
   itself. Specialist agents with a planner and a critic can retry, re-plan and verify. [compare with baseline]
3. **System design** - Paste the architecture diagram + `docs/DESIGN.md` (features, data, agents, layers).
   Explain the two feedback loops with a diagram.
4. **Implementation** - Stack (Python, SQLite, Gemini free tier, Streamlit, Plotly). Describe each agent's prompt
   and tool. Show a screenshot of the UI and one agent trace (`logs/run_*.json`).
5. **Evaluation** - Dataset: [name, rows]. 20 benchmark questions. Paste `eval/results.md`:
   accuracy [x%] vs baseline [y%]; avg time; LLM calls; re-plans [n]; critic rejections [n].
   Error analysis: pick 3-4 failures from `eval/results.csv`, explain the cause (ambiguous question, wrong join,
   rounding...) and what would fix it.
6. **Limitations, ethics and safety** - LLM errors, ambiguous questions, free-tier rate limits, public data only,
   read-only SQL, human approval, critic limits (it checks numbers, not reasoning).
7. **Conclusion and future work** - Port to LangGraph, add memory of past questions, multi-table joins,
   LLM-based critic (`CRITIC_USE_LLM=1`), a larger benchmark.
