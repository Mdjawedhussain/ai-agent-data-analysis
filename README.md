# Agentic Data Analyst (free, runs on a laptop)

A multi-agent system that answers business questions about a dataset. A **Planner** (main agent) coordinates
**SQL, Stats, Visualization, Report and Critic** agents. A human approves the plan before anything runs.

## Quick start (about 15 minutes)

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env
```

1. Get a free key at Google AI Studio and paste it into `.env` as `GOOGLE_API_KEY`.
   If the model name errors, set `GEMINI_MODEL` to a current Flash model listed in AI Studio.
   (No key? Install Ollama, run `ollama pull llama3.1:8b`, set `LLM_PROVIDER=ollama`.)
2. Create the sample data and load it:
   ```bash
   python data/make_sample_data.py
   python load_data.py data/sample_sales.csv
   ```
3. Run the offline tests (no key needed): `python -m unittest tests.test_pipeline -v`
4. Try one question: `python cli.py "Which region has the highest total profit?"`
5. Launch the UI: `streamlit run app.py`
6. Run the evaluation: `python -m eval.run_eval --limit 5` first, then without `--limit`.
   Use `--sleep 8` if you hit free-tier rate limits.

To use your own Kaggle CSV: `python load_data.py path/to/file.csv`, then write your own
`eval/questions.json` (same format: id, question, answer).

## Project layout

```
agents/planner.py       main agent: plans, re-plans after failures
agents/sql_agent.py     writes + validates + runs read-only SQL, self-corrects on errors
agents/stats_agent.py   pandas tool: verified facts (top/bottom, share, change %)
agents/viz_agent.py     picks a chart spec (LLM) and draws it (Plotly)
agents/report_agent.py  writes the answer using only numbers from the data
agents/critic_agent.py  checks every number in the report against the data; sends it back if wrong
agents/baseline.py      single-agent baseline for the evaluation
graph.py                orchestrator: shared state, retries, critic loop, JSON trace logs
app.py / cli.py         Streamlit UI and command-line demo
eval/run_eval.py        multi-agent vs single-agent evaluation
tests/                  offline tests with a scripted fake LLM
docs/DESIGN.md          your assignment template, already filled in
docs/REPORT_TEMPLATE.md report skeleton
```

## How the agents cooperate

1. Planner reads the question + schema and writes a plan. **You approve or edit it.**
2. SQL agent runs the query. On an error it retries with the error message; if it still fails or returns
   no rows, the Planner **re-plans** (max 2 times).
3. Stats and Viz agents work on the result (shared state).
4. Report agent writes the answer. Critic verifies every number. If any is unsupported, the report is
   **rewritten with the critic's feedback** (max 2 times).
5. Every step is logged to `logs/run_*.json` (use these for screenshots).

## Safety features (mention these in your report)

- SQL is validated (single SELECT only, forbidden keywords blocked) and run on a **read-only** connection.
- Human-in-the-loop approval before execution.
- Critic agent reduces hallucinated numbers.
- Free-tier data may be used by Google to improve models: use public data only.

## Honest notes

- The orchestrator is plain Python, not LangGraph. It is the same graph idea with fewer dependencies;
  porting to LangGraph is a good "future work" item.
- The code was tested offline with a scripted fake LLM. Real Gemini answers will vary, so run the
  evaluation yourself and report your own numbers.
