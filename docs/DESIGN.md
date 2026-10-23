# Agentic AI Data Analyst - Design (assignment template)

## 1. Solution Overview
The Agentic Data Analyst helps business users get trustworthy answers from tabular data using natural
language. A main Planner agent breaks each question into steps and coordinates five specialist agents
(SQL, Stats, Visualization, Report, Critic). The system works autonomously (query, analyze, chart, write,
self-check) but keeps a human in control: the user approves the plan before it runs.

## 2. Key Features
- **Feature 1 - Natural-language to verified answer:** question in, SQL + statistics + chart + written report out.
- **Feature 2 - Autonomous self-correction:** SQL errors trigger retries; empty/failed results trigger re-planning.
- **Feature 3 - Anti-hallucination Critic:** every number in the report is checked against the query result.
- **Feature 4 - Human-in-the-loop and audit trail:** plan approval/editing, read-only SQL, and a JSON trace of every agent step.

## 3. Data Sources
- **Data Source 1:** Sales dataset (Superstore-style CSV: orders, region, category, sales, profit, discount), loaded into SQLite.
- **Data Source 2:** Table schema and sample rows, read live from the database and given to the agents.
- **Data Source 3:** Benchmark question set (20 questions with known answers) used for evaluation.
- **Data Source 4:** Agent run logs (JSON traces) used for monitoring and error analysis.
- *Optional:* any public Kaggle CSV (Olist e-commerce, Telco churn) can replace the sample data.

## 4. AI Agents
- **Agent 1 - Planner (main):** interprets the question, creates the plan, delegates, and re-plans after failures.
- **Agent 2 - SQL Agent:** writes SQLite queries, validates them (read-only), runs them, self-corrects on errors.
- **Agent 3 - Stats Agent:** computes verified facts (totals, top/bottom, share, change %) with pandas.
- **Agent 4 - Visualization Agent:** chooses chart type and axes, draws the chart with Plotly.
- **Agent 5 - Report Agent:** writes the answer using only numbers present in the data.
- **Agent 6 - Critic Agent:** verifies the report; rejects it with feedback if a number is unsupported.

Pattern: supervisor (hub-and-spoke) with shared state and two feedback loops
(Planner re-plan loop, Report-Critic revision loop).

## 5. Reference Architecture
- **Processing Layer:** Python, pandas, SQLite (CSV ingestion, query execution, statistics).
- **Application Layer:** Streamlit web UI (question box, plan approval, charts, report, agent trace) and a CLI.
- **Security Layer:** API key in `.env` (git-ignored); SQL allow-list (single SELECT) plus read-only DB connection;
  human approval before execution; public data only (free-tier data may be used for model improvement).
- **Agent Layer:** Planner-led orchestrator in Python; LLM via Gemini free tier (or local Ollama); structured JSON
  outputs between agents; retry and critic loops.
- **Analytics Layer:** JSON run logs; evaluation script comparing multi-agent vs single-agent on accuracy,
  time, LLM calls, re-plans and critic rejections; Plotly charts for end users.
