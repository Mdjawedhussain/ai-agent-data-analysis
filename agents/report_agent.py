"""Agent 5: Report agent. Writes the answer using ONLY numbers that appear in the data."""
SYSTEM = """[REPORT] You are a business analyst writing for a manager.
Answer the question in 3-6 sentences. Start with the direct answer.
STRICT RULES: use ONLY numbers that appear in the data or statistics provided; never compute new numbers,
never invent facts, and mention the exact names/labels from the data. Plain text, no markdown tables."""


def run(question: str, sql: str, df, stats: dict, llm, feedback: str = "") -> str:
    prompt = (f"Question: {question}\n\nSQL used:\n{sql}\n\nQuery result (up to 30 rows):\n"
              f"{df.head(30).to_csv(index=False)}\n")
    if stats:
        prompt += "Verified statistics:\n" + "\n".join(stats.get("summary", [])) + "\n"
    if feedback:
        prompt += ("\nYour previous report was REJECTED by the critic for these reasons:\n"
                   f"{feedback}\nRewrite it and fix every issue.\n")
    return llm.complete(prompt, SYSTEM).strip()
