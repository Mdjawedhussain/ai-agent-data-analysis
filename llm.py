"""LLM access. Switch providers using an environment variable.

LLM_PROVIDER=groq -> Groq API (requires GROQ_API_KEY)
LLM_PROVIDER=fake -> Scripted LLM for tests
"""

import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


class LLM:
    """Base class. Subclasses implement _complete(). `calls` counts LLM usage."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, system: str = "") -> str:
        self.calls += 1
        return self._complete(prompt, system)

    def _complete(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError


def _retry(fn, tries=5):
    """Exponential backoff for API rate limits."""
    delay = 2

    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            limited = (
                "429" in msg
                or "RESOURCE_EXHAUSTED" in msg
                or "rate_limit" in msg.lower()
                or "rate limit" in msg.lower()
            )

            if not limited or i == tries - 1:
                raise

            time.sleep(delay)
            delay *= 2


class GroqLLM(LLM):
    def __init__(self, model=None):
        super().__init__()

        # Old default: llama-3.3-70b-versatile (retired by Groq on 16 Aug 2026)
        self.client = ChatGroq(
            model=model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def _complete(self, prompt, system=""):
        msgs = (
            [SystemMessage(content=system)] if system else []
        ) + [HumanMessage(content=prompt)]

        response = _retry(lambda: self.client.invoke(msgs))
        content = response.content

        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )

        return content


class FakeLLM(LLM):
    """Scripted LLM for tests: responder(prompt, system) -> str."""

    def __init__(self, responder):
        super().__init__()
        self.responder = responder

    def _complete(self, prompt, system=""):
        return self.responder(prompt, system)


def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )
        return GroqLLM()

    if provider == "fake":
        raise ValueError(
            "Use FakeLLM(responder) directly when running tests."
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")