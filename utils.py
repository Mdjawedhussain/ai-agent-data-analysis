"""Small helpers: robust JSON extraction and number parsing/matching."""
import json
import re

# ---------- JSON ----------
def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ``` fences and chatter)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM reply")
    return json.loads(text[start:end + 1])


# ---------- numbers ----------
NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(-?)\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|k\b|m\b|bn\b|million\b|thousand\b|billion\b)?",
    re.I,
)
MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


def parse_numbers(text: str):
    """Return [{value, pct, unit, decimals, has_suffix, raw}] for every number in text."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        sign, whole, frac, suffix = m.groups()
        decimals = len(frac) - 1 if frac else 0
        value = float(whole.replace(",", "") + (frac or ""))
        if sign:
            value = -value
        suf = (suffix or "").lower()
        mult = MULT.get(suf, 1.0)
        out.append({
            "value": value * mult,
            "pct": suf == "%",
            "unit": (10 ** -decimals) * mult,
            "decimals": decimals,
            "has_suffix": bool(suf),
            "raw": m.group(0).strip(),
        })
    return out


def close(x, a, unit_x=1.0, unit_a=0.0, rel=0.005):
    """x (as written, with rounding unit unit_x) matches reference a (rounded to unit_a)?"""
    tol = max(0.5 * unit_x, 0.5 * unit_a, rel * abs(a), 1e-9)
    return abs(x - a) <= tol


def contains_number(text: str, expected: float, expected_decimals: int = 0) -> bool:
    """Strict match used by the evaluation: only rounding differences are tolerated."""
    unit_a = 10 ** -expected_decimals
    for n in parse_numbers(text):
        if close(n["value"], expected, n["unit"], unit_a, rel=0):
            return True
        if n["pct"] and close(n["value"] / 100.0, expected, n["unit"] / 100.0, unit_a, rel=0):
            return True
    return False
