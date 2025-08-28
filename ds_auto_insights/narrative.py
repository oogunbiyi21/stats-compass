# ds_auto_insights/narrative.py
from typing import Dict, Any, List, Optional
import os
import json
import streamlit as st

try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

SYSTEM = (
    "You write brief, executive-ready insight bullets from provided analysis artifacts. "
    "Return 3–5 bullets max. Avoid hedging. Be concrete with numbers when present. "
    "If artifacts are sparse, say what’s most actionable next."
)

def _client() -> Optional[OpenAI]:
    if not OPENAI_OK:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

def generate_narrative(artifacts: Dict[str, Any], user_question: str = "", context: Dict[str, Any] = None) -> str:
    context = context or {}
    client = _client()
    # Fallback: simple heuristic if no API
    if client is None:
        return "- Data profiled; consider segmenting by key categories.\n- Check top correlations for likely drivers.\n- Create a group-by view on your primary segment."

    # Trim artifacts to essentials for prompt size
    compact = {
        "plan": artifacts.get("plan"),
        "steps": artifacts.get("steps", [])[:5],
        "context": context,
        "question": user_question,
    }
    compact_json = json.dumps(compact, ensure_ascii=False)[:8000]  # guard token size

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Artifacts JSON:\n{compact_json}"},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        return txt
    except Exception as e:
        st.warning(f"Narrative API error; showing heuristic. {type(e).__name__}: {e}")
        return "- Data profiled; segment by key categories.\n- Inspect top correlations for drivers.\n- Group by your main segment for a ranked view."
