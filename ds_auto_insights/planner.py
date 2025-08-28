# ds_auto_insights/planner.py
from typing import Any, Dict, List, Optional
import json
import os

import pandas as pd
import streamlit as st

from .util import (
    summarise_dataset,
    key_trends_numeric_only,
)

# Optional OpenAI (guarded import)
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False


# ---------- Allowed operations & simple validators ----------
ALLOWED_OPS = {"summarise", "correlation", "histogram", "top_categories", "groupby_aggregate"}

def validate_plan(plan: Dict[str, Any]) -> Optional[str]:
    """Return error message if invalid, else None."""
    if not isinstance(plan, dict):
        return "Plan must be a JSON object."
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return "Plan must contain a non-empty 'steps' list."
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return f"Step {i} must be an object."
        op = step.get("op")
        if op not in ALLOWED_OPS:
            return f"Step {i}: op '{op}' is not allowed."
        # Basic arg checks
        if op in {"histogram", "top_categories"}:
            if "column" not in step:
                return f"Step {i}: '{op}' requires 'column'."
        if op == "groupby_aggregate":
            for req in ("group_col", "agg_col", "agg_fn"):
                if req not in step:
                    return f"Step {i}: '{op}' requires '{req}'."
    return None


# ---------- LLM planner ----------
PLANNER_SYSTEM = (
    "You are a data analysis planner. Generate a short JSON plan to answer the user's question. "
    "You can only use these operations in 'steps': "
    "summarise, correlation, histogram, top_categories, groupby_aggregate. "
    "Do not include explanations, only JSON. "
    "Schema: {\"steps\":[{\"op\":\"summarise\"}|"
    "{\"op\":\"correlation\"}|"
    "{\"op\":\"histogram\",\"column\":\"<col>\"}|"
    "{\"op\":\"top_categories\",\"column\":\"<col>\",\"n\":15}|"
    "{\"op\":\"groupby_aggregate\",\"group_col\":\"<col>\",\"agg_col\":\"<col>\",\"agg_fn\":\"sum|mean|count\"}]]}"
)

def _infer_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Cheap schema hints for the planner prompt."""
    numeric = df.select_dtypes(include="number").columns.tolist()[:12]
    categorical = df.select_dtypes(exclude="number").columns.tolist()[:12]
    return {"numeric": numeric, "categorical": categorical, "all": df.columns.tolist()[:30]}

def _extract_json_maybe(content: str) -> Dict[str, Any]:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        pass
    raise ValueError("Could not parse JSON plan")

def llm_make_plan(question: str, df: pd.DataFrame, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    context = context or {}
    hints = _infer_columns(df)
    fallback = {"steps": [{"op": "summarise"}]}

    # 1) openai client import / availability
    if not OPENAI_OK:   
        st.info("Planner fallback: OpenAI package not available.")
        return fallback

    # 2) api key presence
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.info("Planner fallback: OPENAI_API_KEY not set.")
        return fallback

    client = OpenAI(api_key=api_key)
    user_prompt = (
        f"Dataset columns (numeric): {hints['numeric']}\n"
        f"Dataset columns (categorical): {hints['categorical']}\n\n"
        f"Context notes (if any): {context}\n\n"
        f"User question: {question}\n"
        "Return ONLY the JSON plan per the schema."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
        try:
            plan = _extract_json_maybe(content)
        except Exception as e:
            st.warning(f"Planner returned non-JSON; using fallback. Raw: {content[:200]}...")
            return fallback
        return plan
    except Exception as e:
        # Surface the error so you can see it in the UI
        st.warning(f"Planner API error; using fallback. {type(e).__name__}: {e}")
        return fallback



# ---------- Plan executor (renders into Streamlit) ----------
def execute_plan(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute validated plan steps against df, render results into Streamlit,
    and return a lightweight 'artifacts' dict for narrative/export.
    """
    err = validate_plan(plan)
    if err:
        st.error(f"Invalid plan: {err}")
        st.json(plan)
        return {"plan": plan, "steps": [], "notes": [f"invalid: {err}"]}

    artifacts: Dict[str, Any] = {"plan": plan, "steps": [], "notes": []}

    for i, step in enumerate(plan["steps"]):
        op = step["op"]
        st.markdown(f"**Step {i+1}: {op}**")
        step_art = {"op": op}

        if op == "summarise":
            summary, missing_by_col, numeric_desc, top_cats = summarise_dataset(df)
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: st.metric("Rows", f"{summary['rows']:,}")
            with c2: st.metric("Columns", f"{summary['cols']:,}")
            with c3: st.metric("Missing (total)", f"{summary['missing_values_total']:,}")
            st.dataframe(numeric_desc.head(10), use_container_width=True)
            if len(missing_by_col) > 0:
                st.markdown("Missing values by column (top 20)")
                st.bar_chart(missing_by_col, use_container_width=True)
            if top_cats:
                st.markdown("Top categories")
                for col, vc in top_cats.items():
                    st.write(f"• {col}")
                    st.bar_chart(vc, use_container_width=True)

            # Save compact bits for narrative
            step_art.update({
                "summary": summary,
                "numeric_desc_head": numeric_desc.head(10).to_dict(),
                "missing_by_col_head": missing_by_col.head(10).to_dict(),
                "top_cats_cols": list(top_cats.keys()),
            })

        elif op == "correlation":
            corr = key_trends_numeric_only(df)
            if corr is None:
                st.info("Not enough numeric columns to compute correlations.")
                step_art["corr"] = None
            else:
                st.dataframe(corr, use_container_width=True)
                # Keep top absolute correlations (off-diagonal)
                try:
                    corr_vals = corr.copy()
                    for c in corr.columns:
                        corr_vals.loc[c, c] = 0.0
                    pairs = (
                        corr_vals.abs()
                        .stack()
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={"level_0":"col_a","level_1":"col_b",0:"abs_corr"})
                    )
                    top_pairs = pairs.drop_duplicates(subset=["col_a","col_b"]).head(10).to_dict(orient="records")
                except Exception:
                    top_pairs = []
                step_art["top_corr_pairs"] = top_pairs

        elif op == "histogram":
            col = step["column"]
            if col not in df.columns:
                st.warning(f"Column '{col}' not found.")
                step_art["error"] = f"missing {col}"
            else:
                series = df[col].dropna().value_counts().sort_index()
                st.bar_chart(series, use_container_width=True)
                step_art.update({"column": col, "top_values": series.head(15).to_dict()})

        elif op == "top_categories":
            col = step["column"]
            n = int(step.get("n", 15))
            if col not in df.columns:
                st.warning(f"Column '{col}' not found.")
                step_art["error"] = f"missing {col}"
            else:
                series = df[col].astype(str).value_counts().head(n)
                st.bar_chart(series, use_container_width=True)
                step_art.update({"column": col, "top_values": series.to_dict()})

        elif op == "groupby_aggregate":
            g = step["group_col"]; a = step["agg_col"]; fn = step["agg_fn"]
            if g not in df.columns or a not in df.columns:
                st.warning(f"Columns '{g}' or '{a}' not found.")
                step_art["error"] = f"missing {g} or {a}"
            elif fn not in {"sum", "mean", "count"}:
                st.warning(f"Aggregation '{fn}' not supported.")
                step_art["error"] = f"bad agg {fn}"
            else:
                sub = df[[g, a]].copy()
                if fn == "count":
                    series = sub.groupby(g)[a].count().sort_values(ascending=False)
                elif fn == "sum":
                    series = sub.groupby(g)[a].sum().sort_values(ascending=False)
                else:
                    sub[a] = pd.to_numeric(sub[a], errors="coerce")
                    series = sub.groupby(g)[a].mean().sort_values(ascending=False)
                st.bar_chart(series.head(30), use_container_width=True)
                step_art.update({"group_col": g, "agg_col": a, "agg_fn": fn, "series_head": series.head(15).to_dict()})

        artifacts["steps"].append(step_art)

    return artifacts