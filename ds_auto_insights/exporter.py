# ds_auto_insights/exporter.py
from typing import Dict, Any, List
import io
import pandas as pd

def _dict_to_md_table(d: Dict[str, Any], max_rows: int = 20) -> str:
    if not d:
        return "_(empty)_"
    # Try to normalize to tabular form
    try:
        df = pd.DataFrame(d)
    except Exception:
        df = pd.DataFrame.from_dict(d, orient="index", columns=["value"])
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=True)

def artifacts_to_markdown(artifacts: Dict[str, Any], narrative_text: str | None = None, title: str = "DS Auto Insights Export") -> bytes:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # Narrative first (if present)
    if narrative_text:
        lines.append("## Narrative")
        lines.append(narrative_text.strip())
        lines.append("")

    # Plan
    plan = artifacts.get("plan", {})
    if plan:
        lines.append("## Plan")
        lines.append("```json")
        import json
        lines.append(json.dumps(plan, indent=2))
        lines.append("```")
        lines.append("")

    # Steps
    steps = artifacts.get("steps", [])
    if steps:
        lines.append("## Results")
        for i, s in enumerate(steps, start=1):
            op = s.get("op", "unknown")
            lines.append(f"### Step {i}: {op}")
            # Summarise bits
            if op == "summarise":
                summary = s.get("summary", {})
                lines.append("**Summary**")
                lines.append(_dict_to_md_table(summary))
                lines.append("")
                missing = s.get("missing_by_col_head", {})
                if missing:
                    lines.append("**Missing values (top)**")
                    lines.append(_dict_to_md_table(missing))
                    lines.append("")
                numdesc = s.get("numeric_desc_head", {})
                if numdesc:
                    lines.append("**Numeric summary (head)**")
                    lines.append(_dict_to_md_table(numdesc))
                    lines.append("")
                topcats = s.get("top_cats_cols", [])
                if topcats:
                    lines.append(f"**Top categories columns:** {', '.join(topcats)}")
                    lines.append("")
            elif op in {"histogram", "top_categories"}:
                col = s.get("column")
                top_vals = s.get("top_values", {})
                lines.append(f"**Column:** `{col}`")
                if top_vals:
                    # Convert series dict to 2-col table
                    df = pd.DataFrame(list(top_vals.items()), columns=["value", "count"])
                    lines.append(df.to_markdown(index=False))
                lines.append("")
            elif op == "correlation":
                pairs = s.get("top_corr_pairs", [])
                if pairs:
                    df = pd.DataFrame(pairs)
                    lines.append("**Top correlation pairs (by |corr|)**")
                    lines.append(df.to_markdown(index=False))
                    lines.append("")
            elif op == "groupby_aggregate":
                g, a, fn = s.get("group_col"), s.get("agg_col"), s.get("agg_fn")
                lines.append(f"**Group:** `{g}` — **Metric:** `{a}` — **Agg:** `{fn}`")
                series_head = s.get("series_head", {})
                if series_head:
                    df = pd.DataFrame(list(series_head.items()), columns=[g, f"{fn}({a})"])
                    lines.append(df.to_markdown(index=False))
                lines.append("")
            else:
                lines.append("_No structured export for this op yet._\n")

    content = "\n".join(lines)
    return content.encode("utf-8")
