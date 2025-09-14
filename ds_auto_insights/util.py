# ds_auto_insights/utils.py
import io
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Data loading ----------
def load_table(uploaded_file):
    name = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else ""
    is_excel = name.endswith(".xlsx") or name.endswith(".xls")

    if is_excel:
        try:
            # Check if Excel file has multiple sheets
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Load the first sheet
            df = pd.read_excel(uploaded_file, sheet_name=0)
            
            # Show warning if multiple sheets detected
            if len(sheet_names) > 1:
                first_sheet = sheet_names[0]
                other_sheets = sheet_names[1:]
                other_sheets_str = ", ".join([f"'{sheet}'" for sheet in other_sheets])
                
                st.warning(
                    f"⚠️ **Multiple sheets detected in Excel file!**\n\n"
                    f"📊 **Currently analyzing:** '{first_sheet}' (first sheet)\n\n"
                    f"📋 **Other sheets found:** {other_sheets_str}\n\n"
                    f"💡 **Tip:** To analyze other sheets, save them as separate Excel files or CSV files and upload individually."
                )
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

    for opts in ({}, {"encoding": "utf-8-sig"}, {"sep": ";"}, {"on_bad_lines": "skip"}):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, **opts)
        except Exception:
            continue

    try:
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        for opts in ({}, {"encoding": "utf-8-sig"}, {"sep": ";"}, {"on_bad_lines": "skip"}):
            try:
                df = pd.read_csv(io.BytesIO(raw), **opts)
                # Check if DataFrame is empty or has no columns
                if df.empty or df.shape[1] == 0:
                    raise ValueError("The uploaded file appears to be empty or contains no data columns.")
                return df
            except ValueError as ve:
                # Re-raise ValueError to maintain the empty file message
                if "empty" in str(ve).lower():
                    raise ve
            except Exception:
                continue
    except Exception:
        pass

    raise ValueError("Unable to parse file as CSV or Excel. Try re-exporting or cleaning the file.")

# ---------- Context extraction (heuristic) ----------
def extract_goal_kpis(text: str):
    if not text or not text.strip():
        return {"goal": "", "kpis": []}

    goal = text.strip().split(".")[0][:160].strip()
    common_kpis = [
        "retention", "churn", "activation", "conversion", "revenue", "arpu",
        "ltv", "cac", "mau", "wau", "dau", "engagement", "trial",
        "trial-to-paid", "cohort", "drop-off", "signup", "onboarding",
        "nps", "csat", "aov", "gmv"
    ]
    lower = text.lower()
    kpis = sorted({k for k in common_kpis if k in lower})
    return {"goal": goal, "kpis": kpis}

# ---------- Analytics helpers (no time series) ----------
def summarise_dataset(df: pd.DataFrame):
    # Handle empty DataFrame case
    
    
    summary = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "numeric_cols": df.select_dtypes(include=np.number).shape[1],
        "non_numeric_cols": df.select_dtypes(exclude=np.number).shape[1],
        "missing_values_total": int(df.isna().sum().sum()),
    }
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    missing_by_col = missing_by_col[missing_by_col > 0].head(20)
    numeric_desc = df.select_dtypes(include=np.number).describe().T

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    top_cats = {}
    for c in cat_cols[:5]:
        vc = df[c].astype(str).value_counts(dropna=False).head(10)
        top_cats[c] = vc

    return summary, missing_by_col, numeric_desc, top_cats

def key_trends_numeric_only(df: pd.DataFrame):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        return num.corr(numeric_only=True)
    return None

def suggest_visualisations(df: pd.DataFrame):
    suggestions = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        def _render_hist(col):
            st.bar_chart(df[col].dropna().value_counts().sort_index(), use_container_width=True)
        suggestions.append((f"Distribution of {num_cols[0]}", lambda col=num_cols[0]: _render_hist(col)))

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        def _render_topcats(col):
            st.bar_chart(df[col].astype(str).value_counts().head(15), use_container_width=True)
        suggestions.append((f"Top categories in {cat_cols[0]}", lambda col=cat_cols[0]: _render_topcats(col)))

    return suggestions
