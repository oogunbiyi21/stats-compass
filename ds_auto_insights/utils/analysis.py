# ds_auto_insights/utils/analysis.py

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Any


def extract_goal_kpis(text: str) -> Dict[str, Any]:
    """
    Extract business goals and KPIs from user input text.
    
    Args:
        text: User input text to analyze
        
    Returns:
        Dictionary containing extracted goal and KPIs
    """
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


def summarise_dataset(df: pd.DataFrame) -> Tuple[Dict, pd.Series, pd.DataFrame, Dict]:
    """
    Generate comprehensive dataset summary with key statistics.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Tuple of (summary_stats, missing_by_column, numeric_description, top_categories)
    """
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


def key_trends_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns only.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Correlation matrix or None if insufficient numeric columns
    """
    num = df.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        return num.corr(numeric_only=True)
    return None


def suggest_visualisations(df: pd.DataFrame) -> List[Tuple[str, callable]]:
    """
    Suggest relevant visualizations based on DataFrame structure.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of tuples containing (description, render_function)
    """
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
