# ds_auto_insights/smart_suggestions.py

import pandas as pd
from typing import List, Dict, Any
import numpy as np


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain date/time data"""
    date_columns = []
    
    for col in df.columns:
        # Check if column is already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
            
        # Try to parse as datetime for object columns
        if df[col].dtype == 'object':
            try:
                # Sample a few non-null values
                sample_values = df[col].dropna().head(5).astype(str)
                if len(sample_values) > 0:
                    # Check if values look like dates (basic patterns)
                    sample_str = sample_values.iloc[0] if len(sample_values) > 0 else ""
                    
                    # Skip if it looks like regular text (more than 50 chars or contains many letters)
                    if len(sample_str) > 50 or sum(c.isalpha() for c in sample_str) > len(sample_str) * 0.7:
                        continue
                        
                    # Try to parse with common date formats first
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        # Try common formats first
                        try:
                            pd.to_datetime(sample_values, format='%Y-%m-%d', errors='raise')
                        except:
                            try:
                                pd.to_datetime(sample_values, format='%m/%d/%Y', errors='raise')
                            except:
                                try:
                                    pd.to_datetime(sample_values, format='%d/%m/%Y', errors='raise') 
                                except:
                                    # Fall back to automatic parsing
                                    pd.to_datetime(sample_values, errors='raise')
                    date_columns.append(col)
            except:
                continue
    
    return date_columns


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are likely IDs (high cardinality, unique values)"""
    id_columns = []
    
    for col in df.columns:
        # Check if column name suggests it's an ID
        col_lower = col.lower()
        if any(id_term in col_lower for id_term in ['id', 'key', 'index', 'pk']):
            id_columns.append(col)
            continue
        
        # Check cardinality - if unique values > 50% of total rows, likely an ID
        if len(df) > 0:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5 and df[col].nunique() > 10:
                id_columns.append(col)
    
    return id_columns


def analyze_column_relationships(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze relationships between columns to suggest meaningful combinations"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = detect_date_columns(df)
    id_cols = detect_id_columns(df)
    
    # Remove ID columns from analysis columns (less interesting for analysis)
    analysis_numeric = [col for col in numeric_cols if col not in id_cols]
    analysis_categorical = [col for col in categorical_cols if col not in id_cols]
    
    return {
        'numeric_columns': analysis_numeric,
        'categorical_columns': analysis_categorical,
        'date_columns': date_cols,
        'id_columns': id_cols,
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }


def generate_smart_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate intelligent analysis suggestions based on dataset characteristics"""
    
    if df is None or len(df) == 0:
        return []
    
    suggestions = []
    analysis = analyze_column_relationships(df)
    
    # Time Series Suggestions
    if analysis['date_columns'] and analysis['numeric_columns']:
        for date_col in analysis['date_columns'][:2]:  # Limit to first 2 date columns
            for numeric_col in analysis['numeric_columns'][:3]:  # Top 3 numeric columns
                suggestions.append({
                    "title": f"📈 Time Trends: {numeric_col}",
                    "description": f"Analyze {numeric_col} trends over {date_col}",
                    "query": f"Analyze {numeric_col} trends over time using {date_col}",
                    "tool": "time_series_analysis",
                    "priority": 9,
                    "category": "temporal",
                    "why": f"Found date column '{date_col}' and numeric data - time series analysis reveals patterns"
                })
    
    # Correlation Analysis
    if len(analysis['numeric_columns']) >= 2:
        suggestions.append({
            "title": "🔗 Correlation Heatmap",
            "description": "Visualize relationships between all numeric variables",
            "query": "Show me a correlation heatmap of all numeric variables",
            "tool": "create_correlation_heatmap",
            "priority": 8,
            "category": "relationships",
            "why": f"Found {len(analysis['numeric_columns'])} numeric columns - correlations reveal hidden relationships"
        })
        
        # Specific high-value correlation suggestions
        if len(analysis['numeric_columns']) >= 3:
            suggestions.append({
                "title": "🎯 Key Relationships",
                "description": "Find the strongest correlations between variables",
                "query": "What are the strongest correlations in my data?",
                "tool": "correlation_matrix",
                "priority": 7,
                "category": "relationships", 
                "why": "Multiple numeric variables - identifying key relationships helps focus analysis"
            })
    
    # Distribution Analysis
    for col in analysis['numeric_columns'][:3]:  # Top 3 numeric columns
        suggestions.append({
            "title": f"📊 Distribution: {col}",
            "description": f"Explore the distribution and outliers in {col}",
            "query": f"Show me the distribution of {col}",
            "tool": "create_histogram_chart",
            "priority": 6,
            "category": "distribution",
            "why": f"Understanding {col} distribution reveals data patterns and potential outliers"
        })
    
    # Categorical Analysis
    for col in analysis['categorical_columns'][:3]:  # Top 3 categorical columns
        suggestions.append({
            "title": f"🏷️ Top Categories: {col}",
            "description": f"See the most common values in {col}",
            "query": f"Show me the top categories in {col}",
            "tool": "create_bar_chart",
            "priority": 5,
            "category": "categorical",
            "why": f"Categorical data in {col} - top categories show important patterns"
        })
    
    # Cross-analysis between categorical and numeric
    if analysis['categorical_columns'] and analysis['numeric_columns']:
        cat_col = analysis['categorical_columns'][0]
        num_col = analysis['numeric_columns'][0]
        suggestions.append({
            "title": f"🔍 {num_col} by {cat_col}",
            "description": f"Compare {num_col} across different {cat_col} groups",
            "query": f"Show me {num_col} grouped by {cat_col}",
            "tool": "groupby_aggregate",
            "priority": 7,
            "category": "segmentation",
            "why": f"Combining categorical ({cat_col}) and numeric ({num_col}) data reveals group differences"
        })
    
    # Outlier Detection
    if analysis['numeric_columns']:
        suggestions.append({
            "title": "🚨 Outlier Detection",
            "description": "Identify unusual data points that might need attention",
            "query": "Find outliers in my numeric data",
            "tool": "run_pandas_query",
            "priority": 4,
            "category": "quality",
            "why": "Outliers often reveal data quality issues or interesting edge cases"
        })
    
    # Data Quality Overview
    suggestions.append({
        "title": "🔍 Data Quality Check",
        "description": "Review missing values, duplicates, and basic statistics",
        "query": "Give me a data quality overview",
        "tool": "dataset_preview",
        "priority": 3,
        "category": "quality",
        "why": "Understanding data quality is essential before deeper analysis"
    })
    
    # Sort suggestions by priority (higher first) and limit to top suggestions
    suggestions.sort(key=lambda x: x['priority'], reverse=True)
    
    # Return top 6 suggestions to avoid overwhelming the user
    return suggestions[:6]


def get_category_emoji(category: str) -> str:
    """Get emoji for suggestion category"""
    emoji_map = {
        'temporal': '📈',
        'relationships': '🔗', 
        'distribution': '📊',
        'categorical': '🏷️',
        'segmentation': '🔍',
        'quality': '🚨'
    }
    return emoji_map.get(category, '💡')


def format_suggestion_for_ui(suggestion: Dict[str, Any]) -> str:
    """Format a suggestion for display in the UI"""
    emoji = get_category_emoji(suggestion['category'])
    return f"{emoji} **{suggestion['title']}**\n\n_{suggestion['description']}_\n\n💭 {suggestion['why']}"
