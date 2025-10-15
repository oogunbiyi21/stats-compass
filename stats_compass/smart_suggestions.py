# stats_compass/smart_suggestions.py

import pandas as pd
from typing import List, Dict, Any
import numpy as np


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain date/time data"""
    date_columns = []
    
    for col in df.columns:
        try:
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
        except:
            # Skip any problematic columns (malformed names, access issues, etc.)
            continue
    
    return date_columns


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are likely IDs (high cardinality, unique values)"""
    id_columns = []
    
    for col in df.columns:
        try:
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
        except:
            # Skip any problematic columns
            continue
    
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
    
    # Data Quality - Missing Values Detection
    missing_values_total = df.isnull().sum().sum()
    missing_percentage = (missing_values_total / (len(df) * len(df.columns))) * 100
    
    if missing_values_total > 0:
        suggestions.append({
            "title": "🧹 Clean Dataset",
            "description": f"Handle {missing_values_total:,} missing values ({missing_percentage:.1f}% of data)",
            "query": "Clean my dataset by handling missing values and data quality issues",
            "tool": "suggest_data_cleaning",
            "priority": 8,  # High priority - data cleaning should come before analysis
            "category": "quality",
            "why": f"Found {missing_values_total:,} missing values - cleaning data improves analysis reliability"
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
    
    # MACHINE LEARNING SUGGESTIONS (always include at least one)
    ml_suggestions = []
    
    # Regression suggestions (if we have numeric target candidates)
    if len(analysis['numeric_columns']) >= 2:
        target_col = analysis['numeric_columns'][0]  # Use first numeric as potential target
        feature_cols = analysis['numeric_columns'][1:4]  # Next 3 as features
        
        ml_suggestions.append({
            "title": f"🤖 Linear Regression: Predict {target_col}",
            "description": f"Build a model to predict {target_col} using other variables",
            "query": f"Build a linear regression model to predict {target_col}",
            "tool": "run_linear_regression",
            "priority": 8,
            "category": "ml",
            "why": f"Predict {target_col} using regression - reveals what factors drive this outcome"
        })
    
    # Classification suggestions (if we have categorical targets and numeric features)
    if analysis['categorical_columns'] and analysis['numeric_columns']:
        target_col = analysis['categorical_columns'][0]  # Use first categorical as target
        # Check if it's a reasonable classification target (not too many categories)
        if len(df[target_col].unique()) <= 10:
            ml_suggestions.append({
                "title": f"🎯 Classification: Predict {target_col}",
                "description": f"Build a model to classify {target_col} categories",
                "query": f"Build a logistic regression model to predict {target_col}",
                "tool": "run_logistic_regression", 
                "priority": 8,
                "category": "ml",
                "why": f"Classify {target_col} categories - identify patterns that determine outcomes"
            })
    
    # Time series ML (if we have time data)
    if analysis['date_columns'] and analysis['numeric_columns']:
        time_col = analysis['date_columns'][0]
        target_col = analysis['numeric_columns'][0]
        ml_suggestions.append({
            "title": f"📈 ARIMA Forecasting: {target_col}",
            "description": f"Forecast future values of {target_col} over time",
            "query": f"Build an ARIMA model to forecast {target_col} using {time_col}",
            "tool": "run_arima_analysis",
            "priority": 9,
            "category": "ml",
            "why": f"Forecast {target_col} trends - predict future patterns from historical data"
        })
    
    # Add at least one ML suggestion if we have any
    if ml_suggestions:
        suggestions.extend(ml_suggestions[:1])  # Add the highest priority ML suggestion
    
    # STATISTICAL TEST SUGGESTIONS (always include at least one)
    stat_suggestions = []
    
    # T-test suggestions (comparing groups)
    if analysis['categorical_columns'] and analysis['numeric_columns']:
        cat_col = analysis['categorical_columns'][0]
        num_col = analysis['numeric_columns'][0]
        # Check if categorical has exactly 2 groups (perfect for t-test)
        unique_groups = df[cat_col].nunique()
        if unique_groups == 2:
            stat_suggestions.append({
                "title": f"📊 T-Test: {num_col} by {cat_col}",
                "description": f"Test if {num_col} differs significantly between {cat_col} groups",
                "query": f"Run a t-test to compare {num_col} between {cat_col} groups",
                "tool": "run_t_test",
                "priority": 7,
                "category": "stats",
                "why": f"Compare {num_col} across {cat_col} groups - test if differences are statistically significant"
            })
        elif unique_groups <= 5:
            stat_suggestions.append({
                "title": f"🧪 Statistical Test: {num_col} by {cat_col}",
                "description": f"Test relationships between {num_col} and {cat_col}",
                "query": f"Test if {num_col} varies significantly across {cat_col} groups",
                "tool": "run_t_test", 
                "priority": 6,
                "category": "stats",
                "why": f"Statistical testing reveals if group differences are meaningful or just random"
            })
    
    # Chi-square test for categorical relationships
    if len(analysis['categorical_columns']) >= 2:
        cat1 = analysis['categorical_columns'][0]
        cat2 = analysis['categorical_columns'][1]
        stat_suggestions.append({
            "title": f"🔗 Chi-Square Test: {cat1} vs {cat2}",
            "description": f"Test if {cat1} and {cat2} are independent",
            "query": f"Run a chi-square test between {cat1} and {cat2}",
            "tool": "run_chi_square_test",
            "priority": 6,
            "category": "stats", 
            "why": f"Test independence between categorical variables - are they related?"
        })
    
    # Correlation significance test
    if len(analysis['numeric_columns']) >= 2:
        num1 = analysis['numeric_columns'][0]
        num2 = analysis['numeric_columns'][1]
        stat_suggestions.append({
            "title": f"📈 Correlation Test: {num1} vs {num2}",
            "description": f"Test if correlation between {num1} and {num2} is significant",
            "query": f"Test the statistical significance of correlation between {num1} and {num2}",
            "tool": "correlation_matrix",
            "priority": 5,
            "category": "stats",
            "why": f"Statistical testing shows if correlations are meaningful or just coincidence"
        })
    
    # Add at least one statistical test suggestion if we have any
    if stat_suggestions:
        suggestions.extend(stat_suggestions[:1])  # Add the highest priority statistical test
    
    # Sort suggestions by priority (higher first)
    suggestions.sort(key=lambda x: x['priority'], reverse=True)
    
    # Reorganize to ensure top 3 slots are: ML, Time Series ML, Statistical Test
    prioritized = []
    remaining = []
    
    # Find best from each priority category
    best_ml = None
    best_ts_ml = None
    best_stats = None
    
    for suggestion in suggestions:
        # Check if it's a time series ML (ARIMA)
        if suggestion['category'] == 'ml' and 'ARIMA' in suggestion['title']:
            if best_ts_ml is None:
                best_ts_ml = suggestion
            else:
                remaining.append(suggestion)
        # Check if it's a regular ML suggestion
        elif suggestion['category'] == 'ml':
            if best_ml is None:
                best_ml = suggestion
            else:
                remaining.append(suggestion)
        # Check if it's a statistical test
        elif suggestion['category'] == 'stats':
            if best_stats is None:
                best_stats = suggestion
            else:
                remaining.append(suggestion)
        else:
            remaining.append(suggestion)
    
    # Build prioritized list: ML, Time Series ML, Stats Test, then rest
    if best_ml:
        prioritized.append(best_ml)
    if best_ts_ml:
        prioritized.append(best_ts_ml)
    if best_stats:
        prioritized.append(best_stats)
    
    # Add remaining suggestions
    prioritized.extend(remaining)
    
    return prioritized


def get_category_emoji(category: str) -> str:
    """Get emoji for suggestion category"""
    emoji_map = {
        'temporal': '📈',
        'relationships': '🔗', 
        'distribution': '📊',
        'categorical': '🏷️',
        'segmentation': '🔍',
        'quality': '🚨',
        'ml': '🤖',
        'stats': '📊'
    }
    return emoji_map.get(category, '💡')


def format_suggestion_for_ui(suggestion: Dict[str, Any]) -> str:
    """Format a suggestion for display in the UI"""
    emoji = get_category_emoji(suggestion['category'])
    return f"{emoji} **{suggestion['title']}**\n\n_{suggestion['description']}_\n\n💭 {suggestion['why']}"
