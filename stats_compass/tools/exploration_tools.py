# stats_compass/tools/exploration_tools.py
"""
Data exploration tools for DS Auto Insights.
Provides comprehensive data exploration and analysis capabilities.
"""


from typing import Type, Optional, List
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool

# Import centralized configuration constants
from stats_compass.constants import (
    STRONG_CORRELATION_THRESHOLD,
)


# ============================================
# State Management
# ============================================

class DataFrameStateManager:
    """
    Centralized state management for dataframes and user variables.
    
    Provides a single source of truth for all state, preventing race conditions
    and synchronization issues between tool instances and session state.
    """
    
    @staticmethod
    def get_active_df() -> Optional[pd.DataFrame]:
        """Get the currently active dataframe from session state."""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state['df']
        return None
    
    @staticmethod
    def set_active_df(df: pd.DataFrame) -> None:
        """Update the active dataframe in session state."""
        if hasattr(st, 'session_state'):
            st.session_state['df'] = df
    
    @staticmethod
    def get_user_var(name: str) -> Optional[any]:
        """Get a user-defined variable from session state."""
        if hasattr(st, 'session_state') and name in st.session_state:
            return st.session_state[name]
        return None
    
    @staticmethod
    def set_user_var(name: str, value: any) -> None:
        """Store a user-defined variable in session state."""
        if hasattr(st, 'session_state'):
            st.session_state[name] = value
    
    @staticmethod
    def get_all_user_vars() -> dict:
        """Get all user-defined variables (excluding system variables)."""
        if not hasattr(st, 'session_state'):
            return {}
        
        # System/protected keys to exclude
        protected_keys = {'df', 'pd', 'np', 'messages', 'chat_history', 
                         'current_response_charts', 'ml_model_results',
                         'ml_workflow_state', 'available_encoded_columns'}
        
        return {k: v for k, v in st.session_state.items() 
                if k not in protected_keys and not k.startswith('_')}
    
    @staticmethod
    def clear_user_vars() -> None:
        """Clear all user-defined variables from session state."""
        if not hasattr(st, 'session_state'):
            return
        
        user_vars = DataFrameStateManager.get_all_user_vars()
        for key in user_vars.keys():
            del st.session_state[key]
    
    @staticmethod
    def list_available_dataframes() -> List[str]:
        """List all DataFrame objects stored in session state."""
        if not hasattr(st, 'session_state'):
            return []
        
        return [k for k, v in st.session_state.items() 
                if isinstance(v, pd.DataFrame)]


# ============================================
# Exploration Tools
# ============================================


class GroupByAggregateInput(BaseModel):
    group_column: str = Field(description="Column to group by")
    metric_column: str = Field(description="Column to aggregate")
    aggregation: str = Field(description="Aggregation function: mean, sum, count, max, min, std")


class GroupByAggregateTool(BaseTool):
    name: str = "groupby_aggregate"
    description: str = "Group data by a column and aggregate a metric. Safer than writing groupby manually."
    args_schema: Type[BaseModel] = GroupByAggregateInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, group_column: str, metric_column: str, aggregation: str) -> str:
        try:
            agg_func = aggregation.lower()

            if group_column not in self._df.columns:
                return f"❌ Group column '{group_column}' not found."
            if metric_column not in self._df.columns:
                return f"❌ Metric column '{metric_column}' not found."

            valid_aggs = ['mean', 'sum', 'count', 'max', 'min', 'std', 'median']
            if agg_func not in valid_aggs:
                return f"❌ Invalid aggregation. Use one of: {valid_aggs}"

            grouped = self._df.groupby(group_column)[metric_column]
            
            if agg_func == 'mean':
                result = grouped.mean()
            elif agg_func == 'sum':
                result = grouped.sum()
            elif agg_func == 'count':
                result = grouped.count()
            elif agg_func == 'max':
                result = grouped.max()
            elif agg_func == 'min':
                result = grouped.min()
            elif agg_func == 'std':
                result = grouped.std()
            elif agg_func == 'median':
                result = grouped.median()

            return f"📊 {agg_func.title()} of {metric_column} by {group_column}:\n{result.to_string()}"

        except Exception as e:
            return f"❌ Error in groupby aggregation: {e}"

    def _arun(self, group_column: str, metric_column: str, aggregation: str):
        raise NotImplementedError("Async not supported")

    
class TopCategoriesInput(BaseModel):
    column: str = Field(description="Column to analyze")
    n: int = Field(default=10, description="Number of top categories to return")


class TopCategoriesTool(BaseTool):
    name: str = "top_categories"
    description: str = "Get the top N most frequent values in a categorical column."
    args_schema: Type[BaseModel] = TopCategoriesInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column: str, n: int = 10) -> str:
        try:
            if column not in self._df.columns:
                return f"❌ Column '{column}' not found."

            value_counts = self._df[column].value_counts().head(n)
            total_count = len(self._df)
            
            result_lines = [f"📊 Top {n} categories in '{column}':"]
            for value, count in value_counts.items():
                percentage = (count / total_count) * 100
                result_lines.append(f"  {value}: {count} ({percentage:.1f}%)")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error analyzing categories: {e}"

    def _arun(self, column: str, n: int = 10):
        raise NotImplementedError("Async not supported")


class HistogramInput(BaseModel):
    column: str = Field(description="Numeric column to create histogram for")
    bins: int = Field(default=10, description="Number of bins for the histogram")


class HistogramTool(BaseTool):
    name: str = "histogram"
    description: str = "Generate histogram data for a numeric column to understand distribution."
    args_schema: Type[BaseModel] = HistogramInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column: str, bins: int = 10) -> str:
        try:
            if column not in self._df.columns:
                return f"❌ Column '{column}' not found."

            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(self._df[column]):
                return f"❌ Column '{column}' is not numeric. Use top_categories for categorical data."

            series = self._df[column].dropna()
            if len(series) == 0:
                return f"❌ No data available in column '{column}' after removing missing values."

            # Generate histogram
            counts, bin_edges = pd.cut(series, bins=bins, retbins=True)
            hist_data = counts.value_counts().sort_index()

            result_lines = [f"📊 Distribution of '{column}' ({len(series)} values):"]
            result_lines.append(f"Range: {series.min():.2f} to {series.max():.2f}")
            result_lines.append(f"Mean: {series.mean():.2f}, Median: {series.median():.2f}")
            result_lines.append("\nHistogram:")
            
            for interval, count in hist_data.items():
                percentage = (count / len(series)) * 100
                result_lines.append(f"  {interval}: {count} ({percentage:.1f}%)")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error creating histogram: {e}"

    def _arun(self, column: str, bins: int = 10):
        raise NotImplementedError("Async not supported")


class CorrelationMatrixInput(BaseModel):
    columns: Optional[List[str]] = Field(default=None, description="Specific columns to include (optional)")
    method: str = Field(default="pearson", description="Correlation method: pearson, spearman, or kendall")


class CorrelationMatrixTool(BaseTool):
    name: str = "correlation_matrix"
    description: str = "Calculate correlation matrix for numeric columns."
    args_schema: Type[BaseModel] = CorrelationMatrixInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, columns: Optional[List[str]] = None, method: str = "pearson") -> str:
        try:
            method = method.lower()
            if method not in ['pearson', 'spearman', 'kendall']:
                return f"❌ Invalid method. Use: pearson, spearman, or kendall"

            # Select numeric columns
            if columns:
                # Use specified columns
                numeric_cols = [col for col in columns if col in self._df.columns and pd.api.types.is_numeric_dtype(self._df[col])]
                if not numeric_cols:
                    return f"❌ No valid numeric columns found in specified list."
                df_numeric = self._df[numeric_cols]
            else:
                # Use all numeric columns
                df_numeric = self._df.select_dtypes(include=['number'])
                
            if len(df_numeric.columns) < 2:
                return f"❌ Need at least 2 numeric columns for correlation. Found: {list(df_numeric.columns)}"

            # Calculate correlation
            corr_matrix = df_numeric.corr(method=method)
            
            result_lines = [f"📊 Correlation Matrix ({method.title()})"]
            result_lines.append(f"Columns analyzed: {list(df_numeric.columns)}")
            result_lines.append("\n📈 Correlation Summary:")
            
            # Only show a summary instead of the full matrix
            result_lines.append(f"  • Matrix size: {len(corr_matrix.columns)} × {len(corr_matrix.columns)}")
            result_lines.append(f"  • Method: {method.title()}")
            result_lines.append(f"  • Range: {corr_matrix.values[corr_matrix.values != 1.0].min():.3f} to {corr_matrix.values[corr_matrix.values != 1.0].max():.3f}")

            # Highlight strong correlations (exclude diagonal)
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > STRONG_CORRELATION_THRESHOLD:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        strong_corrs.append(f"{col1} ↔ {col2}: {corr_val:.3f}")

            if strong_corrs:
                result_lines.append(f"\n🔍 Strong correlations (|r| > {STRONG_CORRELATION_THRESHOLD}):")
                for corr in strong_corrs:
                    result_lines.append(f"  {corr}")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error calculating correlation: {e}"

    def _arun(self, columns: Optional[List[str]] = None, method: str = "pearson"):
        raise NotImplementedError("Async not supported")


class CreateHistogramChartInput(BaseModel):
    column: str = Field(description="Column name to create histogram for")
    bins: int = Field(default=30, description="Number of bins for the histogram")
    title: str = Field(default="", description="Custom title for the chart")


class CreateHistogramChartTool(BaseTool):
    name: str = "create_histogram_chart"
    description: str = "Creates a histogram chart for a numeric column and returns chart data for display."
    args_schema: Type[BaseModel] = CreateHistogramChartInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column: str, bins: int = 30, title: str = "") -> str:
        try:
            if column not in self._df.columns:
                return f"❌ Column '{column}' not found. Available columns: {list(self._df.columns)}"

            col_data = self._df[column].dropna()
            
            if not pd.api.types.is_numeric_dtype(col_data):
                return f"❌ Column '{column}' is not numeric. Use bar chart for categorical data."

            if len(col_data) == 0:
                return f"❌ No non-null data found in column '{column}'"

            # Create histogram data
            counts, bin_edges = pd.cut(col_data, bins=bins, retbins=True)
            hist_data = counts.value_counts().sort_index()

            # Prepare chart data for Streamlit
            chart_data = pd.DataFrame({
                'bin_range': [f"{edge:.2f}-{bin_edges[i+1]:.2f}" for i, edge in enumerate(bin_edges[:-1])],
                'count': [hist_data.get(interval, 0) for interval in hist_data.index]
            })

            chart_title = title or f"Distribution of {column}"
            
            result = f"📊 {chart_title}\n\n"
            result += f"Statistics:\n"
            result += f"  Mean: {col_data.mean():.2f}\n"
            result += f"  Median: {col_data.median():.2f}\n"
            result += f"  Std Dev: {col_data.std():.2f}\n"
            result += f"  Range: {col_data.min():.2f} to {col_data.max():.2f}\n"
            result += f"  Count: {len(col_data)}\n\n"

            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'histogram',
                    'data': chart_data,
                    'column': column,
                    'title': chart_title,
                    'bins': bins
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "Chart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating histogram chart: {e}"

    def _arun(self, column: str, bins: int = 30, title: str = ""):
        raise NotImplementedError("Async not supported")
