# ds_auto_insights/tools/statistical_tools.py
"""
Statistical analysis tools for DS Auto Insights.
Provides advanced statistical analysis capabilities.
"""

from typing import Type, Optional, List
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool


class DescribeDataInput(BaseModel):
    columns: Optional[List[str]] = Field(default=None, description="Specific columns to describe (optional)")


class DescribeDataTool(BaseTool):
    name: str = "describe_data"
    description: str = "Generate comprehensive descriptive statistics for the dataset or specified columns."
    args_schema: Type[BaseModel] = DescribeDataInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, columns: Optional[List[str]] = None) -> str:
        try:
            if columns:
                # Use specified columns
                missing_cols = [col for col in columns if col not in self._df.columns]
                if missing_cols:
                    return f"❌ Columns not found: {missing_cols}"
                df_to_describe = self._df[columns]
            else:
                df_to_describe = self._df

            # Get descriptive statistics
            description = df_to_describe.describe(include='all')
            
            result_lines = [f"📊 Descriptive Statistics"]
            result_lines.append(f"Dataset shape: {df_to_describe.shape[0]} rows × {df_to_describe.shape[1]} columns")
            result_lines.append(f"\nStatistics:")
            result_lines.append(str(description))
            
            # Additional insights
            result_lines.append(f"\n📋 Data Types:")
            for col, dtype in df_to_describe.dtypes.items():
                null_count = df_to_describe[col].isnull().sum()
                null_pct = (null_count / len(df_to_describe)) * 100
                result_lines.append(f"  {col}: {dtype} ({null_count} nulls, {null_pct:.1f}%)")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error generating descriptive statistics: {e}"

    def _arun(self, columns: Optional[List[str]] = None):
        raise NotImplementedError("Async not supported")


class CorrelationAnalysisInput(BaseModel):
    target_column: str = Field(description="Target column to analyze correlations with")
    method: str = Field(default="pearson", description="Correlation method: pearson, spearman, or kendall")
    threshold: float = Field(default=0.3, description="Minimum correlation threshold to report")


class CorrelationAnalysisTool(BaseTool):
    name: str = "correlation_analysis"
    description: str = "Analyze correlations between a target column and all other numeric columns."
    args_schema: Type[BaseModel] = CorrelationAnalysisInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, target_column: str, method: str = "pearson", threshold: float = 0.3) -> str:
        try:
            if target_column not in self._df.columns:
                return f"❌ Target column '{target_column}' not found."

            if not pd.api.types.is_numeric_dtype(self._df[target_column]):
                return f"❌ Target column '{target_column}' must be numeric."

            method = method.lower()
            if method not in ['pearson', 'spearman', 'kendall']:
                return f"❌ Invalid method. Use: pearson, spearman, or kendall"

            # Get all numeric columns except the target
            numeric_cols = self._df.select_dtypes(include=['number']).columns
            other_cols = [col for col in numeric_cols if col != target_column]

            if len(other_cols) == 0:
                return f"❌ No other numeric columns found for correlation analysis."

            # Calculate correlations
            correlations = []
            for col in other_cols:
                corr = self._df[target_column].corr(self._df[col], method=method)
                if not pd.isna(corr) and abs(corr) >= threshold:
                    correlations.append((col, corr))

            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            result_lines = [f"📊 Correlation Analysis: {target_column} vs Other Columns"]
            result_lines.append(f"Method: {method.title()}")
            result_lines.append(f"Threshold: {threshold}")
            result_lines.append(f"Columns analyzed: {len(other_cols)}")

            if correlations:
                result_lines.append(f"\n🔍 Significant Correlations (|r| >= {threshold}):")
                for col, corr in correlations:
                    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.5 else "Weak"
                    direction = "Positive" if corr > 0 else "Negative"
                    result_lines.append(f"  {col}: {corr:.3f} ({strength} {direction})")
            else:
                result_lines.append(f"\n📋 No significant correlations found above threshold {threshold}")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error in correlation analysis: {e}"

    def _arun(self, target_column: str, method: str = "pearson", threshold: float = 0.3):
        raise NotImplementedError("Async not supported")


class DisplayTimeSeriesChartInput(BaseModel):
    pass


class DisplayTimeSeriesChartTool(BaseTool):
    name: str = "display_time_series_chart"
    description: str = "Display the time series chart from the most recent time_series_analysis. Use this after running time_series_analysis to show the chart."
    args_schema: Type[BaseModel] = DisplayTimeSeriesChartInput

    def _run(self) -> str:
        try:
            if not hasattr(st, 'session_state') or 'time_series_chart_data' not in st.session_state:
                return "❌ No time series data available. Run 'time_series_analysis' first."
            
            chart_data = st.session_state.time_series_chart_data
            
            # Create the chart using plotly and display it
            import plotly.express as px
            
            fig = px.line(
                chart_data, 
                x='Date', 
                y='Value',
                title=chart_data.get('title', 'Time Series Analysis')
            )
            
            # Enhance the chart
            fig.update_traces(
                line=dict(width=3),
                hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
            )
            
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title=chart_data.get('ylabel', 'Value'),
                hovermode='x unified',
                height=500
            )
            
            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            return "✅ Time series chart displayed successfully!"
            
        except Exception as e:
            return f"❌ Error displaying time series chart: {str(e)}"

    def _arun(self):
        raise NotImplementedError("Async not supported")
