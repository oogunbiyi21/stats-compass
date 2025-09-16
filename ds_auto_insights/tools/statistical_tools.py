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
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


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


class TTestInput(BaseModel):
    column: str = Field(description="Name of the numeric column to test")
    test_type: str = Field(default="one_sample", description="Type of t-test: 'one_sample', 'two_sample', or 'paired'")
    null_value: Optional[float] = Field(default=0, description="Null hypothesis value for one-sample test")
    group_column: Optional[str] = Field(default=None, description="Column to group by for two-sample test")
    group_values: Optional[List[str]] = Field(default=None, description="For two-sample test: specify exactly 2 group values to compare")
    column2: Optional[str] = Field(default=None, description="Second column for paired t-test")
    alpha: float = Field(default=0.05, description="Significance level (default 0.05)")


class RunTTestTool(BaseTool):
    name: str = "run_t_test"
    description: str = "Perform t-tests (one-sample, two-sample, or paired) to test hypotheses about means. Includes visualizations and practical interpretation for business decisions."
    args_schema: Type[BaseModel] = TTestInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column: str, test_type: str = "one_sample", null_value: float = 0, 
             group_column: Optional[str] = None, group_values: Optional[List[str]] = None, 
             column2: Optional[str] = None, alpha: float = 0.05) -> str:
        try:
            # Validate inputs
            if column not in self._df.columns:
                return f"❌ Column '{column}' not found. Available columns: {list(self._df.columns)}"
            
            if not pd.api.types.is_numeric_dtype(self._df[column]):
                return f"❌ Column '{column}' must be numeric for t-test"
            
            # Clean data - remove missing values
            data = self._df[column].dropna()
            if len(data) == 0:
                return f"❌ No valid data in column '{column}' after removing missing values"
            
            if len(data) < 2:
                return f"❌ Need at least 2 observations for t-test, got {len(data)}"

            result_lines = [f"📊 **T-Test Analysis: {column}**\n"]
            
            # Perform the appropriate t-test
            if test_type == "one_sample":
                # One-sample t-test
                t_stat, p_value = stats.ttest_1samp(data, null_value)
                degrees_freedom = len(data) - 1
                mean_value = data.mean()
                std_value = data.std()
                effect_size = (mean_value - null_value) / std_value  # Cohen's d
                
                result_lines.extend([
                    f"**Test Type:** One-sample t-test",
                    f"**Null Hypothesis:** Mean = {null_value}",
                    f"**Alternative Hypothesis:** Mean ≠ {null_value}",
                    f"",
                    f"📈 **Sample Statistics:**",
                    f"  • Sample size: {len(data):,}",
                    f"  • Sample mean: {mean_value:.4f}",
                    f"  • Sample std: {std_value:.4f}",
                    f"  • Difference from null: {mean_value - null_value:.4f}",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • t-statistic: {t_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Degrees of freedom: {degrees_freedom}",
                    f"  • Effect size (Cohen's d): {effect_size:.4f}",
                ])
                
                # Create visualization
                fig = go.Figure()
                
                # Histogram of data
                fig.add_trace(go.Histogram(
                    x=data,
                    name="Data Distribution",
                    opacity=0.7,
                    nbinsx=30
                ))
                
                # Add vertical lines for sample mean and null value
                fig.add_vline(x=mean_value, line_dash="dash", line_color="red", 
                             annotation_text=f"Sample Mean: {mean_value:.3f}")
                fig.add_vline(x=null_value, line_dash="solid", line_color="blue", 
                             annotation_text=f"Null Value: {null_value}")
                
                fig.update_layout(
                    title=f"One-Sample T-Test: {column}",
                    xaxis_title=column,
                    yaxis_title="Frequency",
                    showlegend=True
                )
                
            elif test_type == "two_sample":
                # Two-sample t-test
                if not group_column or group_column not in self._df.columns:
                    return f"❌ For two-sample t-test, specify a valid group_column"
                
                # Get unique groups from the group column
                unique_groups = list(self._df[group_column].unique())
                
                # If group_values specified, use those; otherwise check for exactly 2 groups
                if group_values:
                    if len(group_values) != 2:
                        return f"❌ For two-sample t-test, group_values must contain exactly 2 values, got {len(group_values)}: {group_values}"
                    
                    # Check if specified groups exist
                    missing_groups = [g for g in group_values if g not in unique_groups]
                    if missing_groups:
                        return f"❌ Groups not found in data: {missing_groups}. Available groups: {unique_groups}"
                    
                    selected_groups = group_values
                else:
                    if len(unique_groups) != 2:
                        return f"❌ Two-sample t-test requires exactly 2 groups, found {len(unique_groups)}: {unique_groups}. Use group_values parameter to specify which 2 groups to compare."
                    
                    selected_groups = unique_groups
                
                # Extract data for the selected groups
                group1_name = selected_groups[0]
                group2_name = selected_groups[1]
                group1_data = self._df[self._df[group_column] == group1_name][column].dropna()
                group2_data = self._df[self._df[group_column] == group2_name][column].dropna()
                
                if len(group1_data) < 2 or len(group2_data) < 2:
                    return f"❌ Each group needs at least 2 observations. Group sizes: {group1_name}={len(group1_data)}, {group2_name}={len(group2_data)}"
                
                # Perform two-sample t-test (assuming unequal variances by default - Welch's t-test)
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                
                # Calculate effect size (Cohen's d for two groups)
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                     (len(group2_data) - 1) * group2_data.var()) / 
                                    (len(group1_data) + len(group2_data) - 2))
                effect_size = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                result_lines.extend([
                    f"**Test Type:** Two-sample t-test (Welch's - unequal variances assumed)",
                    f"**Null Hypothesis:** Mean({group1_name}) = Mean({group2_name})",
                    f"**Alternative Hypothesis:** Mean({group1_name}) ≠ Mean({group2_name})",
                    f"",
                    f"📈 **Group Statistics:**",
                    f"  • **{group1_name}:** n={len(group1_data)}, mean={group1_data.mean():.4f}, std={group1_data.std():.4f}",
                    f"  • **{group2_name}:** n={len(group2_data)}, mean={group2_data.mean():.4f}, std={group2_data.std():.4f}",
                    f"  • **Difference:** {group1_data.mean() - group2_data.mean():.4f}",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • t-statistic: {t_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Effect size (Cohen's d): {effect_size:.4f}",
                ])
                
                # Create box plot comparison
                fig = go.Figure()
                fig.add_trace(go.Box(y=group1_data, name=str(group1_name), boxpoints="outliers"))
                fig.add_trace(go.Box(y=group2_data, name=str(group2_name), boxpoints="outliers"))
                
                fig.update_layout(
                    title=f"Two-Sample T-Test: {column} by {group_column}",
                    xaxis_title=group_column,
                    yaxis_title=column,
                    showlegend=True
                )
                
            elif test_type == "paired":
                # Paired t-test
                if not column2 or column2 not in self._df.columns:
                    return f"❌ For paired t-test, specify a valid column2"
                
                if not pd.api.types.is_numeric_dtype(self._df[column2]):
                    return f"❌ Column '{column2}' must be numeric for paired t-test"
                
                # Get paired data (remove rows where either value is missing)
                paired_data = self._df[[column, column2]].dropna()
                if len(paired_data) < 2:
                    return f"❌ Need at least 2 complete pairs for paired t-test, got {len(paired_data)}"
                
                data1 = paired_data[column]
                data2 = paired_data[column2]
                differences = data1 - data2
                
                t_stat, p_value = stats.ttest_rel(data1, data2)
                effect_size = differences.mean() / differences.std()  # Cohen's d for paired data
                
                result_lines.extend([
                    f"**Test Type:** Paired t-test",
                    f"**Null Hypothesis:** Mean difference = 0",
                    f"**Alternative Hypothesis:** Mean difference ≠ 0",
                    f"",
                    f"📈 **Paired Statistics:**",
                    f"  • Pairs: {len(paired_data):,}",
                    f"  • **{column}:** mean={data1.mean():.4f}, std={data1.std():.4f}",
                    f"  • **{column2}:** mean={data2.mean():.4f}, std={data2.std():.4f}",
                    f"  • **Mean difference:** {differences.mean():.4f}",
                    f"  • **Std of differences:** {differences.std():.4f}",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • t-statistic: {t_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Effect size (Cohen's d): {effect_size:.4f}",
                ])
                
                # Create scatter plot with line of equality
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data1, y=data2, mode='markers',
                    name='Data Points', opacity=0.6
                ))
                
                # Add line of equality
                min_val = min(data1.min(), data2.min())
                max_val = max(data1.max(), data2.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='Line of Equality',
                    line=dict(dash='dash', color='red')
                ))
                
                fig.update_layout(
                    title=f"Paired T-Test: {column} vs {column2}",
                    xaxis_title=column,
                    yaxis_title=column2,
                    showlegend=True
                )
                
            else:
                return f"❌ Invalid test_type '{test_type}'. Use: 'one_sample', 'two_sample', or 'paired'"
            
            # Interpret results
            result_lines.extend([
                f"",
                f"🎯 **Statistical Interpretation:**"
            ])
            
            if p_value < alpha:
                result_lines.append(f"  • **Significant result** (p < {alpha}): Reject null hypothesis")
                result_lines.append(f"  • There IS a statistically significant difference")
            else:
                result_lines.append(f"  • **Not significant** (p ≥ {alpha}): Fail to reject null hypothesis")
                result_lines.append(f"  • There is NO statistically significant difference")
            
            # Effect size interpretation
            abs_effect = abs(effect_size)
            if abs_effect < 0.2:
                effect_magnitude = "negligible"
            elif abs_effect < 0.5:
                effect_magnitude = "small"
            elif abs_effect < 0.8:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            result_lines.extend([
                f"  • **Effect size:** {effect_magnitude} (|d| = {abs_effect:.3f})",
                f"",
                f"⚠️ **Assumptions Made:**",
                f"  • Data is approximately normally distributed",
                f"  • Observations are independent",
                f"  • Data is measured at interval/ratio level"
            ])
            
            if test_type == "two_sample":
                result_lines.append(f"  • Unequal variances assumed (Welch's t-test)")
            
            # Store chart for display
            chart_info = {
                'type': 't_test',
                'title': f"T-Test Analysis: {column}",
                'data': None,  # Plotly figure will be created directly
                'fig': fig,
                'chart_config': {
                    'chart_type': 'statistical_test',
                    'test_type': test_type
                }
            }
            
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                st.session_state.current_response_charts.append(chart_info)
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in t-test analysis: {str(e)}"

    def _arun(self, column: str, test_type: str = "one_sample", null_value: float = 0, 
              group_column: Optional[str] = None, column2: Optional[str] = None, alpha: float = 0.05):
        raise NotImplementedError("Async not supported")
