# ds_auto_insights/tools/statistical_test_tools.py
"""
Statistical test tools for DS Auto Insights.
Provides T-test and Z-test implementations for hypothesis testing.
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
    description: str = "Perform t-tests (one-sample, two-sample, or paired) to test hypotheses about means. Uses t-distribution for small samples or unknown population variance. Includes visualizations and practical interpretation."
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
                f"⚠️ **Assumptions:**",
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
              group_column: Optional[str] = None, group_values: Optional[List[str]] = None, 
              column2: Optional[str] = None, alpha: float = 0.05):
        raise NotImplementedError("Async not supported")


class ZTestInput(BaseModel):
    column: str = Field(description="Name of the numeric column to test")
    test_type: str = Field(default="one_sample", description="Type of z-test: 'one_sample', 'two_sample', or 'paired'")
    null_value: Optional[float] = Field(default=0, description="Null hypothesis value for one-sample test")
    group_column: Optional[str] = Field(default=None, description="Column to group by for two-sample test")
    group_values: Optional[List[str]] = Field(default=None, description="For two-sample test: specify exactly 2 group values to compare")
    column2: Optional[str] = Field(default=None, description="Second column for paired z-test")
    population_std: Optional[float] = Field(default=None, description="Known population standard deviation (if unknown, sample std will be used)")
    alpha: float = Field(default=0.05, description="Significance level (default 0.05)")


class RunZTestTool(BaseTool):
    name: str = "run_z_test"
    description: str = "Perform z-tests (one-sample, two-sample, or paired) to test hypotheses about means. Uses normal distribution - appropriate for large samples (n≥30) or when population standard deviation is known. Includes visualizations and practical interpretation."
    args_schema: Type[BaseModel] = ZTestInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column: str, test_type: str = "one_sample", null_value: float = 0, 
             group_column: Optional[str] = None, group_values: Optional[List[str]] = None, 
             column2: Optional[str] = None, population_std: Optional[float] = None, 
             alpha: float = 0.05) -> str:
        try:
            # Validate inputs
            if column not in self._df.columns:
                return f"❌ Column '{column}' not found. Available columns: {list(self._df.columns)}"
            
            if not pd.api.types.is_numeric_dtype(self._df[column]):
                return f"❌ Column '{column}' must be numeric for z-test"
            
            # Clean data - remove missing values
            data = self._df[column].dropna()
            if len(data) == 0:
                return f"❌ No valid data in column '{column}' after removing missing values"
            
            if len(data) < 2:
                return f"❌ Need at least 2 observations for z-test, got {len(data)}"

            # Check if sample size is appropriate for z-test
            if len(data) < 30 and population_std is None:
                return f"⚠️ Warning: Z-test is most appropriate for large samples (n≥30) or when population standard deviation is known. Your sample has n={len(data)}. Consider using t-test instead."

            result_lines = [f"📊 **Z-Test Analysis: {column}**\n"]
            
            # Perform the appropriate z-test
            if test_type == "one_sample":
                # One-sample z-test
                mean_value = data.mean()
                std_value = data.std()
                n = len(data)
                
                # Use population std if provided, otherwise use sample std
                if population_std is not None:
                    test_std = population_std
                    std_note = f"Using provided population std: {population_std:.4f}"
                else:
                    test_std = std_value
                    std_note = f"Using sample std as population estimate: {test_std:.4f}"
                
                # Calculate z-statistic
                std_error = test_std / np.sqrt(n)
                z_stat = (mean_value - null_value) / std_error
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
                
                effect_size = (mean_value - null_value) / std_value  # Cohen's d
                
                result_lines.extend([
                    f"**Test Type:** One-sample z-test",
                    f"**Null Hypothesis:** Mean = {null_value}",
                    f"**Alternative Hypothesis:** Mean ≠ {null_value}",
                    f"",
                    f"📈 **Sample Statistics:**",
                    f"  • Sample size: {n:,}",
                    f"  • Sample mean: {mean_value:.4f}",
                    f"  • Sample std: {std_value:.4f}",
                    f"  • {std_note}",
                    f"  • Difference from null: {mean_value - null_value:.4f}",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • z-statistic: {z_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Standard error: {std_error:.4f}",
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
                    title=f"One-Sample Z-Test: {column}",
                    xaxis_title=column,
                    yaxis_title="Frequency",
                    showlegend=True
                )
                
            elif test_type == "two_sample":
                # Two-sample z-test
                if not group_column or group_column not in self._df.columns:
                    return f"❌ For two-sample z-test, specify a valid group_column"
                
                # Get unique groups from the group column
                unique_groups = list(self._df[group_column].unique())
                
                # If group_values specified, use those; otherwise check for exactly 2 groups
                if group_values:
                    if len(group_values) != 2:
                        return f"❌ For two-sample z-test, group_values must contain exactly 2 values, got {len(group_values)}: {group_values}"
                    
                    # Check if specified groups exist
                    missing_groups = [g for g in group_values if g not in unique_groups]
                    if missing_groups:
                        return f"❌ Groups not found in data: {missing_groups}. Available groups: {unique_groups}"
                    
                    selected_groups = group_values
                else:
                    if len(unique_groups) != 2:
                        return f"❌ Two-sample z-test requires exactly 2 groups, found {len(unique_groups)}: {unique_groups}. Use group_values parameter to specify which 2 groups to compare."
                    
                    selected_groups = unique_groups
                
                # Extract data for the selected groups
                group1_name = selected_groups[0]
                group2_name = selected_groups[1]
                group1_data = self._df[self._df[group_column] == group1_name][column].dropna()
                group2_data = self._df[self._df[group_column] == group2_name][column].dropna()
                
                if len(group1_data) < 2 or len(group2_data) < 2:
                    return f"❌ Each group needs at least 2 observations. Group sizes: {group1_name}={len(group1_data)}, {group2_name}={len(group2_data)}"
                
                # Check sample sizes for z-test appropriateness
                if (len(group1_data) < 30 or len(group2_data) < 30) and population_std is None:
                    return f"⚠️ Warning: Z-test is most appropriate when both groups have n≥30 or population std is known. Group sizes: {group1_name}={len(group1_data)}, {group2_name}={len(group2_data)}. Consider using t-test instead."
                
                mean1, mean2 = group1_data.mean(), group2_data.mean()
                n1, n2 = len(group1_data), len(group2_data)
                
                # Calculate standard error
                if population_std is not None:
                    # Use provided population std for both groups
                    std_error = population_std * np.sqrt(1/n1 + 1/n2)
                    std_note = f"Using provided population std: {population_std:.4f}"
                else:
                    # Use pooled sample standard deviation as population estimate
                    pooled_var = ((n1-1)*group1_data.var() + (n2-1)*group2_data.var()) / (n1+n2-2)
                    std_error = np.sqrt(pooled_var * (1/n1 + 1/n2))
                    std_note = f"Using pooled sample std as population estimate: {np.sqrt(pooled_var):.4f}"
                
                z_stat = (mean1 - mean2) / std_error
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
                
                # Calculate effect size (Cohen's d for two groups)
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                     (len(group2_data) - 1) * group2_data.var()) / 
                                    (len(group1_data) + len(group2_data) - 2))
                effect_size = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                result_lines.extend([
                    f"**Test Type:** Two-sample z-test",
                    f"**Null Hypothesis:** Mean({group1_name}) = Mean({group2_name})",
                    f"**Alternative Hypothesis:** Mean({group1_name}) ≠ Mean({group2_name})",
                    f"",
                    f"📈 **Group Statistics:**",
                    f"  • **{group1_name}:** n={len(group1_data)}, mean={group1_data.mean():.4f}, std={group1_data.std():.4f}",
                    f"  • **{group2_name}:** n={len(group2_data)}, mean={group2_data.mean():.4f}, std={group2_data.std():.4f}",
                    f"  • **Difference:** {group1_data.mean() - group2_data.mean():.4f}",
                    f"  • {std_note}",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • z-statistic: {z_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Standard error: {std_error:.4f}",
                    f"  • Effect size (Cohen's d): {effect_size:.4f}",
                ])
                
                # Create box plot comparison
                fig = go.Figure()
                fig.add_trace(go.Box(y=group1_data, name=str(group1_name), boxpoints="outliers"))
                fig.add_trace(go.Box(y=group2_data, name=str(group2_name), boxpoints="outliers"))
                
                fig.update_layout(
                    title=f"Two-Sample Z-Test: {column} by {group_column}",
                    xaxis_title=group_column,
                    yaxis_title=column,
                    showlegend=True
                )
                
            elif test_type == "paired":
                # Paired z-test
                if not column2 or column2 not in self._df.columns:
                    return f"❌ For paired z-test, specify a valid column2"
                
                if not pd.api.types.is_numeric_dtype(self._df[column2]):
                    return f"❌ Column '{column2}' must be numeric for paired z-test"
                
                # Get paired data (remove rows where either value is missing)
                paired_data = self._df[[column, column2]].dropna()
                if len(paired_data) < 2:
                    return f"❌ Need at least 2 complete pairs for paired z-test, got {len(paired_data)}"
                
                data1 = paired_data[column]
                data2 = paired_data[column2]
                differences = data1 - data2
                n_pairs = len(paired_data)
                
                # Check sample size
                if n_pairs < 30 and population_std is None:
                    return f"⚠️ Warning: Paired z-test is most appropriate with n≥30 pairs or known population std of differences. You have {n_pairs} pairs. Consider using paired t-test instead."
                
                mean_diff = differences.mean()
                
                # Calculate standard error
                if population_std is not None:
                    # Use provided population std of differences
                    std_error = population_std / np.sqrt(n_pairs)
                    std_note = f"Using provided population std of differences: {population_std:.4f}"
                else:
                    # Use sample std of differences as population estimate
                    diff_std = differences.std(ddof=0)  # Population std estimate
                    std_error = diff_std / np.sqrt(n_pairs)
                    std_note = f"Using sample std of differences as population estimate: {diff_std:.4f}"
                
                z_stat = mean_diff / std_error
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
                
                effect_size = differences.mean() / differences.std()  # Cohen's d for paired data
                
                result_lines.extend([
                    f"**Test Type:** Paired z-test",
                    f"**Null Hypothesis:** Mean difference = 0",
                    f"**Alternative Hypothesis:** Mean difference ≠ 0",
                    f"",
                    f"📈 **Paired Statistics:**",
                    f"  • Pairs: {n_pairs:,}",
                    f"  • **{column}:** mean={data1.mean():.4f}, std={data1.std():.4f}",
                    f"  • **{column2}:** mean={data2.mean():.4f}, std={data2.std():.4f}",
                    f"  • **Mean difference:** {differences.mean():.4f}",
                    f"  • **Std of differences:** {differences.std():.4f}",
                    f"  • {std_note}",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • z-statistic: {z_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Standard error: {std_error:.4f}",
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
                    title=f"Paired Z-Test: {column} vs {column2}",
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
                f"⚠️ **Assumptions:**",
                f"  • Data is approximately normally distributed",
                f"  • Observations are independent",
                f"  • Data is measured at interval/ratio level",
                f"  • Large sample size (n≥30) OR population standard deviation is known"
            ])
            
            # Store chart for display
            chart_info = {
                'type': 'z_test',
                'title': f"Z-Test Analysis: {column}",
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
            return f"❌ Error in z-test analysis: {str(e)}"

    def _arun(self, column: str, test_type: str = "one_sample", null_value: float = 0, 
              group_column: Optional[str] = None, group_values: Optional[List[str]] = None, 
              column2: Optional[str] = None, population_std: Optional[float] = None, 
              alpha: float = 0.05):
        raise NotImplementedError("Async not supported")