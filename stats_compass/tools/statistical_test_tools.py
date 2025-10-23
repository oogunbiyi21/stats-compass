# stats_compass/tools/statistical_test_tools.py
"""
Statistical test tools for DS Auto Insights.
Provides T-test and Z-test implementations for hypothesis testing.
"""

from typing import Type, Optional, List, Tuple
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


class BaseHypothesisTestTool(BaseTool):
    """Base class for hypothesis testing tools (T-test, Z-test).
    Extracts common validation, formatting, and visualization logic."""
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _validate_column(self, column: str) -> Tuple[bool, str, pd.Series]:
        """Validate column exists, is numeric, and has sufficient data.
        Returns: (is_valid, error_message, cleaned_data)"""
        if column not in self._df.columns:
            return False, f"❌ Column '{column}' not found. Available columns: {list(self._df.columns)}", None
        
        if not pd.api.types.is_numeric_dtype(self._df[column]):
            return False, f"❌ Column '{column}' must be numeric for hypothesis testing", None
        
        data = self._df[column].dropna()
        if len(data) == 0:
            return False, f"❌ No valid data in column '{column}' after removing missing values", None
        
        if len(data) < 2:
            return False, f"❌ Need at least 2 observations, got {len(data)}", None
        
        return True, "", data
    
    def _extract_group_data(self, column: str, group_column: str, group_values: Optional[List[str]]) -> Tuple[bool, str, any, any, any, any]:
        """Extract data for two groups. Handles group matching gracefully.
        Returns: (is_valid, error_message, group1_name, group2_name, group1_data, group2_data)"""
        if not group_column or group_column not in self._df.columns:
            return False, f"❌ For two-sample test, specify a valid group_column", None, None, None, None
        
        unique_groups = list(self._df[group_column].dropna().unique())
        
        # If group_values specified, use those; otherwise check for exactly 2 groups
        if group_values:
            if len(group_values) != 2:
                return False, f"❌ For two-sample test, group_values must contain exactly 2 values, got {len(group_values)}: {group_values}", None, None, None, None
            
            # Let pandas handle type matching - filter and check if data exists
            group1_data = self._df[self._df[group_column] == group_values[0]][column].dropna()
            group2_data = self._df[self._df[group_column] == group_values[1]][column].dropna()
            
            if len(group1_data) == 0:
                return False, f"❌ No data found for group '{group_values[0]}'. Available groups: {unique_groups}", None, None, None, None
            if len(group2_data) == 0:
                return False, f"❌ No data found for group '{group_values[1]}'. Available groups: {unique_groups}", None, None, None, None
            
            selected_groups = group_values
        else:
            if len(unique_groups) != 2:
                return False, f"❌ Two-sample test requires exactly 2 groups, found {len(unique_groups)}: {unique_groups}. Use group_values parameter to specify which 2 groups to compare.", None, None, None, None
            
            selected_groups = unique_groups
            group1_data = self._df[self._df[group_column] == selected_groups[0]][column].dropna()
            group2_data = self._df[self._df[group_column] == selected_groups[1]][column].dropna()
        
        # Validate group sizes
        if len(group1_data) < 2 or len(group2_data) < 2:
            return False, f"❌ Each group needs at least 2 observations. Group sizes: {selected_groups[0]}={len(group1_data)}, {selected_groups[1]}={len(group2_data)}", None, None, None, None
        
        return True, "", selected_groups[0], selected_groups[1], group1_data, group2_data
    
    def _calculate_effect_size(self, test_type: str, data1, data2=None, null_value=0) -> float:
        """Calculate Cohen's d effect size."""
        if test_type == "one_sample":
            return (data1.mean() - null_value) / data1.std()
        elif test_type == "two_sample":
            pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                                 (len(data2) - 1) * data2.var()) / 
                                (len(data1) + len(data2) - 2))
            return (data1.mean() - data2.mean()) / pooled_std
        elif test_type == "paired":
            differences = data1 - data2
            return differences.mean() / differences.std()
        return 0
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size magnitude."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _format_results(self, result_lines: List[str], p_value: float, effect_size: float, 
                       alpha: float, test_name: str, test_type: str) -> List[str]:
        """Add interpretation and assumptions to results."""
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
        
        effect_magnitude = self._interpret_effect_size(effect_size)
        result_lines.extend([
            f"  • **Effect size:** {effect_magnitude} (|d| = {abs(effect_size):.3f})",
            f"",
            f"⚠️ **Assumptions:**",
            f"  • Data is approximately normally distributed",
            f"  • Observations are independent",
            f"  • Data is measured at interval/ratio level"
        ])
        
        if test_type == "two_sample" and test_name == "t-test":
            result_lines.append(f"  • Unequal variances assumed (Welch's t-test)")
        elif test_name == "z-test":
            result_lines.append(f"  • Large sample size (n≥30) OR population standard deviation is known")
        
        return result_lines
    
    def _create_one_sample_viz(self, data: pd.Series, mean_value: float, null_value: float, 
                               column: str, test_name: str) -> go.Figure:
        """Create histogram visualization for one-sample test."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            name="Data Distribution",
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.add_vline(x=mean_value, line_dash="dash", line_color="red", 
                     annotation_text=f"Sample Mean: {mean_value:.3f}")
        fig.add_vline(x=null_value, line_dash="solid", line_color="blue", 
                     annotation_text=f"Null Value: {null_value}")
        
        fig.update_layout(
            title=f"One-Sample {test_name.title()}: {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            showlegend=True
        )
        
        return fig
    
    def _create_two_sample_viz(self, group1_data: pd.Series, group2_data: pd.Series,
                               group1_name: str, group2_name: str, column: str, 
                               group_column: str, test_name: str) -> go.Figure:
        """Create box plot visualization for two-sample test."""
        fig = go.Figure()
        fig.add_trace(go.Box(y=group1_data, name=str(group1_name), boxpoints="outliers"))
        fig.add_trace(go.Box(y=group2_data, name=str(group2_name), boxpoints="outliers"))
        
        fig.update_layout(
            title=f"Two-Sample {test_name.title()}: {column} by {group_column}",
            xaxis_title=group_column,
            yaxis_title=column,
            showlegend=True
        )
        
        return fig
    
    def _create_paired_viz(self, data1: pd.Series, data2: pd.Series, 
                          column: str, column2: str, test_name: str) -> go.Figure:
        """Create scatter plot visualization for paired test."""
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
            title=f"Paired {test_name.title()}: {column} vs {column2}",
            xaxis_title=column,
            yaxis_title=column2,
            showlegend=True
        )
        
        return fig
    
    def _store_chart(self, chart_type: str, title: str, fig: go.Figure, test_type: str):
        """Store chart in session state for display."""
        chart_info = {
            'type': chart_type,
            'title': title,
            'data': None,
            'figure': fig,  # Standardized to 'figure' (more descriptive than 'fig')
            'chart_config': {
                'chart_type': 'statistical_test',
                'test_type': test_type
            }
        }
        
        if hasattr(st, 'session_state'):
            if 'current_response_charts' not in st.session_state:
                st.session_state.current_response_charts = []
            st.session_state.current_response_charts.append(chart_info)


class TTestInput(BaseModel):
    column: str = Field(description="Name of the numeric column to test")
    test_type: str = Field(default="one_sample", description="Type of t-test: 'one_sample', 'two_sample', or 'paired'")
    null_value: Optional[float] = Field(default=0, description="Null hypothesis value for one-sample test")
    group_column: Optional[str] = Field(default=None, description="Column to group by for two-sample test")
    group_values: Optional[List[str]] = Field(default=None, description="For two-sample test: specify exactly 2 group values to compare")
    column2: Optional[str] = Field(default=None, description="Second column for paired t-test")
    alpha: float = Field(default=0.05, description="Significance level (default 0.05)")


class RunTTestTool(BaseHypothesisTestTool):
    name: str = "run_t_test"
    description: str = "Perform t-tests (one-sample, two-sample, or paired) to test hypotheses about means. Uses t-distribution for small samples or unknown population variance. Includes visualizations and practical interpretation."
    args_schema: Type[BaseModel] = TTestInput

    def _run(self, column: str, test_type: str = "one_sample", null_value: float = 0, 
             group_column: Optional[str] = None, group_values: Optional[List[str]] = None, 
             column2: Optional[str] = None, alpha: float = 0.05) -> str:
        try:
            # Validate primary column
            is_valid, error_msg, data = self._validate_column(column)
            if not is_valid:
                return error_msg
            
            result_lines = [f"📊 **T-Test Analysis: {column}**\n"]
            
            # Perform the appropriate t-test
            if test_type == "one_sample":
                t_stat, p_value = stats.ttest_1samp(data, null_value)
                degrees_freedom = len(data) - 1
                mean_value = data.mean()
                std_value = data.std()
                effect_size = self._calculate_effect_size("one_sample", data, null_value=null_value)
                
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
                
                fig = self._create_one_sample_viz(data, mean_value, null_value, column, "t-test")
                
            elif test_type == "two_sample":
                is_valid, error_msg, group1_name, group2_name, group1_data, group2_data = self._extract_group_data(column, group_column, group_values)
                if not is_valid:
                    return error_msg
                
                # Perform Welch's t-test (unequal variances assumed)
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                effect_size = self._calculate_effect_size("two_sample", group1_data, group2_data)
                
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
                
                fig = self._create_two_sample_viz(group1_data, group2_data, group1_name, group2_name, column, group_column, "t-test")
                
            elif test_type == "paired":
                if not column2 or column2 not in self._df.columns:
                    return f"❌ For paired t-test, specify a valid column2"
                
                if not pd.api.types.is_numeric_dtype(self._df[column2]):
                    return f"❌ Column '{column2}' must be numeric for paired t-test"
                
                paired_data = self._df[[column, column2]].dropna()
                if len(paired_data) < 2:
                    return f"❌ Need at least 2 complete pairs for paired t-test, got {len(paired_data)}"
                
                data1 = paired_data[column]
                data2 = paired_data[column2]
                differences = data1 - data2
                
                t_stat, p_value = stats.ttest_rel(data1, data2)
                effect_size = self._calculate_effect_size("paired", data1, data2)
                
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
                
                fig = self._create_paired_viz(data1, data2, column, column2, "t-test")
                
            else:
                return f"❌ Invalid test_type '{test_type}'. Use: 'one_sample', 'two_sample', or 'paired'"
            
            # Add interpretation and assumptions using base class
            result_lines = self._format_results(result_lines, p_value, effect_size, alpha, "t-test", test_type)
            
            # Store chart using base class
            self._store_chart('t_test', f"T-Test Analysis: {column}", fig, test_type)
            
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


class RunZTestTool(BaseHypothesisTestTool):
    name: str = "run_z_test"
    description: str = "Perform z-tests (one-sample, two-sample, or paired) to test hypotheses about means. Uses normal distribution - appropriate for large samples (n≥30) or when population standard deviation is known. Includes visualizations and practical interpretation."
    args_schema: Type[BaseModel] = ZTestInput

    def _run(self, column: str, test_type: str = "one_sample", null_value: float = 0, 
             group_column: Optional[str] = None, group_values: Optional[List[str]] = None, 
             column2: Optional[str] = None, population_std: Optional[float] = None, 
             alpha: float = 0.05) -> str:
        try:
            # Validate primary column
            is_valid, error_msg, data = self._validate_column(column)
            if not is_valid:
                return error_msg
            
            # Check sample size - convert blocking error to warning
            warnings = []
            if len(data) < 10:
                return f"❌ Sample too small for reliable inference (n < 10). Got n={len(data)}"
            if len(data) < 30 and population_std is None:
                warnings.append(f"⚠️ Small sample size (n={len(data)}) - z-test most appropriate for n≥30 or known population std. Consider t-test instead.")
            
            result_lines = [f"📊 **Z-Test Analysis: {column}**\n"]
            
            # Add warnings if any
            if warnings:
                result_lines.extend(warnings)
                result_lines.append("")
            
            # Perform the appropriate z-test
            if test_type == "one_sample":
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
                
                effect_size = self._calculate_effect_size("one_sample", data, null_value=null_value)
                
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
                
                fig = self._create_one_sample_viz(data, mean_value, null_value, column, "z-test")
                
            elif test_type == "two_sample":
                is_valid, error_msg, group1_name, group2_name, group1_data, group2_data = self._extract_group_data(column, group_column, group_values)
                if not is_valid:
                    return error_msg
                
                # Check sample sizes - warnings instead of blocking
                if len(group1_data) < 10 or len(group2_data) < 10:
                    return f"❌ Sample too small for reliable inference (need n≥10 per group). Got: {group1_name}={len(group1_data)}, {group2_name}={len(group2_data)}"
                if (len(group1_data) < 30 or len(group2_data) < 30) and population_std is None:
                    warnings.append(f"⚠️ Small sample sizes ({group1_name}={len(group1_data)}, {group2_name}={len(group2_data)}) - z-test most appropriate when n≥30 per group. Consider t-test.")
                    result_lines.extend(warnings)
                    result_lines.append("")
                
                mean1, mean2 = group1_data.mean(), group2_data.mean()
                n1, n2 = len(group1_data), len(group2_data)
                
                # Calculate standard error
                if population_std is not None:
                    std_error = population_std * np.sqrt(1/n1 + 1/n2)
                    std_note = f"Using provided population std: {population_std:.4f}"
                else:
                    pooled_var = ((n1-1)*group1_data.var() + (n2-1)*group2_data.var()) / (n1+n2-2)
                    std_error = np.sqrt(pooled_var * (1/n1 + 1/n2))
                    std_note = f"Using pooled sample std as population estimate: {np.sqrt(pooled_var):.4f}"
                
                z_stat = (mean1 - mean2) / std_error
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
                
                effect_size = self._calculate_effect_size("two_sample", group1_data, group2_data)
                
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
                
                fig = self._create_two_sample_viz(group1_data, group2_data, group1_name, group2_name, column, group_column, "z-test")
                
            elif test_type == "paired":
                if not column2 or column2 not in self._df.columns:
                    return f"❌ For paired z-test, specify a valid column2"
                
                if not pd.api.types.is_numeric_dtype(self._df[column2]):
                    return f"❌ Column '{column2}' must be numeric for paired z-test"
                
                paired_data = self._df[[column, column2]].dropna()
                if len(paired_data) < 2:
                    return f"❌ Need at least 2 complete pairs for paired z-test, got {len(paired_data)}"
                
                data1 = paired_data[column]
                data2 = paired_data[column2]
                differences = data1 - data2
                n_pairs = len(paired_data)
                
                # Check sample size - warning instead of blocking
                if n_pairs < 10:
                    return f"❌ Sample too small for reliable inference (need n≥10 pairs). Got n={n_pairs}"
                if n_pairs < 30 and population_std is None:
                    warnings.append(f"⚠️ Small sample size ({n_pairs} pairs) - z-test most appropriate for n≥30. Consider paired t-test.")
                    result_lines.extend(warnings)
                    result_lines.append("")
                
                mean_diff = differences.mean()
                
                # Calculate standard error
                if population_std is not None:
                    std_error = population_std / np.sqrt(n_pairs)
                    std_note = f"Using provided population std of differences: {population_std:.4f}"
                else:
                    diff_std = differences.std(ddof=0)  # Population std estimate
                    std_error = diff_std / np.sqrt(n_pairs)
                    std_note = f"Using sample std of differences as population estimate: {diff_std:.4f}"
                
                z_stat = mean_diff / std_error
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
                
                effect_size = self._calculate_effect_size("paired", data1, data2)
                
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
                
                fig = self._create_paired_viz(data1, data2, column, column2, "z-test")
                
            else:
                return f"❌ Invalid test_type '{test_type}'. Use: 'one_sample', 'two_sample', or 'paired'"
            
            # Add interpretation and assumptions using base class
            result_lines = self._format_results(result_lines, p_value, effect_size, alpha, "z-test", test_type)
            
            # Store chart using base class
            self._store_chart('z_test', f"Z-Test Analysis: {column}", fig, test_type)
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in z-test analysis: {str(e)}"
        except Exception as e:
            return f"❌ Error in z-test analysis: {str(e)}"

    def _arun(self, column: str, test_type: str = "one_sample", null_value: float = 0, 
              group_column: Optional[str] = None, group_values: Optional[List[str]] = None, 
              column2: Optional[str] = None, population_std: Optional[float] = None, 
              alpha: float = 0.05):
        raise NotImplementedError("Async not supported")


class ChiSquareTestInput(BaseModel):
    test_type: str = Field(default="independence", description="Type of chi-square test: 'independence' or 'goodness_of_fit'")
    column1: str = Field(description="First categorical column for independence test, or the column to test for goodness of fit")
    column2: Optional[str] = Field(default=None, description="Second categorical column for independence test (not used for goodness of fit)")
    expected_frequencies: Optional[List[float]] = Field(default=None, description="Expected frequencies for goodness of fit test (if not provided, equal frequencies assumed)")
    alpha: float = Field(default=0.05, description="Significance level (default 0.05)")


class RunChiSquareTestTool(BaseTool):
    name: str = "run_chi_square_test"
    description: str = "Perform chi-square tests for independence (relationship between two categorical variables) or goodness of fit (whether data follows expected distribution). Includes comprehensive statistical analysis and visualizations."
    args_schema: Type[BaseModel] = ChiSquareTestInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, test_type: str = "independence", column1: str = None, column2: str = None, 
             expected_frequencies: Optional[List[float]] = None, alpha: float = 0.05) -> str:
        try:
            # Validate inputs
            if column1 not in self._df.columns:
                return f"❌ Column '{column1}' not found. Available columns: {list(self._df.columns)}"
            
            if test_type not in ["independence", "goodness_of_fit"]:
                return f"❌ Invalid test_type '{test_type}'. Use: 'independence' or 'goodness_of_fit'"

            result_lines = [f"📊 **Chi-Square Test Analysis**\n"]
            
            if test_type == "independence":
                # Chi-square test of independence
                if not column2 or column2 not in self._df.columns:
                    return f"❌ For independence test, specify a valid column2"
                
                # Create contingency table
                contingency_table = pd.crosstab(self._df[column1], self._df[column2])
                
                # Check if we have enough data
                if contingency_table.size == 0:
                    return f"❌ No data available for contingency table"
                
                # Check for minimum expected frequencies (rule of thumb: all expected frequencies >= 5)
                chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
                
                # Check assumptions
                low_expected = (expected_freq < 5).sum()
                total_cells = expected_freq.size
                low_expected_pct = (low_expected / total_cells) * 100
                
                result_lines.extend([
                    f"**Test Type:** Chi-square test of independence",
                    f"**Null Hypothesis:** {column1} and {column2} are independent (no relationship)",
                    f"**Alternative Hypothesis:** {column1} and {column2} are not independent (there is a relationship)",
                    f"",
                    f"📈 **Contingency Table:**",
                    f"```",
                    str(contingency_table),
                    f"```",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • Chi-square statistic: {chi2_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Degrees of freedom: {dof}",
                    f"  • Sample size: {contingency_table.sum().sum():,}",
                ])
                
                # Calculate effect size (Cramér's V)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                
                result_lines.extend([
                    f"  • Cramér's V (effect size): {cramers_v:.4f}",
                    f"",
                    f"📋 **Expected Frequencies:**",
                    f"```",
                    str(pd.DataFrame(expected_freq, 
                                   index=contingency_table.index, 
                                   columns=contingency_table.columns).round(2)),
                    f"```",
                ])
                
                # Create visualization - heatmap of observed vs expected
                fig = go.Figure()
                
                # Add heatmap of observed frequencies
                fig.add_trace(go.Heatmap(
                    z=contingency_table.values,
                    x=contingency_table.columns,
                    y=contingency_table.index,
                    text=contingency_table.values,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorscale='Blues',
                    name="Observed Frequencies",
                    hovertemplate='<b>%{y}</b> & <b>%{x}</b><br>Observed: %{z}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"Chi-Square Independence Test: {column1} vs {column2}<br>Observed Frequencies",
                    xaxis_title=column2,
                    yaxis_title=column1,
                    height=max(400, len(contingency_table.index) * 50),
                    width=max(600, len(contingency_table.columns) * 100)
                )
                
            elif test_type == "goodness_of_fit":
                # Chi-square goodness of fit test
                # Get value counts for the column
                observed_freq = self._df[column1].value_counts().sort_index()
                categories = observed_freq.index.tolist()
                observed_values = observed_freq.values
                n_total = observed_values.sum()
                
                if len(observed_values) < 2:
                    return f"❌ Need at least 2 categories for goodness of fit test, found {len(observed_values)}"
                
                # Determine expected frequencies
                if expected_frequencies is not None:
                    # Validate expected frequencies
                    if any(f <= 0 for f in expected_frequencies):
                        return "❌ Expected frequencies must all be positive (greater than 0)"
                    
                    if len(expected_frequencies) != len(observed_values):
                        return f"❌ Expected frequencies length ({len(expected_frequencies)}) must match number of categories ({len(observed_values)})"
                    
                    # Normalize expected frequencies to sum to total observations
                    expected_prop = np.array(expected_frequencies) / np.sum(expected_frequencies)
                    expected_values = expected_prop * n_total
                    expected_note = "Using provided expected frequencies"
                else:
                    # Assume equal frequencies (uniform distribution)
                    expected_values = np.full(len(observed_values), n_total / len(observed_values))
                    expected_note = "Using equal expected frequencies (uniform distribution)"
                
                # Perform chi-square test
                chi2_stat, p_value = stats.chisquare(observed_values, expected_values)
                dof = len(observed_values) - 1
                
                # Check assumptions
                low_expected = (expected_values < 5).sum()
                low_expected_pct = (low_expected / len(expected_values)) * 100
                
                result_lines.extend([
                    f"**Test Type:** Chi-square goodness of fit test",
                    f"**Null Hypothesis:** Data follows the expected distribution",
                    f"**Alternative Hypothesis:** Data does not follow the expected distribution", 
                    f"",
                    f"📈 **Frequency Comparison:**",
                ])
                
                # Create comparison table
                comparison_df = pd.DataFrame({
                    'Category': categories,
                    'Observed': observed_values,
                    'Expected': expected_values.round(2),
                    'Difference': (observed_values - expected_values).round(2),
                    'Contribution to Chi²': ((observed_values - expected_values)**2 / expected_values).round(4)
                })
                
                result_lines.extend([
                    f"```",
                    str(comparison_df),
                    f"```",
                    f"",
                    f"🧮 **Test Results:**",
                    f"  • Chi-square statistic: {chi2_stat:.4f}",
                    f"  • p-value: {p_value:.6f}",
                    f"  • Degrees of freedom: {dof}",
                    f"  • Sample size: {n_total:,}",
                    f"  • {expected_note}",
                ])
                
                # Create bar chart comparing observed vs expected
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=observed_values,
                    name='Observed',
                    marker_color='lightblue',
                    opacity=0.8
                ))
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=expected_values,
                    name='Expected',
                    marker_color='orange',
                    opacity=0.8
                ))
                
                fig.update_layout(
                    title=f"Chi-Square Goodness of Fit Test: {column1}<br>Observed vs Expected Frequencies",
                    xaxis_title=column1,
                    yaxis_title="Frequency",
                    barmode='group',
                    showlegend=True,
                    height=500
                )
            
            # Interpret results (common for both tests)
            result_lines.extend([
                f"",
                f"🎯 **Statistical Interpretation:**"
            ])
            
            if p_value < alpha:
                result_lines.append(f"  • **Significant result** (p < {alpha}): Reject null hypothesis")
                if test_type == "independence":
                    result_lines.append(f"  • There IS a statistically significant relationship between {column1} and {column2}")
                else:
                    result_lines.append(f"  • The data does NOT follow the expected distribution")
            else:
                result_lines.append(f"  • **Not significant** (p ≥ {alpha}): Fail to reject null hypothesis")
                if test_type == "independence":
                    result_lines.append(f"  • There is NO statistically significant relationship between {column1} and {column2}")
                else:
                    result_lines.append(f"  • The data DOES follow the expected distribution")
            
            # Effect size interpretation (for independence test)
            if test_type == "independence":
                if cramers_v < 0.1:
                    effect_magnitude = "negligible"
                elif cramers_v < 0.3:
                    effect_magnitude = "small"
                elif cramers_v < 0.5:
                    effect_magnitude = "medium"
                else:
                    effect_magnitude = "large"
                
                result_lines.append(f"  • **Effect size (Cramér's V):** {effect_magnitude} ({cramers_v:.3f})")
            
            # Assumptions check
            result_lines.extend([
                f"",
                f"⚠️ **Assumptions:**",
                f"  • Observations are independent",
                f"  • Data consists of frequencies/counts",
                f"  • Expected frequency in each cell ≥ 5 (ideally)"
            ])
            
            if low_expected > 0:
                result_lines.append(f"  • ⚠️ Warning: {low_expected} cells ({low_expected_pct:.1f}%) have expected frequency < 5")
                result_lines.append(f"    Consider combining categories or using exact tests for small samples")
            
            # Store chart for display
            chart_info = {
                'type': 'chi_square_test',
                'title': f"Chi-Square Test: {column1}" + (f" vs {column2}" if test_type == "independence" else ""),
                'data': None,
                'figure': fig,  # Standardized to 'figure' (more descriptive than 'fig')
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
            return f"❌ Error in chi-square test analysis: {str(e)}"

    def _arun(self, test_type: str = "independence", column1: str = None, column2: str = None, 
              expected_frequencies: Optional[List[float]] = None, alpha: float = 0.05):
        raise NotImplementedError("Async not supported")
