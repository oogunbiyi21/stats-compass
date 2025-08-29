# ds_auto_insights/mcp_tools.py

from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import Tool
from langchain.tools.base import BaseTool
import pandas as pd
import re


# Tool: Get schema
class GetSchemaInput(BaseModel):
    pass


class GetSchemaTool(BaseTool):
    name: str = "get_schema"
    description: str = "Returns the names and types of columns in the uploaded dataframe."
    args_schema: Type[BaseModel] = GetSchemaInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self) -> str:
        return str(self._df.dtypes)

    def _arun(self):
        raise NotImplementedError("Async not supported")


# Tool: Get sample rows
class GetSampleRowsInput(BaseModel):
    num_rows: int


class GetSampleRowsTool(BaseTool):
    name: str = "get_sample_rows"
    description: str = "Returns a few rows from the dataframe to understand the data format."
    args_schema: Type[BaseModel] = GetSampleRowsInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, num_rows: int) -> str:
        return str(self._df.head(num_rows).to_markdown())

    def _arun(self, num_rows: int):
        raise NotImplementedError("Async not supported")


# Tool: Describe a column
class DescribeColumnInput(BaseModel):
    column_name: str


class DescribeColumnTool(BaseTool):
    name: str = "describe_column"
    description: str = "Returns descriptive statistics of a specified column."
    args_schema: Type[BaseModel] = DescribeColumnInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column_name: str) -> str:
        if column_name not in self._df.columns:
            return f"Column '{column_name}' not found."
        return str(self._df[column_name].describe(include='all'))

    def _arun(self, column_name: str):
        raise NotImplementedError("Async not supported")


class RunPandasQueryToolInput(BaseModel):
    query: str = Field(description="Python expression to run on the dataframe `df`.")

class RunPandasQueryTool(BaseTool):
    name: str = "run_pandas_query"
    description: str = "Run a safe, read-only Python expression on the dataframe `df` (e.g. df.describe(), df['col'].mean())"
    args_schema = RunPandasQueryToolInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _is_safe_expression(self, query: str) -> bool:
        # Disallow dangerous keywords and assignment operations
        banned_patterns: list[str] = [
            r"__.*?__",          # dunder methods
            r"\bimport\b",       # import statements
            r"\bexec\b",         # exec function
            r"\beval\b",         # eval function
            r"\bos\b",           # os module
            r"\bsys\b",          # sys module
            r"\bopen\b",         # open() function
            r"\bsubprocess\b",   # subprocess module
            r"\bglobals\(\)",    # globals access
            r"\blocals\(\)",     # locals access
            r"\bdel\b",          # del statement
            r"=(?!=)",           # assignment operators (but not equality ==)
            r"\+=",              # augmented assignment
            r"-=",               # augmented assignment  
            r"\*=",              # augmented assignment
            r"/=",               # augmented assignment
        ]
        return not any(re.search(pattern, query) for pattern in banned_patterns)

    def _run(self, query: str) -> str:
        if not self._is_safe_expression(query):
            # Provide specific guidance for common issues
            if "=" in query and "==" not in query:
                return "❌ Assignment operations are not allowed. This tool is for read-only data exploration. " \
                       "If you need to create a derived column (like opponent), use groupby_aggregate or top_categories tools instead."
            else:
                return "❌ Unsafe query detected. Only simple pandas expressions are allowed."

        try:
            # Set pandas display options to show columns (but limit for safety)
            max_cols = 100 if len(self._df.columns) <= 100 else 50
            with pd.option_context('display.max_columns', max_cols, 
                                   'display.width', None, 
                                   'display.max_colwidth', 50):
                local_vars: dict = {"df": self._df}
                result = eval(query, {}, local_vars)
                return str(result)
        except Exception as e:
            error_msg = str(e)
            if "invalid syntax" in error_msg.lower():
                return f"❌ Syntax error: The query contains invalid Python syntax. " \
                       f"Please use simple pandas expressions like df['column'].unique() or df.describe(). " \
                       f"Error details: {error_msg}"
            else:
                return f"❌ Error running query: {error_msg}"

# Tool: Explain a result
class NarrativeExplainInput(BaseModel):
    raw_result: str


class NarrativeExplainTool(BaseTool):
    name: str = "narrative_explain"
    description: str = "Given raw analysis results, generate a plain-English explanation."
    args_schema: Type[BaseModel] = NarrativeExplainInput

    def _run(self, inputs: NarrativeExplainInput):
        return f"Here's what the result means: {inputs.raw_result}"

    def _arun(self, inputs: NarrativeExplainInput):
        raise NotImplementedError("Async not supported")


# Tool: Get comprehensive dataset preview
class DatasetPreviewInput(BaseModel):
    num_rows: int = Field(default=5, description="Number of rows to show (max 20)")


class DatasetPreviewTool(BaseTool):
    name: str = "dataset_preview"
    description: str = "Get a comprehensive preview of the dataset showing ALL columns (no truncation). Limited to 100 columns for safety."
    args_schema: Type[BaseModel] = DatasetPreviewInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, num_rows: int = 5) -> str:
        try:
            # Safety limits
            if num_rows > 20:
                num_rows = 20
                
            num_cols = len(self._df.columns)
            if num_cols > 100:
                return f"❌ Dataset has too many columns ({num_cols}). Maximum supported: 100 columns for preview."
            
            # Set pandas display options to show columns (safe limits)
            max_cols = min(100, num_cols)
            with pd.option_context('display.max_columns', max_cols, 
                                   'display.width', None, 
                                   'display.max_colwidth', 50):
                preview = self._df.head(num_rows)
                
                result_lines = [f"📊 Dataset Preview ({len(self._df)} total rows, {num_cols} columns):"]
                
                # Show column list (but limit if too many)
                if num_cols <= 20:
                    result_lines.append(f"Columns: {list(self._df.columns)}")
                else:
                    first_10 = list(self._df.columns[:10])
                    last_10 = list(self._df.columns[-10:])
                    result_lines.append(f"Columns (first 10): {first_10}")
                    result_lines.append(f"Columns (last 10): {last_10}")
                    result_lines.append(f"... and {num_cols - 20} more columns")
                
                result_lines.append(f"\nFirst {num_rows} rows:")
                result_lines.append(str(preview))
                
                return "\n".join(result_lines)
        except Exception as e:
            return f"❌ Error creating dataset preview: {e}"

    def _arun(self, num_rows: int = 5):
        raise NotImplementedError("Async not supported")


# ========== FIRST-CLASS TOOLS ==========

# Tool: Group by and aggregate
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


# Tool: Get top categories
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


# Tool: Generate histogram data
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


# Tool: Correlation matrix
class CorrelationMatrixInput(BaseModel):
    columns: list[str] = Field(default=None, description="Specific columns to include (optional)")
    method: str = Field(default="pearson", description="Correlation method: pearson, spearman, or kendall")


class CorrelationMatrixTool(BaseTool):
    name: str = "correlation_matrix"
    description: str = "Calculate correlation matrix for numeric columns."
    args_schema: Type[BaseModel] = CorrelationMatrixInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, columns: list = None, method: str = "pearson") -> str:
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
            result_lines.append("\nCorrelation Matrix:")
            result_lines.append(corr_matrix.to_string())

            # Highlight strong correlations (exclude diagonal)
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        strong_corrs.append(f"{col1} ↔ {col2}: {corr_val:.3f}")

            if strong_corrs:
                result_lines.append(f"\n🔍 Strong correlations (|r| > 0.7):")
                for corr in strong_corrs:
                    result_lines.append(f"  {corr}")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error calculating correlation: {e}"

    def _arun(self, columns: list = None, method: str = "pearson"):
        raise NotImplementedError("Async not supported")


# Tool: Create Histogram Chart
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
            result += f"  • Mean: {col_data.mean():.2f}\n"
            result += f"  • Median: {col_data.median():.2f}\n"
            result += f"  • Std Dev: {col_data.std():.2f}\n"
            result += f"  • Range: {col_data.min():.2f} to {col_data.max():.2f}\n\n"
            
            # Store chart data for Streamlit rendering
            import streamlit as st
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'histogram',
                    'data': chart_data,
                    'column': column,
                    'title': chart_title
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "Chart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating histogram: {e}"

    def _arun(self, column: str, bins: int = 30, title: str = ""):
        raise NotImplementedError("Async not supported")


# Tool: Create Bar Chart
class CreateBarChartInput(BaseModel):
    column: str = Field(description="Column name to create bar chart for")
    top_n: int = Field(default=10, description="Number of top categories to show")
    title: str = Field(default="", description="Custom title for the chart")


class CreateBarChartTool(BaseTool):
    name: str = "create_bar_chart"
    description: str = "Creates a bar chart for categorical data showing top categories by count."
    args_schema: Type[BaseModel] = CreateBarChartInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column: str, top_n: int = 10, title: str = "") -> str:
        try:
            if column not in self._df.columns:
                return f"❌ Column '{column}' not found. Available columns: {list(self._df.columns)}"

            value_counts = self._df[column].value_counts().head(top_n)
            
            if len(value_counts) == 0:
                return f"❌ No data found in column '{column}'"

            # Create chart data
            chart_data = pd.DataFrame({
                'category': value_counts.index.astype(str),
                'count': value_counts.values
            })

            chart_title = title or f"Top {top_n} {column} by Count"
            
            result = f"📊 {chart_title}\n\n"
            result += f"Total unique values: {self._df[column].nunique()}\n"
            result += f"Showing top {len(value_counts)} categories:\n\n"
            
            for i, (cat, count) in enumerate(value_counts.items(), 1):
                percentage = (count / len(self._df)) * 100
                result += f"  {i}. {cat}: {count} ({percentage:.1f}%)\n"
            
            # Store chart data for Streamlit rendering
            import streamlit as st
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'bar',
                    'data': chart_data,
                    'column': column,
                    'title': chart_title
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "\nChart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating bar chart: {e}"

    def _arun(self, column: str, top_n: int = 10, title: str = ""):
        raise NotImplementedError("Async not supported")


# Tool: Create Scatter Plot
class CreateScatterPlotInput(BaseModel):
    x_column: str = Field(description="Column name for x-axis")
    y_column: str = Field(description="Column name for y-axis")
    color_column: str = Field(default="", description="Optional column for color coding")
    title: str = Field(default="", description="Custom title for the chart")


class CreateScatterPlotTool(BaseTool):
    name: str = "create_scatter_plot"
    description: str = "Creates a scatter plot to show relationship between two numeric columns."
    args_schema: Type[BaseModel] = CreateScatterPlotInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, x_column: str, y_column: str, color_column: str = "", title: str = "") -> str:
        try:
            # Validate columns
            if x_column not in self._df.columns:
                return f"❌ X column '{x_column}' not found. Available columns: {list(self._df.columns)}"
            if y_column not in self._df.columns:
                return f"❌ Y column '{y_column}' not found. Available columns: {list(self._df.columns)}"
            
            if color_column and color_column not in self._df.columns:
                return f"❌ Color column '{color_column}' not found. Available columns: {list(self._df.columns)}"

            # Check if columns are numeric
            if not pd.api.types.is_numeric_dtype(self._df[x_column]):
                return f"❌ X column '{x_column}' must be numeric"
            if not pd.api.types.is_numeric_dtype(self._df[y_column]):
                return f"❌ Y column '{y_column}' must be numeric"

            # Prepare data
            plot_data = self._df[[x_column, y_column]].dropna()
            if color_column:
                plot_data = self._df[[x_column, y_column, color_column]].dropna()
            
            if len(plot_data) == 0:
                return f"❌ No complete data pairs found for {x_column} and {y_column}"

            chart_title = title or f"{y_column} vs {x_column}"
            
            # Calculate correlation
            correlation = plot_data[x_column].corr(plot_data[y_column])
            
            result = f"📊 {chart_title}\n\n"
            result += f"Data points: {len(plot_data)}\n"
            result += f"Correlation: {correlation:.3f}\n"
            
            if abs(correlation) > 0.7:
                result += "🔍 Strong correlation detected!\n"
            elif abs(correlation) > 0.3:
                result += "📈 Moderate correlation detected.\n"
            else:
                result += "📊 Weak correlation.\n"
            
            result += f"\n{x_column} range: {plot_data[x_column].min():.2f} to {plot_data[x_column].max():.2f}\n"
            result += f"{y_column} range: {plot_data[y_column].min():.2f} to {plot_data[y_column].max():.2f}\n"

            # Store chart data for Streamlit rendering
            import streamlit as st
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'scatter',
                    'data': plot_data,
                    'x_column': x_column,
                    'y_column': y_column,
                    'color_column': color_column if color_column else None,
                    'title': chart_title,
                    'correlation': correlation
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "\nChart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating scatter plot: {e}"

    def _arun(self, x_column: str, y_column: str, color_column: str = "", title: str = ""):
        raise NotImplementedError("Async not supported")


# Tool: Create Line Chart
class CreateLineChartInput(BaseModel):
    x_column: str = Field(description="Column name for x-axis (typically time/date)")
    y_column: str = Field(description="Column name for y-axis")
    title: str = Field(default="", description="Custom title for the chart")


class CreateLineChartTool(BaseTool):
    name: str = "create_line_chart"
    description: str = "Creates a line chart to show trends over time or ordered data."
    args_schema: Type[BaseModel] = CreateLineChartInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, x_column: str, y_column: str, title: str = "") -> str:
        try:
            # Validate columns
            if x_column not in self._df.columns:
                return f"❌ X column '{x_column}' not found. Available columns: {list(self._df.columns)}"
            if y_column not in self._df.columns:
                return f"❌ Y column '{y_column}' not found. Available columns: {list(self._df.columns)}"

            # Check if y column is numeric
            if not pd.api.types.is_numeric_dtype(self._df[y_column]):
                return f"❌ Y column '{y_column}' must be numeric"

            # Prepare data
            plot_data = self._df[[x_column, y_column]].dropna().sort_values(x_column)
            
            if len(plot_data) == 0:
                return f"❌ No complete data found for {x_column} and {y_column}"

            chart_title = title or f"{y_column} over {x_column}"
            
            result = f"📊 {chart_title}\n\n"
            result += f"Data points: {len(plot_data)}\n"
            result += f"{y_column} range: {plot_data[y_column].min():.2f} to {plot_data[y_column].max():.2f}\n"
            
            # Calculate trend
            if len(plot_data) > 1:
                first_val = plot_data[y_column].iloc[0]
                last_val = plot_data[y_column].iloc[-1]
                change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                
                if change > 5:
                    result += f"📈 Upward trend: +{change:.1f}%\n"
                elif change < -5:
                    result += f"📉 Downward trend: {change:.1f}%\n"
                else:
                    result += f"➡️ Relatively stable: {change:.1f}%\n"

            # Store chart data for Streamlit rendering
            import streamlit as st
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'line',
                    'data': plot_data,
                    'x_column': x_column,
                    'y_column': y_column,
                    'title': chart_title
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "\nChart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating line chart: {e}"

    def _arun(self, x_column: str, y_column: str, title: str = ""):
        raise NotImplementedError("Async not supported")


# Tool: Create New Column
class CreateColumnInput(BaseModel):
    column_name: str = Field(description="Name of the new column to create")
    operation: str = Field(description="The pandas operation to create the column. Examples: 'df[\"goals\"] * 2', 'df[\"goals\"].apply(lambda x: \"High\" if x > 10 else \"Low\")', 'df[\"h_team\"] + \" vs \" + df[\"a_team\"]'")
    description: str = Field(default="", description="Description of what this column represents")


class CreateColumnTool(BaseTool):
    name: str = "create_column"
    description: str = """Create a new column in the dataset using pandas operations.
    This tool allows you to add calculated columns, conditional columns, or transform existing data.
    Use this for complex data transformations that require creating new variables."""
    args_schema: Type[BaseModel] = CreateColumnInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _run(self, column_name: str, operation: str, description: str = "") -> str:
        try:
            import streamlit as st
            import numpy as np
            import pandas as pd
            
            # Safety checks
            if column_name in self._df.columns:
                return f"❌ Column '{column_name}' already exists. Choose a different name or use a different operation."
            
            # Check for dangerous operations
            dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file', 'system', 'os.', 'subprocess', 'globals', 'locals']
            if any(keyword in operation.lower() for keyword in dangerous_keywords):
                return f"❌ Operation contains potentially dangerous keywords. Please use only pandas operations."
            
            # Ensure operation is a valid pandas expression
            if not (operation.strip().startswith('df[') or operation.strip().startswith('df.') or 
                   any(func in operation for func in ['np.', 'pd.', 'lambda', '"', "'"])):
                return f"❌ Operation must start with 'df[' or 'df.' or use allowed functions (np., pd., lambda). Got: {operation}"
            
            # Create a safe namespace for evaluation
            safe_dict = {
                'df': self._df,
                'pd': pd,
                'np': np,
                '__builtins__': {'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool, 'max': max, 'min': min}
            }
            
            # Execute the operation
            try:
                result = eval(operation, safe_dict)
                
                # Validate result is appropriate for a column
                if hasattr(result, '__len__') and len(result) != len(self._df):
                    return f"❌ Operation result length ({len(result)}) doesn't match dataframe length ({len(self._df)})"
                
                # Add the new column
                self._df[column_name] = result
                
                # Update the dataframe in session state if available
                if hasattr(st, 'session_state') and 'uploaded_df' in st.session_state:
                    st.session_state.uploaded_df = self._df
                
                # Show preview of new column
                preview = self._df[[column_name]].head(10)
                
                success_msg = f"✅ Created column '{column_name}'"
                if description:
                    success_msg += f" - {description}"
                
                success_msg += f"\n\nPreview of new column (first 10 rows):\n{preview.to_string()}"
                
                # Show basic stats if numeric
                if pd.api.types.is_numeric_dtype(self._df[column_name]):
                    stats = self._df[column_name].describe()
                    success_msg += f"\n\nColumn statistics:\n{stats.to_string()}"
                else:
                    # Show value counts for categorical
                    value_counts = self._df[column_name].value_counts().head(5)
                    success_msg += f"\n\nTop 5 values:\n{value_counts.to_string()}"
                
                return success_msg
                
            except Exception as eval_error:
                return f"❌ Error executing operation: {str(eval_error)}. Please check your pandas syntax."
            
        except Exception as e:
            return f"❌ Error creating column: {str(e)}"
    
    def _arun(self, column_name: str, operation: str, description: str = ""):
        raise NotImplementedError("Async not supported")
