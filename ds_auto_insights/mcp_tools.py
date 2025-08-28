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

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def _run(self, _: GetSchemaInput):
        return str(self.df.dtypes)

    def _arun(self, _: GetSchemaInput):
        raise NotImplementedError("Async not supported")


# Tool: Get sample rows
class GetSampleRowsInput(BaseModel):
    num_rows: int


class GetSampleRowsTool(BaseTool):
    name: str = "get_sample_rows"
    description: str = "Returns a few rows from the dataframe to understand the data format."
    args_schema: Type[BaseModel] = GetSampleRowsInput

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def _run(self, inputs: GetSampleRowsInput):
        return str(self.df.head(inputs.num_rows).to_markdown())

    def _arun(self, inputs: GetSampleRowsInput):
        raise NotImplementedError("Async not supported")


# Tool: Describe a column
class DescribeColumnInput(BaseModel):
    column_name: str


class DescribeColumnTool(BaseTool):
    name: str = "describe_column"
    description: str = "Returns descriptive statistics of a specified column."
    args_schema: Type[BaseModel] = DescribeColumnInput

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def _run(self, inputs: DescribeColumnInput):
        col = inputs.column_name
        if col not in self.df.columns:
            return f"Column '{col}' not found."
        return str(self.df[col].describe(include='all'))

    def _arun(self, inputs: DescribeColumnInput):
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
        # Disallow dangerous keywords
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
        ]
        return not any(re.search(pattern, query) for pattern in banned_patterns)

    def _run(self, query: str) -> str:
        if not self._is_safe_expression(query):
            return "❌ Unsafe query detected. Only simple pandas expressions are allowed."

        try:
            local_vars: dict = {"df": self._df}
            result = eval(query, {}, local_vars)
            return str(result)
        except Exception as e:
            return f"❌ Error running query: {e}"

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

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def _run(self, inputs: GroupByAggregateInput):
        try:
            group_col = inputs.group_column
            metric_col = inputs.metric_column
            agg_func = inputs.aggregation.lower()

            if group_col not in self.df.columns:
                return f"❌ Group column '{group_col}' not found."
            if metric_col not in self.df.columns:
                return f"❌ Metric column '{metric_col}' not found."

            valid_aggs = ['mean', 'sum', 'count', 'max', 'min', 'std', 'median']
            if agg_func not in valid_aggs:
                return f"❌ Invalid aggregation. Use one of: {valid_aggs}"

            grouped = self.df.groupby(group_col)[metric_col]
            
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

            return f"📊 {agg_func.title()} of {metric_col} by {group_col}:\n{result.to_string()}"

        except Exception as e:
            return f"❌ Error in groupby aggregation: {e}"

    def _arun(self, inputs: GroupByAggregateInput):
        raise NotImplementedError("Async not supported")


# Tool: Get top categories
class TopCategoriesInput(BaseModel):
    column: str = Field(description="Column to analyze")
    n: int = Field(default=10, description="Number of top categories to return")


class TopCategoriesTool(BaseTool):
    name: str = "top_categories"
    description: str = "Get the top N most frequent values in a categorical column."
    args_schema: Type[BaseModel] = TopCategoriesInput

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def _run(self, inputs: TopCategoriesInput):
        try:
            column = inputs.column
            n = inputs.n

            if column not in self.df.columns:
                return f"❌ Column '{column}' not found."

            value_counts = self.df[column].value_counts().head(n)
            total_count = len(self.df)
            
            result_lines = [f"📊 Top {n} categories in '{column}':"]
            for value, count in value_counts.items():
                percentage = (count / total_count) * 100
                result_lines.append(f"  {value}: {count} ({percentage:.1f}%)")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error analyzing categories: {e}"

    def _arun(self, inputs: TopCategoriesInput):
        raise NotImplementedError("Async not supported")


# Tool: Generate histogram data
class HistogramInput(BaseModel):
    column: str = Field(description="Numeric column to create histogram for")
    bins: int = Field(default=10, description="Number of bins for the histogram")


class HistogramTool(BaseTool):
    name: str = "histogram"
    description: str = "Generate histogram data for a numeric column to understand distribution."
    args_schema: Type[BaseModel] = HistogramInput

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def _run(self, inputs: HistogramInput):
        try:
            column = inputs.column
            bins = inputs.bins

            if column not in self.df.columns:
                return f"❌ Column '{column}' not found."

            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                return f"❌ Column '{column}' is not numeric. Use top_categories for categorical data."

            series = self.df[column].dropna()
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

    def _arun(self, inputs: HistogramInput):
        raise NotImplementedError("Async not supported")


# Tool: Correlation matrix
class CorrelationMatrixInput(BaseModel):
    columns: list[str] = Field(default=None, description="Specific columns to include (optional)")
    method: str = Field(default="pearson", description="Correlation method: pearson, spearman, or kendall")


class CorrelationMatrixTool(BaseTool):
    name: str = "correlation_matrix"
    description: str = "Calculate correlation matrix for numeric columns."
    args_schema: Type[BaseModel] = CorrelationMatrixInput

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def _run(self, inputs: CorrelationMatrixInput):
        try:
            method = inputs.method.lower()
            if method not in ['pearson', 'spearman', 'kendall']:
                return f"❌ Invalid method. Use: pearson, spearman, or kendall"

            # Select numeric columns
            if inputs.columns:
                # Use specified columns
                numeric_cols = [col for col in inputs.columns if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]
                if not numeric_cols:
                    return f"❌ No valid numeric columns found in specified list."
                df_numeric = self.df[numeric_cols]
            else:
                # Use all numeric columns
                df_numeric = self.df.select_dtypes(include=['number'])
                
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

    def _arun(self, inputs: CorrelationMatrixInput):
        raise NotImplementedError("Async not supported")
