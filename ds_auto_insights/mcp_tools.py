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
