# stats_compass/tools/query_tools.py
"""
Query and inspection tools for Stats Compass.

This module contains tools for querying, inspecting, and manipulating data:
- InspectDataTool: Read-only queries and inspection
- ModifyColumnTool: In-place column modifications
- FilterDataframeTool: Create filtered subsets of data
- Plus basic inspection tools (schema, samples, describe, preview)
"""

import re
import ast
from typing import Type, Optional
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool

# Import DataFrameStateManager from exploration_tools
from stats_compass.tools.exploration_tools import DataFrameStateManager

# Import centralized configuration constants
from stats_compass.constants import (
    MAX_PREVIEW_ROWS,
    MAX_PREVIEW_COLUMNS,
)


# ============================================
# Helper Functions
# ============================================


def _get_current_dataframe(state_manager_df, fallback_df):
    """
    Safely get current dataframe without triggering boolean evaluation.
    
    Avoids the "truth value of DataFrame is ambiguous" error by using
    explicit None check instead of 'or' operator.
    """
    df_from_state = state_manager_df
    if df_from_state is not None:
        return df_from_state
    return fallback_df


# ============================================
# Core Query Tools (Replaces RunPandasQueryTool)
# ============================================


class InspectDataInput(BaseModel):
    expression: str = Field(
        description="Read-only pandas expression to evaluate. Can return single values, Series, or DataFrames."
    )


class InspectDataTool(BaseTool):
    """
    Evaluate read-only pandas expressions for data inspection and calculation.
    
    ✅ USE FOR:
    - Query syntax: df.query("price > 100"), df.query("category == 'A'")
    - Getting statistics: df['col'].mean(), df['col'].describe()
    - Checking values: df['col'].unique(), df['col'].value_counts()
    - Aggregations: df.groupby('cat')['val'].sum()
    - Date ranges: df['date'].min(), df['date'].max()
    - Counting: len(df), df.query("price > 100").shape[0]
    - Multiple values: (df['date'].min(), df['date'].max())
    - Data types: df.dtypes, df['col'].dtype
    
    💡 TIP: Use df.query() for simple conditions - it's cleaner and less error-prone than boolean indexing.
    
    ❌ DOES NOT:
    - Modify data (use modify_column or create_column instead)
    - Create new dataframes (use filter_dataframe instead)
    - Store variables (read-only evaluation)
    
    Security: No assignments, no imports, no file operations.
    """
    
    name: str = "inspect_data"
    description: str = """Evaluate read-only pandas expressions for data inspection and calculation.

USE FOR (prefer .query() for simple conditions):
- Query syntax: df.query("price > 100"), df.query("category == 'A'")
- Statistics: df['col'].mean(), df['col'].describe()
- Unique values: df['col'].unique(), df['col'].value_counts()
- Aggregations: df.groupby('cat')['val'].sum()
- Date ranges: df['date'].min(), df['date'].max()
- Counting: len(df), df.query("price > 100").shape[0]
- Multiple values: (df['date'].min(), df['date'].max())
- Data types: df.dtypes, df['col'].dtype

TIP: df.query() is cleaner than df[df['col'] > value] for simple filters.
DOES NOT modify data or create variables. Read-only only."""
    
    args_schema: Type[BaseModel] = InspectDataInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _is_safe_read_only(self, expression: str) -> tuple:
        """
        Validate that expression is read-only (no assignments, imports, etc.)
        
        Returns:
            (is_safe: bool, error_message: str)
        """
        # Check for assignments
        try:
            tree = ast.parse(expression)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    return False, "❌ Assignments not allowed. Use modify_column to change data."
        except SyntaxError as e:
            return False, f"❌ Syntax error: {e}"
        
        # Block dangerous operations
        dangerous_patterns = [
            (r"\bimport\b", "imports"),
            (r"\bexec\b", "exec()"),
            (r"\beval\b", "eval()"),
            (r"\bopen\b", "file operations"),
            (r"\b__.*?__\b", "dunder methods"),
            (r"\.to_csv", "file writing"),
            (r"\.to_excel", "file writing"),
            (r"\.to_sql", "database operations"),
            (r"\bdel\b", "del statement"),
        ]
        
        for pattern, name in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False, f"❌ {name} not allowed in read-only expressions."
        
        return True, ""
    
    def _run(self, expression: str) -> str:
        # Validate expression
        is_safe, error_msg = self._is_safe_read_only(expression)
        if not is_safe:
            return error_msg
        
        try:
            # Get current dataframe from StateManager (avoid boolean evaluation)
            df_from_state = DataFrameStateManager.get_active_df()
            if df_from_state is not None:
                df = df_from_state
            else:
                df = self._df
            
            # Create safe namespace (read-only)
            namespace = {
                "df": df,
                "pd": pd,
                "np": np,
                # Useful builtins
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
            }
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, namespace)
            
            # Format result safely (avoid DataFrame boolean ambiguity bug)
            if result is None:
                return "✅ Expression evaluated (no return value)"
            
            # Handle different result types
            if isinstance(result, pd.DataFrame):
                # For DataFrames, show summary info
                if len(result) <= 20:
                    result_str = result.to_string()
                else:
                    # Truncate large results
                    result_str = f"DataFrame with {len(result)} rows × {len(result.columns)} columns\n"
                    result_str += result.head(10).to_string()
                    result_str += f"\n... ({len(result) - 10} more rows)"
            elif isinstance(result, pd.Series):
                # For Series, show up to 10 values
                if len(result) <= 10:
                    result_str = result.to_string()
                else:
                    result_str = f"Series with {len(result)} values\n"
                    result_str += result.head(10).to_string()
                    result_str += f"\n... ({len(result) - 10} more values)"
            elif isinstance(result, tuple):
                # Format tuples nicely (common for multiple return values)
                # Convert each item to string safely to avoid DataFrame ambiguity
                formatted_items = []
                for i, item in enumerate(result):
                    if isinstance(item, (pd.DataFrame, pd.Series)):
                        item_str = item.to_string()
                    else:
                        item_str = str(item)
                    formatted_items.append(f"  {i+1}. {item_str}")
                result_str = "\n".join(formatted_items)
            else:
                # Single values, lists, etc.
                result_str = str(result)
            
            return f"📊 Result:\n{result_str}"
            
        except Exception as e:
            error_type = type(e).__name__
            return f"❌ {error_type}: {str(e)}"
    
    def _arun(self, expression: str):
        raise NotImplementedError("Async not supported")


class ModifyColumnInput(BaseModel):
    column: str = Field(description="Name of column to modify")
    operation: str = Field(description="Pandas operation that produces new values for the column")


class ModifyColumnTool(BaseTool):
    """
    Modify an existing column in-place with pandas operations.
    
    ✅ USE FOR:
    - Type conversion: pd.to_datetime(df['date'])
    - String operations: df['name'].str.lower()
    - Math operations: df['price'] * 1.1
    - Replacing values: df['col'].replace('old', 'new')
    - Filling missing: df['col'].fillna(0)
    - Stripping whitespace: df['col'].str.strip()
    
    ❌ USE create_column FOR:
    - Creating NEW columns (not modifying existing)
    
    The operation should return a Series that will replace the column.
    
    Example:
        modify_column(
            column='price',
            operation="df['price'] * 1.1"
        )
        # Increases all prices by 10%
    """
    
    name: str = "modify_column"
    description: str = """Modify an existing column in-place with pandas operations.

USE FOR:
- Type conversion: pd.to_datetime(df['date'])
- String operations: df['name'].str.lower()
- Math operations: df['price'] * 1.1
- Replacing values: df['col'].replace('old', 'new')
- Filling missing: df['col'].fillna(0)

The operation should return a Series. Use create_column for NEW columns."""
    
    args_schema: Type[BaseModel] = ModifyColumnInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _run(self, column: str, operation: str) -> str:
        # Get current dataframe from StateManager (avoid boolean evaluation)
        df = _get_current_dataframe(DataFrameStateManager.get_active_df(), self._df)
        
        # Validate column exists
        if column not in df.columns:
            return f"❌ Column '{column}' not found. Available: {list(df.columns)}"
        
        # Security check (same dangerous patterns as inspect_data)
        dangerous_patterns = [
            r"\bimport\b", r"\bexec\b", r"\beval\b", 
            r"\bopen\b", r"\.to_csv", r"\.to_excel",
            r"\b__.*?__\b", r"\bdel\b"
        ]
        if any(re.search(p, operation, re.IGNORECASE) for p in dangerous_patterns):
            return "❌ Unsafe operation detected"
        
        try:
            # Create namespace
            namespace = {
                "df": df,
                "pd": pd,
                "np": np,
            }
            
            # Evaluate operation
            new_values = eval(operation, {"__builtins__": {}}, namespace)
            
            # Validate result is Series-like with correct length
            if not hasattr(new_values, '__len__'):
                return f"❌ Operation must return a Series or array, got {type(new_values).__name__}"
            
            if len(new_values) != len(df):
                return f"❌ Operation returned {len(new_values)} values, but dataframe has {len(df)} rows"
            
            # Store old dtype for comparison
            old_dtype = df[column].dtype
            
            # Modify column IN-PLACE
            df[column] = new_values
            
            # Update StateManager
            DataFrameStateManager.set_active_df(df)
            self._df = df
            
            # Format result
            new_dtype = df[column].dtype
            result = f"✅ Modified column '{column}'\n"
            result += f"\nType change: {old_dtype} → {new_dtype}\n"
            result += f"\nPreview (first 10 values):\n"
            result += df[column].head(10).to_string()
            
            if len(df) > 10:
                result += f"\n... ({len(df) - 10} more values)"
            
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            return f"❌ Error modifying column ({error_type}): {str(e)}"
    
    def _arun(self, column: str, operation: str):
        raise NotImplementedError("Async not supported")


class FilterDataframeInput(BaseModel):
    condition: str = Field(description="Boolean condition to filter rows")
    result_name: str = Field(description="Name for the filtered dataframe variable")


class FilterDataframeTool(BaseTool):
    """
    Create a filtered subset of the dataframe and store it as a variable.
    
    ✅ USE FOR:
    - Query syntax: df.query("price > 100")
    - Date filtering: df.query("date > '2020-01-01'")
    - Multiple conditions: df.query("price > 100 and category == 'A'")
    - Complex filters (use boolean indexing): df['status'].isin(['active', 'pending'])
    - String filtering: df['name'].str.contains('test')
    
    💡 TIP: Use df.query() for simple numeric/string comparisons. Use boolean indexing 
    (df[...]) for complex operations like .isin(), .str.contains(), or .isna().
    
    The condition should return a boolean Series. The filtered dataframe
    will be stored with the given result_name and can be used in subsequent
    operations.
    
    Example:
        filter_dataframe(
            condition="df.query('price > 100')",
            result_name="expensive_items"
        )
        # Creates 'expensive_items' dataframe with rows where price > 100
    """
    
    name: str = "filter_dataframe"
    description: str = """Create a filtered subset of the dataframe and store it as a variable.

USE FOR (prefer .query() for simple conditions):
- Query syntax: df.query("price > 100")
- Date filtering: df.query("date > '2020-01-01'")
- Multiple conditions: df.query("price > 100 and category == 'A'")
- Complex filters: df['status'].isin(['active', 'pending'])
- String filtering: df['name'].str.contains('test')

TIP: df.query() is cleaner for simple comparisons. Use df[...] for .isin(), .str methods.
The condition returns a boolean Series. Filtered dataframe is stored with result_name."""
    
    args_schema: Type[BaseModel] = FilterDataframeInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _run(self, condition: str, result_name: str) -> str:
        # Validate result_name
        if not result_name.isidentifier():
            return f"❌ Invalid variable name '{result_name}'. Use letters, numbers, underscores only (no spaces or special characters)."
        
        # Security check
        dangerous_patterns = [
            r"\bimport\b", r"\bexec\b", r"\beval\b", 
            r"\bopen\b", r"\.to_csv", r"\.to_excel",
            r"\b__.*?__\b", r"\bdel\b"
        ]
        if any(re.search(p, condition, re.IGNORECASE) for p in dangerous_patterns):
            return "❌ Unsafe condition detected"
        
        try:
            # Get active dataframe (avoid boolean evaluation)
            df = _get_current_dataframe(DataFrameStateManager.get_active_df(), self._df)
            
            # Create namespace
            namespace = {
                "df": df,
                "pd": pd,
                "np": np,
            }
            
            # Evaluate condition
            mask = eval(condition, {"__builtins__": {}}, namespace)
            
            # Validate mask is boolean-like
            if not isinstance(mask, (pd.Series, np.ndarray)):
                return f"❌ Condition must return a boolean Series, got {type(mask).__name__}"
            
            if len(mask) != len(df):
                return f"❌ Condition returned {len(mask)} values, but dataframe has {len(df)} rows"
            
            # Apply filter
            filtered_df = df[mask].copy()
            
            # Store in StateManager
            DataFrameStateManager.set_user_var(result_name, filtered_df)
            
            # Format result
            n_rows_kept = len(filtered_df)
            n_rows_removed = len(df) - n_rows_kept
            pct_kept = (n_rows_kept / len(df) * 100) if len(df) > 0 else 0
            
            result = f"✅ Created filtered dataframe '{result_name}'\n"
            result += f"\n📊 Filter Results:\n"
            result += f"   Original rows: {len(df):,}\n"
            result += f"   Kept: {n_rows_kept:,} ({pct_kept:.1f}%)\n"
            result += f"   Removed: {n_rows_removed:,}\n"
            result += f"\n💾 Variable '{result_name}' is now available for use\n"
            
            if n_rows_kept > 0:
                result += f"\nPreview (first 5 rows):\n"
                result += filtered_df.head(5).to_string()
            else:
                result += f"\n⚠️ No rows matched the filter condition"
            
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            return f"❌ Error filtering dataframe ({error_type}): {str(e)}"
    
    def _arun(self, condition: str, result_name: str):
        raise NotImplementedError("Async not supported")


# ============================================
# Basic Inspection Tools
# ============================================


class GetSchemaInput(BaseModel):
    pass


class GetSchemaTool(BaseTool):
    name: str = "get_schema"
    description: str = "Returns column names and their data types."
    args_schema: Type[BaseModel] = GetSchemaInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self) -> str:
        # Get current dataframe from StateManager (avoid boolean evaluation)
        df = _get_current_dataframe(DataFrameStateManager.get_active_df(), self._df)
        return str(df.dtypes)

    def _arun(self):
        raise NotImplementedError("Async not supported")


class GetSampleRowsInput(BaseModel):
    num_rows: int = Field(description="Number of rows to sample")


class GetSampleRowsTool(BaseTool):
    name: str = "get_sample_rows"
    description: str = "Returns a few rows from the dataframe to understand the data format."
    args_schema: Type[BaseModel] = GetSampleRowsInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, num_rows: int) -> str:
        # Get current dataframe from StateManager (avoid boolean evaluation)
        df = _get_current_dataframe(DataFrameStateManager.get_active_df(), self._df)
        return str(df.head(num_rows).to_markdown())

    def _arun(self, num_rows: int):
        raise NotImplementedError("Async not supported")


class DescribeColumnInput(BaseModel):
    column_name: str = Field(description="Name of the column to describe")


class DescribeColumnTool(BaseTool):
    name: str = "describe_column"
    description: str = "Returns descriptive statistics of a specified column."
    args_schema: Type[BaseModel] = DescribeColumnInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column_name: str) -> str:
        # Get current dataframe from StateManager (avoid boolean evaluation)
        df = _get_current_dataframe(DataFrameStateManager.get_active_df(), self._df)
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found."
        return str(df[column_name].describe(include='all'))

    def _arun(self, column_name: str):
        raise NotImplementedError("Async not supported")


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
            # Get current dataframe from StateManager (avoid boolean evaluation)
            df = _get_current_dataframe(DataFrameStateManager.get_active_df(), self._df)
            
            # Safety limits
            num_rows = min(num_rows, MAX_PREVIEW_ROWS)
            num_cols = len(df.columns)
            
            if num_cols > MAX_PREVIEW_COLUMNS:
                return f"⚠️ Dataset has {num_cols} columns (max {MAX_PREVIEW_COLUMNS} for preview). Use get_schema to see all column names."
            
            # Build comprehensive preview
            preview = f"📊 Dataset Preview ({len(df)} total rows, {num_cols} columns):\n"
            preview += f"Columns: {list(df.columns)}\n\n"
            preview += f"First {num_rows} rows:\n"
            preview += df.head(num_rows).to_string()
            
            return preview
            
        except Exception as e:
            return f"❌ Error generating preview: {str(e)}"

    def _arun(self, num_rows: int = 5):
        raise NotImplementedError("Async not supported")
