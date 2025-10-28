# stats_compass/tools/exploration_tools.py
"""
Data exploration tools for DS Auto Insights.
Provides comprehensive data exploration and analysis capabilities.
"""

import re
import ast
import psutil
import os
from typing import Type, Optional, List
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool

# Import centralized configuration constants
from stats_compass.constants import (
    MAX_QUERY_LINES,
    MAX_VARIABLE_SIZE_MB,
    MAX_MEMORY_USAGE_MB,
    MEMORY_WARNING_THRESHOLD_MB,
    MAX_DISPLAY_COLUMNS,
    MAX_DISPLAY_COLWIDTH,
    MAX_PREVIEW_ROWS,
    MAX_PREVIEW_COLUMNS,
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
    description: str = """Run safe pandas QUERIES and COLUMN MODIFICATIONS on dataframe `df`. 
    
USE THIS TOOL FOR:
✅ Querying data: df['col'].unique(), df.columns, df.groupby('col').mean()
✅ Modifying columns IN-PLACE: df['col'] = df['col'].replace('old', 'new')
✅ Simple calculations: result = df['price'].mean()

DO NOT USE FOR:
❌ Filtering/subsetting dataframe: df = df[df['col'] > 5]  → Use other tools instead
❌ Creating new columns: df['new_col'] = ...  → Use create_column tool instead
❌ Complex transformations  → Use specialized tools (create_column, apply_transformation, etc.)

SECURITY - These operations are BLOCKED:
🚫 df = new_df (replacing entire dataframe)
🚫 imports, exec, eval, file operations

IMPORTANT - Type Matching in Filters:
- ALWAYS match data types when using .isin() or comparisons
- For integer columns, use integers: df[df['col'].isin([0, 1])]  ✓
- For string columns, use strings: df[df['col'].isin(['0', '1'])]  ✓
- AVOID mixing types: df[df['col'].isin(['0', 1])]  ✗ (unpredictable!)
- Check dtype first: df['col'].dtype or df['col'].unique() to see actual values"""
    args_schema: Type[BaseModel] = RunPandasQueryToolInput

    _df: pd.DataFrame = PrivateAttr()
    _user_vars: dict = PrivateAttr(default_factory=dict)

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        self._user_vars = {}

    def _is_safe_expression(self, query: str) -> bool:
        """Enhanced validation that allows safe assignments while blocking dangerous patterns."""
        # Check for multi-line queries 
        lines = [line.strip() for line in query.strip().split('\n') if line.strip()]
        
        if len(lines) > MAX_QUERY_LINES:
            return False
        
        # Critical security patterns (always blocked)
        critical_patterns = [
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
        
        # Dangerous assignment patterns (protect core objects)
        dangerous_assignments = [
            r"^\s*df\s*=(?!=)",         # Don't overwrite main dataframe (but allow df['col'] = ...)
            r"^\s*pd\s*=(?!=)",         # Don't overwrite pandas
            r"^\s*np\s*=(?!=)",         # Don't overwrite numpy
            r".*=.*\.random\.",         # Block random data generation
            r".*=.*range\(\s*\d{6,}",   # Block large ranges (100k+ items)
            r".*=.*zeros\(\s*\d{6,}",   # Block large arrays
            r".*=.*ones\(\s*\d{6,}",    # Block large arrays
        ]
        
        full_query = ' '.join(lines)
        
        # Check critical patterns
        if any(re.search(pattern, full_query, re.IGNORECASE) for pattern in critical_patterns):
            return False
            
        # Check dangerous assignments
        if any(re.search(pattern, full_query, re.IGNORECASE) for pattern in dangerous_assignments):
            return False
            
        return True
    
    def _is_assignment_ast(self, query: str) -> bool:
        """
        Use AST parsing to reliably detect assignment statements.
        
        This correctly handles:
        - filtered_df = df[df['col'] <= 3]  (assignment with comparison inside)
        - df.loc[df['x'] == 5, 'y'] = 10    (assignment with comparison)
        - lambda x: x if x else 0           (ternary, not assignment)
        - df['col'] = df['old'] * 2         (column assignment)
        
        Returns:
            True if query contains an assignment, False otherwise
        """
        try:
            tree = ast.parse(query)
            # Check if any node is an assignment
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    return True
            return False
        except SyntaxError:
            # If we can't parse it, assume it's an expression
            # (will likely fail at exec/eval stage with better error)
            return False

    def _get_security_error_message(self, query: str) -> str:
        """Return appropriate error message based on detected security issue."""
        if any(pattern in query.lower() for pattern in ['df =', 'pd =', 'np =']):
            # Check if this is trying to filter/subset the dataframe
            if 'df[' in query and '=' in query and query.strip().startswith('df ='):
                return """❌ Cannot replace entire dataframe (security protection).

💡 To create a filtered/subset dataframe, use one of these approaches:

1. For simple row filtering:
   - Use: filtered = df[df['col'] > 5]  (assigns to new variable)
   - Then work with 'filtered' variable

2. For creating new columns:
   - Use: create_column(column_name='new_col', operation="df['old_col'] * 2")

3. For replacing values in existing columns:
   - Use: df['col'] = df['col'].replace('old', 'new')  (modifies column, not df)

❌ Don't use: df = df[condition]  (blocked for security)"""
            else:
                return "❌ Cannot overwrite core objects (df, pd, np). These are protected for system stability."
        elif 'random' in query.lower() and '=' in query:
            return "❌ Random data generation is not allowed to prevent memory issues."
        elif re.search(r'range\(\s*\d{6,}', query) or re.search(r'zeros\(\s*\d{6,}', query):
            return "❌ Large data structure creation is blocked to prevent memory issues."
        else:
            return "❌ Unsafe query detected. Check for imports, file operations, or dangerous functions."
    
    def _build_safe_namespace(self) -> dict:
        """
        Build sandboxed namespace for query execution with defense-in-depth protection.
        
        Returns dict with user variables + protected core objects (df, pd, np, builtins).
        Core objects are force-overridden to prevent shadowing.
        Uses StateManager for consistent state access.
        
        Security Model:
            - Layer 1: StateManager filters protected keys
            - Layer 2: Double-check filtering here (defense-in-depth)
            - Layer 3: Explicit .update() to force-override core objects
        """
        # Get user variables from StateManager (already filtered)
        user_vars = DataFrameStateManager.get_all_user_vars()
        
        # DEFENSE IN DEPTH: Double-check protected keys aren't present
        # (in case StateManager has a bug or state gets corrupted)
        protected_keys = {'df', 'pd', 'np', 'str', 'int', 'float', 'bool', 
                         'list', 'dict', 'len', 'min', 'max', 'sum', 'abs', 'round'}
        safe_user_vars = {k: v for k, v in user_vars.items() if k not in protected_keys}
        
        # Get current dataframe from StateManager
        current_df = DataFrameStateManager.get_active_df() or self._df
        
        # Build namespace starting with user vars
        safe_vars = {**safe_user_vars}
        
        # FORCE-SET core objects with explicit .update() (these CANNOT be shadowed)
        # Using .update() makes the override explicit - even if safe_user_vars 
        # somehow contains 'df', this will overwrite it
        safe_vars.update({
            "df": current_df, 
            "pd": pd, 
            "np": np,
            # Add basic Python types for common operations
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
        })
        
        return safe_vars
    
    def _execute_query(self, query: str, safe_vars: dict) -> tuple:
        """
        Execute the query safely using exec or eval.
        
        Returns:
            Tuple of (result, is_assignment) where result is the query output
        """
        is_assignment = self._is_assignment_ast(query)
        
        if is_assignment:
            # This is an assignment statement, use exec
            exec(query, {"__builtins__": {}}, safe_vars)
            result = "✅ Assignment completed successfully"
        else:
            # This is an expression, use eval
            result = eval(query, {"__builtins__": {}}, safe_vars)
        
        return result, is_assignment
    
    def _update_stored_state(self, safe_vars: dict):
        """
        Update internal state and session state with variables from executed query.
        
        Handles:
        - DataFrame updates (both in-place modifications and replacements)
        - User variable storage (with size limits)
        - Session state synchronization
        
        Uses StateManager for centralized state management.
        """
        # Get current dataframe from StateManager
        current_df = DataFrameStateManager.get_active_df() or self._df
        
        # Always update the dataframe if it exists in safe_vars
        # This handles both in-place modifications (df['col'] = value) and replacements (df = new_df)
        if 'df' in safe_vars:
            new_df = safe_vars['df']
            self._df = new_df
            DataFrameStateManager.set_active_df(new_df)
        
        # Update user variables (filter out protected keys)
        protected_keys = {'df', 'pd', 'np', 'str', 'int', 'float', 'bool', 'list', 
                         'dict', 'len', 'min', 'max', 'sum', 'abs', 'round'}
        
        for key, value in safe_vars.items():
            if key not in protected_keys:
                # Only store reasonable-sized objects
                try:
                    if hasattr(value, '__sizeof__'):
                        size_mb = value.__sizeof__() / 1024 / 1024
                        if size_mb < MAX_VARIABLE_SIZE_MB:
                            # Store via StateManager for consistency
                            DataFrameStateManager.set_user_var(key, value)
                            # Also update local cache for backward compatibility
                            self._user_vars[key] = value
                except:
                    # If we can't check size, store it anyway (probably safe)
                    DataFrameStateManager.set_user_var(key, value)
                    self._user_vars[key] = value
    
    def _format_result(self, result: any, memory_used: float) -> str:
        """
        Format the query result for display.
        
        Includes memory warnings and stored variable info.
        Uses StateManager to access current variables.
        """
        # Check for excessive memory usage
        if memory_used > MEMORY_WARNING_THRESHOLD_MB:
            # Clear via StateManager
            DataFrameStateManager.clear_user_vars()
            self._user_vars.clear()
            return f"⚠️ Query used {memory_used:.1f}MB memory. User variables cleared for safety.\nResult: {str(result)}"
        
        # Show helpful info about stored variables
        var_info = ""
        user_vars = DataFrameStateManager.get_all_user_vars()
        if user_vars:
            var_names = list(user_vars.keys())
            var_info = f"\n💾 Stored variables: {var_names}"
        
        return f"{str(result)}{var_info}"
    
    def _handle_execution_error(self, error: Exception) -> str:
        """
        Provide user-friendly error messages based on exception type.
        """
        error_msg = str(error)
        
        if "invalid syntax" in error_msg.lower():
            return f"❌ Syntax error: {error_msg}"
        elif "name" in error_msg.lower() and "not defined" in error_msg.lower():
            available_vars = list(self._user_vars.keys()) + ['df', 'pd', 'np']
            return f"❌ Variable not found: {error_msg}\nAvailable variables: {available_vars}"
        else:
            return f"❌ Error running query: {error_msg}"

    def _run(self, query: str) -> str:
        """
        Execute a pandas query safely with security validation and state management.
        
        This method orchestrates the entire query execution pipeline:
        1. Security validation
        2. Memory monitoring
        3. Namespace setup
        4. Query execution
        5. State updates
        6. Result formatting
        """
        # Step 1: Security validation
        if not self._is_safe_expression(query):
            return self._get_security_error_message(query)

        # Step 2: Start memory monitoring
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Set pandas display options
            with pd.option_context('display.max_columns', MAX_DISPLAY_COLUMNS, 
                                   'display.width', None, 
                                   'display.max_colwidth', MAX_DISPLAY_COLWIDTH):
                
                # Step 3: Build safe execution namespace
                safe_vars = self._build_safe_namespace()
                
                
                # Step 4: Execute the query
                result, is_assignment = self._execute_query(query, safe_vars)
                
                # Step 5: Update stored state (dataframe + user variables)
                self._update_stored_state(safe_vars)
                
                # Step 6: Check memory usage and format result
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                return self._format_result(result, memory_used)
                
        except Exception as e:
            return self._handle_execution_error(e)

    def _arun(self, query: str):
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
            # Safety limits
            if num_rows > MAX_PREVIEW_ROWS:
                num_rows = MAX_PREVIEW_ROWS
                
            num_cols = len(self._df.columns)
            if num_cols > MAX_PREVIEW_COLUMNS:
                return f"❌ Dataset has too many columns ({num_cols}). Maximum supported: {MAX_PREVIEW_COLUMNS} columns for preview."
            
            # Set pandas display options to show columns (safe limits)
            max_cols = min(MAX_PREVIEW_COLUMNS, num_cols)
            with pd.option_context('display.max_columns', max_cols, 
                                   'display.width', None, 
                                   'display.max_colwidth', MAX_DISPLAY_COLWIDTH):
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
