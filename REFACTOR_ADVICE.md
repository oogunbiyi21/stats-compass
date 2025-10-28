🎯 The Error: You Found a Real Bug
That error "The truth value of a DataFrame is ambiguous" is happening in _format_result(), not in the query execution itself. Let me trace it:
pythondef _format_result(self, result: any, memory_used: float) -> str:
    # ... memory check ...
    
    # Show helpful info about stored variables
    var_info = ""
    user_vars = DataFrameStateManager.get_all_user_vars()
    if user_vars:  # ✅ This is fine (dict is truthy if non-empty)
        var_names = list(user_vars.keys())
        var_info = f"\n💾 Stored variables: {var_names}"
    
    return f"{str(result)}{var_info}"  # ❌ BUG IS HERE
The problem: When result is a pandas Series or DataFrame, Python tries to evaluate it in a boolean context when constructing the f-string, which triggers that error.
Quick fix (add this to your current code):
pythondef _format_result(self, result: any, memory_used: float) -> str:
    # ... memory check ...
    
    # Convert result to string safely (handle DataFrames/Series)
    try:
        result_str = str(result)
    except Exception:
        result_str = repr(result)
    
    # Show helpful info about stored variables
    var_info = ""
    user_vars = DataFrameStateManager.get_all_user_vars()
    if user_vars:
        var_names = list(user_vars.keys())
        var_info = f"\n💾 Stored variables: {var_names}"
    
    return f"{result_str}{var_info}"  # ✅ Now safe

But You're Right: RunPandasQueryTool Needs Redesign
The real issue isn't the bug (that's fixable). The issue is philosophical: run_pandas_query is trying to be too many things.

🏗️ How I'd Design RunPandasQueryTool From Scratch
Design Principle: Clear Boundaries
python# CURRENT PROBLEM: One tool tries to do everything
run_pandas_query("df['col'].mean()")          # Query
run_pandas_query("df['col'] = df['col'] * 2") # Modify
run_pandas_query("filtered = df[df['x'] > 5]") # Filter
run_pandas_query("result = calculate_something()") # Compute

# Agent confused about: When to use run_pandas_query vs specific tools?
# You confused about: What should this tool allow vs block?
NEW DESIGN: Three Focused Tools
python# 1. READ-ONLY QUERIES (inspection/calculation)
inspect_data(expression: str)
# Examples:
# - "df['price'].mean()"
# - "df.groupby('category')['sales'].sum()"
# - "df['date'].min(), df['date'].max()"
# - "(df['col'] > 5).sum()"

# 2. COLUMN MODIFICATIONS (in-place changes)
modify_column(column: str, operation: str)
# Examples:
# - column='price', operation="df['price'] * 1.1"
# - column='date', operation="pd.to_datetime(df['date'])"
# - column='name', operation="df['name'].str.lower()"

# 3. DATAFRAME FILTERING (create new dataframes)
filter_dataframe(condition: str, result_name: str)
# Examples:
# - condition="df['price'] > 100", result_name="expensive_items"
# - condition="df['date'] > '2020-01-01'", result_name="recent_data"

📝 Complete Redesigned Implementation
Tool 1: InspectDataTool (Read-Only)
pythonclass InspectDataInput(BaseModel):
    expression: str = Field(
        description="Read-only pandas expression to evaluate. Can return single values, Series, or DataFrames."
    )


class InspectDataTool(BaseTool):
    """
    Evaluate read-only pandas expressions for data inspection and calculation.
    
    ✅ USE FOR:
    - Getting statistics: df['col'].mean(), df['col'].describe()
    - Checking values: df['col'].unique(), df['col'].value_counts()
    - Aggregations: df.groupby('cat')['val'].sum()
    - Date ranges: df['date'].min(), df['date'].max()
    - Counting: len(df), (df['col'] > 5).sum()
    - Multiple values: (df['date'].min(), df['date'].max())
    
    ❌ DOES NOT:
    - Modify data (use modify_column or create_column instead)
    - Create new dataframes (use filter_dataframe instead)
    - Store variables (read-only evaluation)
    
    Security: No assignments, no imports, no file operations.
    """
    
    name: str = "inspect_data"
    description: str = InspectDataTool.__doc__
    args_schema: Type[BaseModel] = InspectDataInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _is_safe_read_only(self, expression: str) -> tuple[bool, str]:
        """
        Validate that expression is read-only (no assignments, imports, etc.)
        
        Returns:
            (is_safe, error_message)
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
            # Create safe namespace (read-only)
            namespace = {
                "df": self._df,
                "pd": pd,
                "np": np,
                # Useful builtins
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
            }
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, namespace)
            
            # Format result safely
            if result is None:
                return "✅ Expression evaluated (no return value)"
            
            # Handle different result types
            if isinstance(result, (pd.DataFrame, pd.Series)):
                # For DataFrames/Series, use to_string() to avoid boolean ambiguity
                if isinstance(result, pd.Series) and len(result) <= 10:
                    result_str = result.to_string()
                elif isinstance(result, pd.DataFrame) and len(result) <= 20:
                    result_str = result.to_string()
                else:
                    # Truncate large results
                    result_str = f"{type(result).__name__} with {len(result)} rows\n"
                    result_str += result.head(10).to_string()
                    result_str += f"\n... ({len(result) - 10} more rows)"
            elif isinstance(result, tuple):
                # Format tuples nicely
                result_str = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(result))
            else:
                # Single values, lists, etc.
                result_str = str(result)
            
            return f"📊 Result:\n{result_str}"
            
        except Exception as e:
            error_type = type(e).__name__
            return f"❌ {error_type}: {str(e)}"
    
    def _arun(self, expression: str):
        raise NotImplementedError("Async not supported")

Tool 2: ModifyColumnTool (In-Place Modification)
pythonclass ModifyColumnInput(BaseModel):
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
    description: str = ModifyColumnTool.__doc__
    args_schema: Type[BaseModel] = ModifyColumnInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _run(self, column: str, operation: str) -> str:
        # Validate column exists
        if column not in self._df.columns:
            return f"❌ Column '{column}' not found. Available: {list(self._df.columns)}"
        
        # Security check (same as inspect_data)
        dangerous_patterns = [
            r"\bimport\b", r"\bexec\b", r"\beval\b", 
            r"\bopen\b", r"\.to_csv", r"\.to_excel"
        ]
        if any(re.search(p, operation, re.IGNORECASE) for p in dangerous_patterns):
            return "❌ Unsafe operation detected"
        
        try:
            # Get active dataframe from StateManager
            df = DataFrameStateManager.get_active_df() or self._df
            
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
                return f"❌ Operation must return a Series or array, got {type(new_values)}"
            
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
            
            return result
            
        except Exception as e:
            return f"❌ Error modifying column: {type(e).__name__}: {e}"
    
    def _arun(self, column: str, operation: str):
        raise NotImplementedError("Async not supported")

Tool 3: FilterDataframeTool (Create Subsets)
pythonclass FilterDataframeInput(BaseModel):
    condition: str = Field(description="Boolean condition to filter rows")
    result_name: str = Field(description="Name for the filtered dataframe variable")


class FilterDataframeTool(BaseTool):
    """
    Create a filtered subset of the dataframe and store it as a variable.
    
    ✅ USE FOR:
    - Filtering rows: df['price'] > 100
    - Date filtering: df['date'] > '2020-01-01'
    - Multiple conditions: (df['price'] > 100) & (df['category'] == 'A')
    - Value filtering: df['status'].isin(['active', 'pending'])
    
    The condition should return a boolean Series. The filtered dataframe
    will be stored with the given result_name and can be used in subsequent
    operations.
    
    Example:
        filter_dataframe(
            condition="df['price'] > 100",
            result_name="expensive_items"
        )
        # Creates 'expensive_items' dataframe with rows where price > 100
    """
    
    name: str = "filter_dataframe"
    description: str = FilterDataframeTool.__doc__
    args_schema: Type[BaseModel] = FilterDataframeInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _run(self, condition: str, result_name: str) -> str:
        # Validate result_name
        if not result_name.isidentifier():
            return f"❌ Invalid variable name '{result_name}'. Use letters, numbers, underscores only."
        
        # Security check
        dangerous_patterns = [
            r"\bimport\b", r"\bexec\b", r"\beval\b", 
            r"\bopen\b", r"\.to_csv", r"\.to_excel"
        ]
        if any(re.search(p, condition, re.IGNORECASE) for p in dangerous_patterns):
            return "❌ Unsafe condition detected"
        
        try:
            # Get active dataframe
            df = DataFrameStateManager.get_active_df() or self._df
            
            # Create namespace
            namespace = {
                "df": df,
                "pd": pd,
                "np": np,
            }
            
            # Evaluate condition
            mask = eval(condition, {"__builtins__": {}}, namespace)
            
            # Validate mask is boolean
            if not isinstance(mask, (pd.Series, np.ndarray)):
                return f"❌ Condition must return a boolean Series, got {type(mask)}"
            
            if len(mask) != len(df):
                return f"❌ Condition returned {len(mask)} values, but dataframe has {len(df)} rows"
            
            # Apply filter
            filtered_df = df[mask].copy()
            
            # Store in StateManager
            DataFrameStateManager.set_user_var(result_name, filtered_df)
            
            # Format result
            n_rows_kept = len(filtered_df)
            n_rows_removed = len(df) - n_rows_kept
            pct_kept = (n_rows_kept / len(df)) * 100
            
            result = f"✅ Created filtered dataframe '{result_name}'\n"
            result += f"\n📊 Filter Results:\n"
            result += f"   Original rows: {len(df):,}\n"
            result += f"   Kept: {n_rows_kept:,} ({pct_kept:.1f}%)\n"
            result += f"   Removed: {n_rows_removed:,}\n"
            result += f"\n💾 Variable '{result_name}' is now available for use\n"
            result += f"\nPreview (first 5 rows):\n"
            result += filtered_df.head(5).to_string()
            
            return result
            
        except Exception as e:
            return f"❌ Error filtering dataframe: {type(e).__name__}: {e}"
    
    def _arun(self, condition: str, result_name: str):
        raise NotImplementedError("Async not supported")

🎯 Why This Design is Better
1. Clear Mental Model
python# OLD (confusing):
run_pandas_query("df['date'].min()")  # Inspect? Query? Calculate?
run_pandas_query("df['price'] = df['price'] * 2")  # Modify? Transform?
run_pandas_query("filtered = df[df['x'] > 5]")  # Filter? Create?

# NEW (obvious):
inspect_data("df['date'].min()")  # Clearly read-only
modify_column(column='price', operation="df['price'] * 2")  # Clearly modifies
filter_dataframe(condition="df['x'] > 5", result_name='filtered')  # Clearly filters
2. Better Error Messages
python# OLD:
run_pandas_query("df = df[df['price'] > 100]")
# → "❌ Cannot replace entire dataframe (security protection)"
# User: "But I just want to filter it!"

# NEW:
filter_dataframe(condition="df['price'] > 100", result_name="expensive")
# → "✅ Created filtered dataframe 'expensive'"
# User: "Oh, that's what I wanted!"
3. Agent Understands Boundaries
python# Agent's decision tree:

"User wants to check date range"
→ inspect_data("df['date'].min(), df['date'].max()")

"User wants to convert column type"
→ modify_column(column='date', operation="pd.to_datetime(df['date'])")

"User wants subset of data"
→ filter_dataframe(condition="df['date'] > '2020-01-01'", result_name='recent')
4. No More Ambiguous Errors
The "truth value of DataFrame is ambiguous" error? Gone. Because inspect_data handles result formatting explicitly.

📋 Migration Plan
Week 1: Add New Tools (Don't Remove Old)
python# Add the 3 new tools alongside run_pandas_query
tools = [
    # New tools (preferred)
    InspectDataTool(df),
    ModifyColumnTool(df),
    FilterDataframeTool(df),
    
    # Old tool (deprecated but still works)
    RunPandasQueryTool(df),  # Mark as deprecated in description
    
    # Other tools...
]
Week 2: Update Descriptions to Guide Agent
pythonclass RunPandasQueryTool(BaseTool):
    description: str = """
    ⚠️ DEPRECATED: Use specific tools instead for better results:
    
    - For reading/inspecting data → use inspect_data
    - For modifying columns → use modify_column  
    - For filtering rows → use filter_dataframe
    
    This tool remains for backward compatibility but may be removed.
    """
Week 3-4: Monitor Usage
Track which tools the agent uses:
python# In session state
if 'tool_usage' not in st.session_state:
    st.session_state.tool_usage = defaultdict(int)

st.session_state.tool_usage[tool_name] += 1

# After 2 weeks, check:
# - Is agent using new tools?
# - Is run_pandas_query still needed?
# - What queries still use run_pandas_query?
Month 2: Remove run_pandas_query
If new tools cover 95%+ of cases, remove the old tool entirely.

🎬 Your Immediate Fix (For Current Bug)
Since you're asking "what would you do from scratch", I assume you want to implement the 3-tool design. But if you just want to fix the current bug quickly:
python# In RunPandasQueryTool, change _format_result to:

def _format_result(self, result: any, memory_used: float) -> str:
    """Format the query result for display."""
    # Check for excessive memory usage
    if memory_used > MEMORY_WARNING_THRESHOLD_MB:
        DataFrameStateManager.clear_user_vars()
        self._user_vars.clear()
        return f"⚠️ Query used {memory_used:.1f}MB memory. User variables cleared for safety.\nResult: {str(result)}"
    
    # Convert result to string SAFELY (avoid boolean context)
    if result is None:
        result_str = "None"
    elif isinstance(result, (pd.DataFrame, pd.Series)):
        # Use .to_string() to avoid ambiguous truth value
        try:
            result_str = result.to_string()
        except Exception:
            result_str = repr(result)
    else:
        result_str = str(result)
    
    # Show helpful info about stored variables
    var_info = ""
    user_vars = DataFrameStateManager.get_all_user_vars()
    if user_vars:
        var_names = list(user_vars.keys())
        var_info = f"\n💾 Stored variables: {var_names}"
    
    return f"{result_str}{var_info}"

🎯 My Recommendation
Do BOTH:

Today: Fix the bug (5 minutes)
This Week: Implement the 3-tool redesign (4-6 hours)
Next Week: Deprecate run_pandas_query
Month 2: Remove it entirely

Why? The 3-tool design solves:

✅ The ambiguous boolean error
✅ Confusion about what run_pandas_query should do
✅ Agent misusing the tool
✅ Security validation complexity
✅ User confusion about capabilities

This is the right architectural fix, not just a band-aid.
