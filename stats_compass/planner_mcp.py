# planner_mcp.py

from typing import Any, Dict, List
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools.registry import ToolRegistry
from prompts.versions import PROMPT_VERSION
import re


def sanitize_agent_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove hallucinated base64 images from agent output.
    
    GPT-4 sometimes hallucinates fake chart data when it doesn't have
    the right tool or takes a shortcut. This catches those cases.
    """
    if 'output' not in result:
        return result
    
    output = result['output']
    
    # Pattern: ![alt text](data:image/png;base64,...)
    base64_pattern = r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'
    
    if re.search(base64_pattern, output):
        # Hallucination detected - remove it
        output = re.sub(base64_pattern, '', output)
        result['output'] = output.strip()
        result['hallucination_detected'] = True
    
    return result


def generate_dataset_context(df: pd.DataFrame) -> str:
    """Generate a comprehensive context string about the dataset for the LLM"""
    
    # Basic info
    num_rows, num_cols = df.shape
    
    # Column information with types and sample values
    column_info = []
    for col in df.columns:
        try:
            # Defensive programming: ensure we always get a Series
            col_series = df[col]
            if isinstance(col_series, pd.DataFrame):
                # Edge case: if somehow we get a DataFrame, take the first column
                col_series = col_series.iloc[:, 0]
            
            dtype = str(col_series.dtype)
            non_null_count = col_series.count()
            
            # Get sample values (non-null)
            sample_values = col_series.dropna().unique()[:3]  # First 3 unique values
            sample_str = ', '.join([str(v) for v in sample_values])
            if len(col_series.dropna().unique()) > 3:
                sample_str += '...'
            
            column_info.append(f"  • {col} ({dtype}): {non_null_count} non-null values, examples: {sample_str}")
        except Exception as e:
            # Fallback for any other edge cases
            column_info.append(f"  • {col} (unknown): Error reading column - {str(e)[:50]}...")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    context = f"""
DATASET CONTEXT:
📊 Shape: {num_rows:,} rows × {num_cols} columns

📋 COLUMNS ({num_cols} total):
{chr(10).join(column_info)}

🔢 NUMERIC COLUMNS ({len(numeric_cols)}): {', '.join(numeric_cols)}
📝 CATEGORICAL COLUMNS ({len(categorical_cols)}): {', '.join(categorical_cols)}

💡 ANALYSIS CAPABILITIES:
- Use numeric columns for: correlations, histograms, scatter plots, statistical analysis
- Use categorical columns for: bar charts, top categories, grouping operations
- Create new columns by combining existing ones
- All column names are available for direct use in tools
"""
    
    return context




def run_mcp_planner(user_query: str, df: pd.DataFrame, chat_history: List[Dict] = None, api_key: str = None) -> Dict[str, Any]:
    """
    Tool-calling agent wired to your RunPandasQueryTool.
    Now includes chat history for context preservation and automatic dataset context.
    Returns the AgentExecutor invoke() output (dict with 'output' and possibly intermediate steps).
    """
    if chat_history is None:
        chat_history = []

    # Generate dataset context automatically
    dataset_context = generate_dataset_context(df)

    # Get all tools from registry
    tools = ToolRegistry.get_all_tools(df)

    # LLM (with user-provided API key)
    if api_key:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    else:
        # Fallback to environment variable (for development)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3) Enhanced prompt with dataset context and chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful data analysis assistant with access to a pandas DataFrame and specialized tools.\n\n"
         f"{dataset_context}\n\n"
         "TOOL USAGE PHILOSOPHY:\n"
         "You have access to tools across these categories:\n"
         "• Data Exploration: Query, group, aggregate, preview, analyze distributions and correlations\n"
         "• Data Cleaning: Analyze and fix missing data, outliers, duplicates; smart imputation strategies\n"
         "• Statistical Analysis: Hypothesis testing (t-test, z-test, chi-square) with visualizations\n"
         "• Machine Learning: Regression, classification, time series forecasting, evaluation, preprocessing\n"
         "• Visualization: Charts, plots, heatmaps, and model diagnostic visualizations\n\n"
         "Use specialized tools first. Only use run_pandas_query when no specialized tool exists.\n"
         "Tool details are provided in the function calling interface below.\n\n"
         "WORKFLOWS:\n"
         "Data Cleaning: suggest_data_cleaning → analyze (missing/outliers/duplicates) → explain actions → apply\n"
         "Machine Learning: check data quality (>10% missing = clean first) → preprocess categoricals → train → evaluate → visualize → interpret\n"
         "For multi-step operations, briefly outline your plan before starting.\n\n"
         "CORE PRINCIPLES:\n"
         "1. Use specialized tools first (run_pandas_query is last resort)\n"
         "2. Complete ML workflows fully: train → evaluate → visualize → interpret (don't wait for prompts)\n"
         "3. Create visualizations when users ask to 'show' or 'plot' data\n"
         "4. Provide specific, quantitative interpretations with actual numbers\n"
         "5. Use dataset context knowledge to avoid unnecessary preview calls\n\n"
         "INTERPRETATION EXAMPLES:\n"
         "✅ GOOD: \"Feature A has coefficient 0.45 (95% CI: 0.32-0.58, p<0.001), increasing target by 0.45 units per unit increase. This is 2.1× stronger than Feature B (coef: 0.21, p=0.03).\"\n"
         "❌ BAD: \"Feature A is the most important predictor\"\n\n"
         "✅ GOOD: \"Model achieves R²=0.73, explaining 73% of variance. Test RMSE=12.4 suggests predictions typically within ±12.4 units of actual values. Suitable for decision-making.\"\n"
         "❌ BAD: \"The model performs well\"\n\n"
         "MODEL QUALITY ASSESSMENT:\n"
         "Linear Regression (R²): ≥0.7 Excellent | 0.5-0.7 Good | 0.3-0.5 Moderate | <0.3 Poor\n"
         "Classification (AUC): ≥0.8 Excellent | 0.7-0.8 Good | 0.6-0.7 Fair | <0.6 Poor\n"
         "Always compare train vs test performance (>10% difference = potential overfitting)\n"
         "Conclude with: 'ready for production', 'needs improvement', or 'not suitable for decisions'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4) Build agent + executor
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,   # prevents crashes on freeform outputs
        return_intermediate_steps=True,
        max_iterations=15  # Prevent infinite loops
    )

    # 5) Convert chat history to langchain format
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(("ai", msg["content"]))

    # 6) Run with chat history
    result = executor.invoke({
        "input": user_query,
        "chat_history": messages
    })
    
    # Sanitize output to remove hallucinated images
    result = sanitize_agent_output(result)
    
    # Add version tracking for metrics correlation
    result['prompt_version'] = PROMPT_VERSION
    
    return result

