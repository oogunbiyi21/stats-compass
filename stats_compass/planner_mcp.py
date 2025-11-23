# planner_mcp.py

from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools.registry import ToolRegistry
from prompts.versions import PROMPT_VERSION


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


def _extract_tool_name(action: Any) -> str:
    """
    Extract tool name from an action object using multiple fallback strategies.
    
    Args:
        action: Action object from LangChain agent (could be AgentAction, dict, etc.)
        
    Returns:
        Tool name string, or "unknown_tool" if extraction fails
    """
    # Strategy 1: Check common attribute names
    if hasattr(action, 'tool'):
        return action.tool
    if hasattr(action, 'tool_name'):
        return action.tool_name
    
    # Strategy 2: Check if it's a dict
    if isinstance(action, dict):
        if 'tool' in action:
            return action['tool']
        if 'name' in action:
            return action['name']
    
    # Strategy 3: Parse from string representation
    action_str = str(action)
    if 'tool=' in action_str:
        match = re.search(r"tool='([^']+)'", action_str)
        if match:
            return match.group(1)
    
    return "unknown_tool"


def _extract_tool_input(action: Any) -> Dict[str, Any]:
    """
    Extract tool input from an action object using multiple fallback strategies.
    
    Args:
        action: Action object from LangChain agent
        
    Returns:
        Tool input dictionary (empty dict if extraction fails)
    """
    if hasattr(action, 'tool_input'):
        return action.tool_input
    if hasattr(action, 'input'):
        return action.input
    if isinstance(action, dict) and 'tool_input' in action:
        return action['tool_input']
    
    return {}


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


def _build_system_prompt(dataset_context: str) -> str:
    """Build the system prompt with dataset context (shared between streaming and non-streaming)."""
    return (
        "You are a careful data analysis assistant with access to a pandas DataFrame and specialized tools.\n\n"
        f"{dataset_context}\n\n"
        "TOOL USAGE PHILOSOPHY:\n"
        "You have access to tools across these categories:\n"
        "• Data Exploration: Query, group, aggregate, preview, analyze distributions and correlations\n"
        "• Data Cleaning: Analyze and fix missing data, outliers, duplicates; smart imputation strategies\n"
        "• Statistical Analysis: Hypothesis testing (t-test, z-test, chi-square) with visualizations\n"
        "• Machine Learning: Regression, classification, time series forecasting, evaluation, preprocessing\n"
        "• Visualization: Charts, plots, heatmaps, and model diagnostic visualizations\n\n"
        "Use specialized tools first.\n"
        "Tool details are provided in the function calling interface below.\n\n"
        "WORKFLOWS:\n"
        "Data Cleaning: suggest_data_cleaning → analyze (missing/outliers/duplicates) → explain actions → apply\n"
        "Machine Learning: check data quality (>10% missing = clean first) → preprocess categoricals → train → evaluate → visualize → interpret\n"
        "ARIMA Time Series: If user requests ARIMA forecast WITHOUT specifying (p,d,q) parameters, include this one-time warning in your response: 'I can use find_optimal_arima_parameters for automatic parameter selection (takes 2-5 min) or default parameters (p=1,d=1,q=1) for faster results. Which do you prefer?' Then wait for their choice.\n"
        "For multi-step operations, briefly outline your plan before starting.\n\n"
        "CORE PRINCIPLES:\n"
        "1. Use specialized tools first\n"
        "2. Complete ML workflows fully: train → evaluate → visualize → interpret (don't wait for prompts)\n"
        "3. Create visualizations when users ask to 'show' or 'plot' data\n"
        "4. Provide specific, quantitative interpretations with actual numbers\n"
        "5. Use dataset context knowledge to avoid unnecessary preview calls\n"
        "6. Never loop through years/categories manually - use groupby or inspect_data with aggregation\n"
        "7. If a task requires >3 tool calls, rethink your approach for efficiency\n\n"
        "INTERPRETATION EXAMPLES:\n"
        "✅ GOOD: \"Feature A has coefficient 0.45 (95% CI: 0.32-0.58, p<0.001), increasing target by 0.45 units per unit increase. This is 2.1× stronger than Feature B (coef: 0.21, p=0.03).\"\n"
        "❌ BAD: \"Feature A is the most important predictor\"\n\n"
        "✅ GOOD: \"Model achieves R²=0.73, explaining 73% of variance. Test RMSE=12.4 suggests predictions typically within ±12.4 units of actual values. Suitable for decision-making.\"\n"
        "❌ BAD: \"The model performs well\"\n\n"
        "MODEL QUALITY ASSESSMENT:\n"
        "Linear Regression (R²): ≥0.7 Excellent | 0.5-0.7 Good | 0.3-0.5 Moderate | <0.3 Poor\n"
        "Classification (AUC): ≥0.8 Excellent | 0.7-0.8 Good | 0.6-0.7 Fair | <0.6 Poor\n"
        "Always compare train vs test performance (>10% difference = potential overfitting)\n"
        "Conclude with: 'ready for production', 'needs improvement', or 'not suitable for decisions'"
    )


def _create_agent_executor(df: pd.DataFrame, api_key: str = None, streaming: bool = False):
    """
    Create and configure the agent executor (shared setup for streaming and non-streaming).
    
    Args:
        df: DataFrame to analyze
        api_key: Optional OpenAI API key
        streaming: Whether to enable streaming mode
        
    Returns:
        tuple: (executor, dataset_context) for use in invoke/stream calls
    """
    # Generate dataset context
    dataset_context = generate_dataset_context(df)
    
    # Get all tools from registry
    tools = ToolRegistry.get_all_tools(df)
    
    # LLM configuration
    llm_kwargs = {
        "model": "gpt-4o",
        "temperature": 0,
        "streaming": streaming
    }
    if api_key:
        llm_kwargs["api_key"] = api_key
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # Build prompt with dataset context
    system_prompt = _build_system_prompt(dataset_context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Build agent + executor
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=10
    )
    
    return executor, dataset_context


def _convert_chat_history(chat_history: List[Dict]) -> List:
    """Convert chat history from dict format to LangChain message tuples."""
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(("ai", msg["content"]))
    return messages


def run_mcp_planner_stream(user_query: str, df: pd.DataFrame, chat_history: List[Dict] = None, api_key: str = None):
    """
    Streaming version of run_mcp_planner with enhanced tool progress tracking.
    
    This provides improved feedback during agent execution:
    - Shows when tools are being called (tool_start events)
    - Shows when tools complete (tool_end events)  
    - Streams final output line-by-line with formatting preserved
    
    While not true token-level streaming (LangChain limitation), this gives
    much better visibility into what the agent is doing during long operations.
    
    Yields chunks:
    - {"type": "tool_start", "name": str, "input": dict} - Tool starting
    - {"type": "tool_end", "name": str, "output": str} - Tool completed
    - {"type": "token", "content": str} - Output text (line-by-line)
    - {"type": "final", "output": str, "intermediate_steps": list} - Complete
    
    Usage:
        for chunk in run_mcp_planner_stream(query, df, history, api_key):
            if chunk["type"] == "tool_start":
                show_info(f"🔧 {chunk['name']}...")
            elif chunk["type"] == "token":
                display_text += chunk["content"]
    """
    if chat_history is None:
        chat_history = []

    # Create agent executor with streaming enabled
    executor, _ = _create_agent_executor(df, api_key, streaming=True)
    
    # Convert chat history to LangChain format
    messages = _convert_chat_history(chat_history)

    # Stream the agent execution with tool tracking
    full_output = ""
    intermediate_steps = []
    
    # Yield initial thinking indicator
    yield {
        "type": "thinking",
        "content": "Analyzing your request..."
    }
    
    try:
        # AgentExecutor.stream() yields chunks as execution progresses
        for chunk in executor.stream({
            "input": user_query,
            "chat_history": messages
        }):
            if isinstance(chunk, dict):
                # Tool calls happening - extract and report them
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        tool_name = _extract_tool_name(action)
                        tool_input = _extract_tool_input(action)
                        
                        # Yield tool start event
                        yield {
                            "type": "tool_start",
                            "name": tool_name,
                            "input": tool_input
                        }
                
                # Collect intermediate steps (preserve for display)
                if "steps" in chunk:
                    for step in chunk["steps"]:
                        intermediate_steps.append(step)
                        
                        # Extract tool info from completed step
                        action, observation = step if isinstance(step, tuple) else (step, None)
                        tool_name = _extract_tool_name(action)
                        
                        # Yield tool end event
                        yield {
                            "type": "tool_end",
                            "name": tool_name,
                            "output": str(observation) if observation else ""
                        }
                
                # Final output arrived - stream it preserving formatting
                if "output" in chunk:
                    output_text = chunk["output"]
                    full_output = output_text
                    
                    # Stream line-by-line to preserve markdown
                    lines = output_text.split('\n')
                    for i, line in enumerate(lines):
                        yield {
                            "type": "token",
                            "content": line
                        }
                        
                        # Add newline if not last line
                        if i < len(lines) - 1:
                            yield {
                                "type": "token",
                                "content": "\n"
                            }
        
        # Sanitize final output
        result = {"output": full_output, "intermediate_steps": intermediate_steps}
        result = sanitize_agent_output(result)
        result['prompt_version'] = PROMPT_VERSION
        
        # Yield final result
        yield {
            "type": "final",
            "output": result.get("output", full_output),
            "intermediate_steps": result.get("intermediate_steps", intermediate_steps),
            "prompt_version": PROMPT_VERSION,
            "hallucination_detected": result.get("hallucination_detected", False)
        }
        
    except Exception as e:
        # Yield error as final result
        yield {
            "type": "final",
            "output": f"❌ Agent error: {e}",
            "intermediate_steps": intermediate_steps,
            "error": str(e)
        }
