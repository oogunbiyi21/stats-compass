# stats_compass/utils/token_tracking.py
"""
Token usage and cost tracking utilities for DS Auto Insights.
Provides transparent cost monitoring for users.
"""

import tiktoken
import streamlit as st
from typing import Dict, Tuple, Optional
from datetime import datetime


# OpenAI GPT-4 pricing (as of 2024)
OPENAI_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: The text to count tokens for
        model: The model name to get the correct encoding
    
    Returns:
        Number of tokens
    """
    try:
        # Get the encoding for the model
        if "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Fallback to cl100k_base encoding (used by GPT-4)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    """
    Calculate the cost for a given number of input and output tokens.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name for pricing
    
    Returns:
        Total cost in USD
    """
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4o"])
    
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    
    return input_cost + output_cost


def track_usage(user_message: str, assistant_response: str, model: str = "gpt-4o") -> Dict:
    """
    Track token usage and cost for an interaction.
    
    Args:
        user_message: The user's input message
        assistant_response: The AI's response
        model: Model name for accurate counting and pricing
    
    Returns:
        Dictionary with usage statistics
    """
    input_tokens = count_tokens(user_message, model)
    output_tokens = count_tokens(assistant_response, model)
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(input_tokens, output_tokens, model)
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "model": model,
        "timestamp": datetime.now().isoformat()
    }


def update_session_usage(usage_stats: Dict):
    """
    Update the session state with cumulative usage statistics.
    
    Args:
        usage_stats: Usage statistics from track_usage()
    """
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "usage_history" not in st.session_state:
        st.session_state.usage_history = []
    
    # Update totals
    st.session_state.total_tokens_used += usage_stats["total_tokens"]
    st.session_state.total_cost += usage_stats["cost"]
    
    # Add to history
    st.session_state.usage_history.append(usage_stats)


def format_cost_display(cost: float) -> str:
    """
    Format cost for display with appropriate precision.
    
    Args:
        cost: Cost in USD
    
    Returns:
        Formatted cost string
    """
    if cost >= 1:
        return f"${cost:.2f}"
    elif cost >= 0.01:
        return f"${cost:.3f}"
    else:
        return f"${cost:.4f}"


def get_usage_summary() -> Tuple[int, float, str]:
    """
    Get current session usage summary.
    
    Returns:
        Tuple of (total_tokens, total_cost, formatted_display)
    """
    total_tokens = st.session_state.get("total_tokens_used", 0)
    total_cost = st.session_state.get("total_cost", 0.0)
    
    if total_tokens == 0:
        display = "💰 No usage yet"
    else:
        cost_str = format_cost_display(total_cost)
        display = f"💰 {total_tokens:,} tokens | {cost_str}"
    
    return total_tokens, total_cost, display


def check_usage_limits(tokens: int, cost: float) -> Optional[str]:
    """
    Check if usage is approaching limits and return warning message.
    
    Args:
        tokens: Current token count
        cost: Current cost
    
    Returns:
        Warning message if approaching limits, None otherwise
    """
    # Define warning thresholds
    TOKEN_WARNING = 50000  # 50K tokens
    TOKEN_LIMIT = 100000   # 100K tokens
    COST_WARNING = 5.0     # $5
    COST_LIMIT = 10.0      # $10
    
    if tokens >= TOKEN_LIMIT or cost >= COST_LIMIT:
        return f"🚨 **Usage Limit Reached!** ({tokens:,} tokens, {format_cost_display(cost)})"
    elif tokens >= TOKEN_WARNING or cost >= COST_WARNING:
        return f"⚠️ **High Usage Warning** ({tokens:,} tokens, {format_cost_display(cost)})"
    else:
        return None


def get_session_stats(chat_history: list) -> dict:
    """
    Calculate session statistics for reporting.
    
    Args:
        chat_history: List of chat messages
    
    Returns:
        Dictionary with session statistics
    """
    user_questions = [msg for msg in chat_history if msg["role"] == "user"]
    assistant_responses = [msg for msg in chat_history if msg["role"] == "assistant"]
    total_charts = sum(len(msg.get("charts", [])) for msg in assistant_responses)
    
    return {
        "user_questions": len(user_questions),
        "assistant_responses": len(assistant_responses),
        "total_charts": total_charts,
        "has_analysis": bool(chat_history)
    }
