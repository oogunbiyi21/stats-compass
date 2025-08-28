# planner_mcp.py

from typing import Any, Dict, List
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .mcp_tools import (
    RunPandasQueryTool,
    GroupByAggregateTool,
    TopCategoriesTool,
    HistogramTool,
    CorrelationMatrixTool
)


def run_mcp_planner(user_query: str, df: pd.DataFrame, chat_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Tool-calling agent wired to your RunPandasQueryTool.
    Now includes chat history for context preservation.
    Returns the AgentExecutor invoke() output (dict with 'output' and possibly intermediate steps).
    """
    if chat_history is None:
        chat_history = []

    # 1) Instantiate your tools
    pandas_query_tool = RunPandasQueryTool(df=df)
    groupby_tool = GroupByAggregateTool(df=df)
    top_categories_tool = TopCategoriesTool(df=df)
    histogram_tool = HistogramTool(df=df)
    correlation_tool = CorrelationMatrixTool(df=df)
    
    tools = [pandas_query_tool, groupby_tool, top_categories_tool, histogram_tool, correlation_tool]

    # 2) LLM (swap to Claude/Gemini later by changing the Chat* class)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3) Enhanced prompt with chat history context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful data analysis assistant. "
         "You have access to a pandas DataFrame named `df` and several specialized tools:\n"
         "- run_pandas_query: For custom pandas expressions (use sparingly)\n"
         "- groupby_aggregate: Group data and calculate aggregations (mean, sum, count, etc.)\n"
         "- top_categories: Find most frequent values in categorical columns\n"
         "- histogram: Analyze distribution of numeric columns\n"
         "- correlation_matrix: Calculate correlations between numeric columns\n\n"
         "PREFER the specialized tools over run_pandas_query when possible - they're safer and more reliable.\n"
         "You maintain context across the conversation - if you've previously identified "
         "information about the dataset (like player names, data types, etc.), remember it. "
         "Refer to previous analysis and build upon it in your responses."),
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
        return_intermediate_steps=True
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
    return result
