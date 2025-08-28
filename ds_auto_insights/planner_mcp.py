# planner_mcp.py

from typing import Any, Dict
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from mcp_tools import RunPandasQueryTool


def run_mcp_planner(user_query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tool-calling agent wired to your RunPandasQueryTool.
    Returns the AgentExecutor invoke() output (dict with 'output' and possibly intermediate steps).
    """

    # 1) Instantiate your tool (inherits BaseTool)
    pandas_query_tool = RunPandasQueryTool(df=df)
    tools = [pandas_query_tool]

    # 2) LLM (swap to Claude/Gemini later by changing the Chat* class)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3) Prompt MUST include `agent_scratchpad`
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful data analysis assistant. "
         "You have access to a pandas DataFrame named `df` via tools. "
         "Always use tools to compute on the data; do not guess."),
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

    # 5) Run
    result = executor.invoke({"input": user_query})
    return result
