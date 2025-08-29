# planner_mcp.py

from typing import Any, Dict, List
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from .mcp_tools import (
        RunPandasQueryTool,
        GroupByAggregateTool,
        TopCategoriesTool,
        HistogramTool,
        CorrelationMatrixTool,
        DatasetPreviewTool,
        CreateHistogramChartTool,
        CreateBarChartTool,
        CreateScatterPlotTool,
        CreateLineChartTool,
        CreateColumnTool
    )
except ImportError:
    from mcp_tools import (
        RunPandasQueryTool,
        GroupByAggregateTool,
        TopCategoriesTool,
        HistogramTool,
        CorrelationMatrixTool,
        DatasetPreviewTool,
        CreateHistogramChartTool,
        CreateBarChartTool,
        CreateScatterPlotTool,
        CreateLineChartTool,
        CreateColumnTool
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
    dataset_preview_tool = DatasetPreviewTool(df=df)
    
    # Chart rendering tools
    histogram_chart_tool = CreateHistogramChartTool(df=df)
    bar_chart_tool = CreateBarChartTool(df=df)
    scatter_plot_tool = CreateScatterPlotTool(df=df)
    line_chart_tool = CreateLineChartTool(df=df)
    
    # Data transformation tools
    create_column_tool = CreateColumnTool(df=df)
    
    tools = [
        pandas_query_tool, groupby_tool, top_categories_tool, histogram_tool, 
        correlation_tool, dataset_preview_tool, histogram_chart_tool, 
        bar_chart_tool, scatter_plot_tool, line_chart_tool, create_column_tool
    ]

    # 2) LLM (swap to Claude/Gemini later by changing the Chat* class)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3) Enhanced prompt with chat history context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful data analysis assistant. "
         "You have access to a pandas DataFrame named `df` and several specialized tools:\n\n"
         "DATA ANALYSIS TOOLS:\n"
         "- dataset_preview: Get a complete view of the dataset with ALL columns visible (use this instead of df.head())\n"
         "- run_pandas_query: For custom pandas expressions (use ONLY when specialized tools can't do the job)\n"
         "- groupby_aggregate: Group data and calculate aggregations (mean, sum, count, etc.) - USE THIS for any grouping questions\n"
         "- top_categories: Find most frequent values in categorical columns - USE THIS for 'top X' questions\n"
         "- histogram: Analyze distribution of numeric columns - USE THIS for distribution/histogram questions\n"
         "- correlation_matrix: Calculate correlations between numeric columns - USE THIS for correlation questions\n\n"
         "DATA TRANSFORMATION TOOLS:\n"
         "- create_column: Create new columns using pandas operations (calculations, conditions, transformations)\n"
         "  Examples: create opponent column, calculate ratios, create categorical bins, etc.\n\n"
         "CHART CREATION TOOLS:\n"
         "- create_histogram_chart: Create visual histogram charts for numeric data distributions\n"
         "- create_bar_chart: Create visual bar charts for categorical data (top categories, counts)\n"
         "- create_scatter_plot: Create scatter plots to visualize relationships between two numeric variables\n"
         "- create_line_chart: Create line charts for trends over time or ordered data\n\n"
         "PRIORITY GUIDELINES:\n"
         "1. For visualization requests (charts, plots, graphs), ALWAYS use the chart creation tools\n"
         "2. When users ask to 'show', 'plot', 'visualize', or 'chart' data, use create_*_chart tools\n"
         "3. For creating new columns or data transformations, use create_column tool\n"
         "4. Always try specialized tools FIRST. Only use run_pandas_query as a last resort\n"
         "5. Use dataset_preview instead of df.head() to see ALL columns without truncation\n"
         "6. For analysis + visualization, do the analysis first, then create the chart\n\n"
         "CONTEXT: You maintain context across the conversation - if you've previously identified "
         "information about the dataset (like column names, data types, etc.), remember it. "
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
