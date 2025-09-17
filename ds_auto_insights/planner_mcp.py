# planner_mcp.py

from typing import Any, Dict, List
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ds_auto_insights.tools.exploration_tools import (
    RunPandasQueryTool,
    GroupByAggregateTool,
    TopCategoriesTool,
    HistogramTool,
    CorrelationMatrixTool,
    DatasetPreviewTool,
    CreateHistogramChartTool
)
from ds_auto_insights.tools.chart_tools import (
    CreateBarChartTool,
    CreateScatterPlotTool,
    CreateLineChartTool,
    CreateColumnTool,
    TimeSeriesAnalysisTool,
    CreateCorrelationHeatmapTool
)
from ds_auto_insights.tools.data_cleaning_tools import (
    AnalyzeMissingDataTool,
    DetectOutliersTool,
    FindDuplicatesTool,
    ApplyBasicCleaningTool,
    SuggestDataCleaningActionsTool,
    SuggestImputationStrategiesTool,
    ApplyImputationTool
)
from ds_auto_insights.tools.statistical_test_tools import (
    RunTTestTool,
    RunZTestTool,
    RunChiSquareTestTool
)

def generate_dataset_context(df: pd.DataFrame) -> str:
    """Generate a comprehensive context string about the dataset for the LLM"""
    
    # Basic info
    num_rows, num_cols = df.shape
    
    # Column information with types and sample values
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()
        
        # Get sample values (non-null)
        sample_values = df[col].dropna().unique()[:3]  # First 3 unique values
        sample_str = ', '.join([str(v) for v in sample_values])
        if len(df[col].dropna().unique()) > 3:
            sample_str += '...'
        
        column_info.append(f"  • {col} ({dtype}): {non_null_count} non-null values, examples: {sample_str}")
    
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




def run_mcp_planner(user_query: str, df: pd.DataFrame, chat_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Tool-calling agent wired to your RunPandasQueryTool.
    Now includes chat history for context preservation and automatic dataset context.
    Returns the AgentExecutor invoke() output (dict with 'output' and possibly intermediate steps).
    """
    if chat_history is None:
        chat_history = []

    # Generate dataset context automatically
    dataset_context = generate_dataset_context(df)

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
    
    # Time series analysis tools
    time_series_analysis_tool = TimeSeriesAnalysisTool(df=df)
    correlation_heatmap_tool = CreateCorrelationHeatmapTool(df=df)
    
    # Data transformation tools
    create_column_tool = CreateColumnTool(df=df)
    
    # Data Cleaning Tools
    analyze_missing_tool = AnalyzeMissingDataTool(df=df)
    detect_outliers_tool = DetectOutliersTool(df=df)
    find_duplicates_tool = FindDuplicatesTool(df=df)
    apply_cleaning_tool = ApplyBasicCleaningTool(df=df)
    suggest_cleaning_tool = SuggestDataCleaningActionsTool(df=df)
    
    # Data Imputation Tools
    suggest_imputation_tool = SuggestImputationStrategiesTool(df=df)
    apply_imputation_tool = ApplyImputationTool(df=df)
    
    # Statistical Analysis Tools
    t_test_tool = RunTTestTool(df=df)
    z_test_tool = RunZTestTool(df=df)
    chi_square_test_tool = RunChiSquareTestTool(df=df)
    
    tools = [
        pandas_query_tool, groupby_tool, top_categories_tool, histogram_tool, 
        correlation_tool, dataset_preview_tool, histogram_chart_tool, 
        bar_chart_tool, scatter_plot_tool, line_chart_tool, create_column_tool,
        time_series_analysis_tool, correlation_heatmap_tool,
        # Data cleaning tools
        analyze_missing_tool, detect_outliers_tool, find_duplicates_tool,
        apply_cleaning_tool, suggest_cleaning_tool,
        # Data imputation tools
        suggest_imputation_tool, apply_imputation_tool,
        # Statistical analysis tools
        t_test_tool, z_test_tool, chi_square_test_tool
    ]

    # 2) LLM (swap to Claude/Gemini later by changing the Chat* class)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3) Enhanced prompt with dataset context and chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful data analysis assistant. "
         "You have access to a pandas DataFrame named `df` and several specialized tools.\n\n"
         f"{dataset_context}\n\n"
         "DATA ANALYSIS TOOLS:\n"
         "- dataset_preview: Get a complete view of the dataset with ALL columns visible (use this instead of df.head())\n"
         "- run_pandas_query: For custom pandas expressions (SINGLE-LINE expressions only, no assignments)\n"
         "  • For missing dates: pd.date_range(start=pd.to_datetime(df['Date'].dropna()).min(), end=pd.to_datetime(df['Date'].dropna()).max(), freq='B').difference(pd.to_datetime(df['Date'].dropna()))\n"
         "  • Use ONLY when specialized tools can't do the job\n"
         "- groupby_aggregate: Group data and calculate aggregations (mean, sum, count, etc.) - USE THIS for any grouping questions\n"
         "- top_categories: Find most frequent values in categorical columns - USE THIS for 'top X' questions\n"
         "- histogram: Analyze distribution of numeric columns - USE THIS for distribution/histogram questions\n"
         "- correlation_matrix: Calculate correlations between numeric columns - USE THIS for correlation questions\n"
         "- time_series_analysis: Analyze trends and patterns over time - AUTOMATICALLY CREATES CHARTS\n\n"
         "DATA CLEANING TOOLS (Competitive Advantage - Use These Intelligently):\n"
         "- analyze_missing_data: Comprehensive missing data analysis with patterns and correlations\n"
         "- detect_outliers: Advanced outlier detection using IQR, Z-score, or modified Z-score methods\n"
         "- find_duplicates: Analyze duplicate rows and uniqueness patterns across columns\n"
         "- suggest_data_cleaning: AI-powered recommendations for data quality improvements\n"
         "- apply_data_cleaning: Execute cleaning actions (remove empty columns, duplicates, high-missing columns)\n\n"
         "DATA IMPUTATION TOOLS (Advanced Missing Data Handling):\n"
         "- suggest_imputation_strategies: Get smart recommendations for handling missing values in each column\n"
         "- apply_imputation: Apply imputation using 'auto' mode (AI recommendations) or 'custom' mode (specific methods)\n\n"
         "DATA TRANSFORMATION TOOLS:\n"
         "- create_column: Create new columns using pandas operations (calculations, conditions, transformations)\n"
         "  Examples: create opponent column, calculate ratios, create categorical bins, etc.\n\n"
         "STATISTICAL ANALYSIS TOOLS (Rigorous Hypothesis Testing):\n"
         "- run_t_test: Perform one-sample, two-sample, or paired t-tests with full statistical interpretation\n"
         "  • Best for smaller samples (n<30) or when population std is unknown\n"
         "  • One-sample: Test if mean differs from a specific value\n"
         "  • Two-sample: Compare means between two groups (use group_values=['Group1', 'Group2'] to specify which groups)\n"
         "  • Paired: Compare two related measurements\n"
         "  • Includes effect sizes, assumptions checking, and business interpretation\n"
         "  • Automatically creates appropriate visualizations (histograms, box plots, scatter plots)\n"
         "- run_z_test: Perform one-sample, two-sample, or paired z-tests with normal distribution\n"
         "  • Best for large samples (n≥30) or when population standard deviation is known\n"
         "  • Same test types as t-test but uses normal distribution instead of t-distribution\n"
         "  • Include population_std parameter if known, otherwise sample std is used\n"
         "  • Provides warnings when sample size is too small for z-test validity\n"
         "- run_chi_square_test: Perform chi-square tests for categorical data analysis\n"
         "  • Independence test: Test relationship between two categorical variables (requires column1 and column2)\n"
         "  • Goodness of fit test: Test if data follows expected distribution (requires only column1)\n"
         "  • For goodness of fit, use expected_frequencies parameter or assume equal frequencies\n"
         "  • Includes effect size (Cramér's V), contingency tables, and assumption checking\n"
         "  • Creates heatmaps for independence tests and bar charts for goodness of fit\n\n"
         "CHART CREATION TOOLS:\n"
         "- create_histogram_chart: Create visual histogram charts for numeric data distributions\n"
         "- create_bar_chart: Create visual bar charts for categorical data (top categories, counts)\n"
         "- create_scatter_plot: Create scatter plots to visualize relationships between two numeric variables\n"
         "- create_line_chart: Create line charts for trends over time or ordered data\n"
         "- create_correlation_heatmap: Create visual correlation heatmaps showing variable relationships\n\n"
         "SMART DATA CLEANING STRATEGY:\n"
         "1. When users upload data or ask about data quality, automatically use suggest_data_cleaning\n"
         "2. For any data quality concerns, use specific analysis tools (analyze_missing_data, detect_outliers, find_duplicates)\n"
         "3. Proactively suggest and apply cleaning when you detect quality issues\n"
         "4. Always explain what cleaning actions will do before applying them\n"
         "5. After cleaning, mention the improved data quality for better analysis results\n\n"
         "PRIORITY GUIDELINES:\n"
         "1. You KNOW the dataset structure - use the column names directly without needing dataset_preview\n"
         "2. For visualization requests (charts, plots, graphs), ALWAYS use the chart creation tools\n"
         "3. When users ask to 'show', 'plot', 'visualize', or 'chart' data, use create_*_chart tools\n"
         "4. For creating new columns or data transformations, use create_column tool\n"
         "5. INTELLIGENTLY suggest data cleaning when you notice potential quality issues\n"
         "6. Always try specialized tools FIRST. Only use run_pandas_query as a last resort\n"
         "7. Use the provided column information to answer questions immediately\n"
         "8. For analysis + visualization, do the analysis first, then create the chart\n\n"
         "SMART ANALYSIS: You can immediately answer questions about available columns, data types, "
         "and suggest appropriate analysis without running dataset_preview first. Use your knowledge "
         "of the dataset structure to provide intelligent recommendations. Be proactive about data quality!"),
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
