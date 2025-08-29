# DS Auto Insights

A PM-first, chat-native data analysis tool that empowers product managers and lean teams to turn raw datasets into actionable insights through natural language conversations.

## 🎯 Problem Statement

Non-technical PMs and lean teams struggle to turn raw CSVs/tables into decisions quickly. Existing BI tools (Power BI/Tableau) are great at dashboards but require modeling and setup; chat-first "AI BI" tools often hallucinate or can't actually operate on the data.

## ✨ Solution

DS Auto Insights provides:
- 📁 **Easy data ingestion** - Upload CSV/XLSX files with parsing
- 💬 **Natural language queries** - Ask questions about your data in plain English
- 🔒 **Safe computations** - All insights come from real operations on your dataframe (no hallucinations)
- 📊 **Interactive dashboards** - Get visualizations and summaries automatically
- 📋 **Export capabilities** - Share results as Markdown reports

## 🏗️ Architecture

- **Frontend**: Streamlit
- **AI Agent**: LangChain + OpenAI (GPT-4o) with tool-calling capabilities
- **Safety Layer**: Whitelisted pandas operations only
- **Data Handling**: CSV/XLSX parsing with pandas
- **Tools**: `RunPandasQueryTool` for safe, read-only dataframe operations

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Poetry
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/oogunbiyi21/ds-auto-insights.git
cd ds-auto-insights
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

4. Run the application:
```bash
poetry run streamlit run ds_auto_insights/app.py
```

## 🎮 Features

### Current Capabilities
- ✅ **File Upload**: CSV/XLSX with encoding detection and error handling
- ✅ **Data Summary**: Automatic dataset overview with missing values and descriptive statistics
- ✅ **Exploration**: Correlation matrices and suggested visualizations
- ✅ **Chat Interface**: Natural language queries with real pandas computations
- ✅ **Safe Operations**: Whitelisted pandas functions

### Coming Soon
- 💾 **Export/Narrative**: One-click Markdown reports from chat results
- 🔌 **SQL Connectors**: Direct database connections
- 🎯 **Context Slotting**: Persistent business context for better analysis

## 📖 Usage Examples

1. **Upload your dataset** (CSV or XLSX)
2. **Explore automatically generated summaries** in the Summary tab
3. **Ask natural language questions** like:
   - "What are the top 5 categories by revenue?"
   - "Show me correlation between price and sales"
   - "Which features predict customer churn?"
   - "Group customers by region and show average spend"

## 🛡️ Safety First

- **No arbitrary code execution** - Only whitelisted pandas operations
- **Transparent operations** - See exactly what computations are performed
- **Read-only operations** - Your data is never modified
- **Deterministic results** - Same query always produces same output

## 🗺️ Roadmap

See [PM Data Tool Roadmap.txt](PM%20Data%20Tool%20Roadmap.txt) for detailed development plans.

**Phase 1 (Current)**: Core functionality with safe pandas operations
**Phase 2**: Enhanced tools and visualization pipeline  
**Phase 3**: SQL connectivity and advanced context handling
**Phase 4**: MCP integration and enterprise features

## 🤝 Contributing

This project is in active development. Feel free to:
- Report issues
- Suggest features
- Submit pull requests
- Share feedback

## 📝 License

mit license

## 🔗 Links

- [Project Documentation](DS%20Auto%20Insights%20—%20where%20we%20are,%20why,%20and%20what's%20next.txt)
- [Use Cases](PM%20Data%20Tool%20Use%20Cases.txt)

---

Built with ❤️ for product managers who want data insights without the complexity.
