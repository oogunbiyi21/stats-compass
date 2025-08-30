# DS Auto Insights

A PM-first, chat-native data analysis tool that empowers product managers and lean teams to turn raw datasets into actionable insights through natural language conversations with intelligent dataset awareness and smart suggestions.

## 🎯 Problem Statement

Non-technical PMs and lean teams struggle to turn raw data into decisions quickly. Existing BI tools (Power BI/Tableau) are great at dashboards but require modeling and setup; chat-first "AI BI" tools often hallucinate or can't actually operate on the data.

## ✨ Solution

DS Auto Insights provides:
- 📁 **Easy data ingestion** - Upload CSV/XLSX files with automatic parsing and schema detection
- 🧠 **Intelligent dataset awareness** - AI immediately knows your data structure without exploration
- 💡 **Smart suggestions** - Proactive analysis recommendations based on your data characteristics
- 💬 **Natural language queries** - Ask questions about your data in plain English
- 🔒 **Safe computations** - All insights come from real operations on your dataframe (no hallucinations)
- 📊 **Rich visualizations** - Interactive charts including time series, correlation heatmaps, and distributions
- 📋 **Comprehensive exports** - Share results as PDF reports, Markdown files, or chart collections
- ⚡ **Persistent charts** - Visualizations persist across queries and export correctly

## 🏗️ Architecture

- **Frontend**: Streamlit with enhanced chart persistence and export capabilities
- **AI Agent**: LangChain + OpenAI (GPT-4o) with intelligent tool orchestration
- **Dataset Context**: Automatic schema injection for immediate data understanding
- **Chart System**: Advanced visualization pipeline with time series and correlation analysis
- **Export Engine**: Multi-format export system (PDF, Markdown, PNG, JSON)
- **Safety Layer**: Whitelisted pandas operations with comprehensive error handling

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

### ✅ Core Capabilities
- **File Upload**: CSV/XLSX with encoding detection and comprehensive error handling
- **Intelligent Dataset Awareness**: AI immediately understands your data structure, columns, and types
- **Smart Suggestions**: Proactive analysis recommendations based on data characteristics
- **Advanced Visualizations**: Time series analysis, correlation heatmaps, distributions, and scatter plots
- **Chart Persistence**: Visualizations maintain state across queries and display correctly
- **Multi-format Export**: PDF reports, Markdown documentation, PNG charts, and JSON session data
- **Safe Operations**: Whitelisted pandas functions with transparent computations

### 📊 Visualization Toolkit
- **Time Series Analysis**: Automatic trend detection with statistical insights
- **Correlation Heatmaps**: Interactive relationship visualization between variables
- **Distribution Charts**: Histograms and frequency analysis for data understanding
- **Scatter Plots**: Relationship exploration with correlation statistics
- **Export Pipeline**: Charts save correctly and appear in all export formats

### 🎯 Smart Features
- **Dataset Context Injection**: AI knows all column names, types, and sample values immediately
- **Proactive Suggestions**: Relevant analysis recommendations based on data patterns
- **Level 1 Smart Suggestions**: Dataset-driven analysis recommendations
- **Enhanced UX**: No more "what columns do you have?" - AI knows your data structure

### 🛠️ Technical Improvements
- **Chart Object Management**: Proper serialization and recreation for persistence
- **Multiple Chart Support**: Generate multiple time series without data conflicts
- **Error Handling**: Comprehensive error catching and user-friendly messages
- **Performance Optimization**: Efficient data processing and visualization rendering

## 📖 Usage Examples

1. **Upload your dataset** (CSV or XLSX) - AI immediately understands structure
2. **Review smart suggestions** - See proactive analysis recommendations
3. **Explore the "What the AI knows" section** - Verify dataset understanding
4. **Ask natural language questions** like:
   - "Show me time series for revenue and user growth"
   - "Create a correlation heatmap for all numeric variables"
   - "What are the top 5 categories by sales volume?"
   - "Analyze trends in customer engagement over time"
   - "Generate a comprehensive analysis report"
5. **Export your results** - PDF reports, charts, or session data

### 📁 Example Datasets & Outputs

Check out the `/examples` folder for real analysis examples:

- **📈 Stock Market Analysis** (`examples/stock_market/`)
  - Time series analysis of TATA MOTORS stock data
  - PDF reports with trend analysis and statistics
  - Distribution visualizations and insights

- **⚽ Football Analytics** (`examples/football/`) 
  - Player performance analysis (Bukayo Saka data)
  - Multi-chart exports with correlation analysis
  - Creative metrics exploration (xG, assists, key passes)

- **🏠 Housing Market** (`examples/housing/`)
  - Real estate price analysis
  - Geographic and feature correlation insights
  - Market trend identification

Each example includes actual generated reports, charts, and session data demonstrating the full export pipeline.

## 🛡️ Safety & Reliability

- **No arbitrary code execution** - Only whitelisted pandas operations
- **Transparent operations** - See exactly what computations are performed
- **Read-only operations** - Your data is never modified
- **Deterministic results** - Same query always produces same output
- **Error resilience** - Comprehensive error handling and recovery
- **Chart validation** - Proper data validation before visualization

## 🗺️ Roadmap

See [PM Data Tool Roadmap.txt](PM%20Data%20Tool%20Roadmap.txt) for detailed development plans.

**✅ Phase 1 (Complete)**: Core functionality with intelligent dataset awareness
**✅ Phase 2 (Complete)**: Enhanced visualization pipeline and export system
**🔄 Phase 3 (In Progress)**: Smart suggestions and advanced context handling
**📋 Phase 4 (Planned)**: SQL connectivity and enterprise features
**🔮 Phase 5 (Future)**: MCP integration and advanced analytics

## 📈 Recent Improvements

### v1.2.0 - Smart Dataset Awareness
- ✅ Intelligent dataset context injection
- ✅ Automatic schema detection and AI knowledge
- ✅ Smart analysis suggestions based on data characteristics
- ✅ Enhanced user experience with immediate data understanding

### v1.1.0 - Advanced Visualization & Export
- ✅ Time series analysis with trend detection
- ✅ Interactive correlation heatmaps  
- ✅ Comprehensive export system (PDF, Markdown, Charts, JSON)
- ✅ Chart persistence and proper state management
- ✅ Fixed multiple time series generation conflicts

## 🤝 Contributing

This project is in active development. Feel free to:
- Report issues and bugs
- Suggest new features or improvements
- Submit pull requests
- Share feedback and use cases
- Contribute to documentation

## 🚀 Performance & Scalability

- **Optimized for datasets up to 100MB** 
- **Efficient memory usage** with pandas optimizations
- **Fast chart rendering** with Plotly and Streamlit
- **Responsive UI** even with complex visualizations
- **Scalable architecture** ready for enterprise deployment

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

## 🔗 Links

- [Project Documentation](DS%20Auto%20Insights%20—%20where%20we%20are,%20why,%20and%20what's%20next.txt)
- [Use Cases & Examples](PM%20Data%20Tool%20Use%20Cases.txt)
- [Development Roadmap](PM%20Data%20Tool%20Roadmap.txt)

---

**Built with ❤️ for product managers who want data insights without the complexity.**

*Turn your data into decisions in minutes, not hours.*
