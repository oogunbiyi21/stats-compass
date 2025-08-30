# DS Auto Insights

An experimental data analysis tool for product managers to explore datasets through natural language conversations. Upload CSV files and ask questions about your data.

> ⚠️ **Early MVP**: This is a prototype tool for experimentation and learning. Not recommended for production use.

## 🎯 Problem Statement

Non-technical PMs and lean teams struggle to turn raw data into decisions quickly. Existing BI tools (Power BI/Tableau) are great at dashboards but require modeling and setup; chat-first "AI BI" tools often hallucinate or can't actually operate on the data.

## ✨ What it does

This prototype lets you:
- 📁 **Upload CSV/XLSX files** and get basic data summaries
- 💬 **Ask questions in plain English** about your data
- 📊 **Generate simple charts** like histograms, scatter plots, and time series
- 📋 **Export results** as PDF reports or Markdown files
- 🔒 **Safe operations** - only reads your data, never modifies it

## 🏗️ How it works

- **Frontend**: Streamlit app for file upload and chat interface
- **AI**: GPT-4o with pandas operations for data analysis
- **Safety**: Only whitelisted pandas functions, no code execution
- **Visualizations**: Basic charts using Plotly and Streamlit
- **Export**: Simple PDF and Markdown report generation

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

## 🎮 Current Features

### ✅ What Works
- **File Upload**: CSV/XLSX files (small to medium datasets)
- **Basic Chat**: Ask simple questions about your data
- **Simple Charts**: Histograms, scatter plots, basic time series
- **Data Summaries**: Column info, missing values, basic stats
- **Export**: PDF and Markdown reports (experimental)

### � Limitations
- **Small datasets only** (~1000-10000 rows work best)
- **Basic analysis** - not a replacement for proper data science tools
- **English only** natural language processing
- **Requires OpenAI API key** (costs money per query)
- **No data persistence** - upload fresh each session

## 📖 How to use it

1. **Upload a CSV file** (keep it small - under 10MB works best)
2. **Wait for the summary** to understand your data
3. **Ask simple questions** like:
   - "What are the top 5 values in the category column?"
   - "Show me a histogram of the price column"
   - "What's the correlation between age and income?"
   - "Create a time series chart for sales over time"
4. **Export your results** if you want to save them

> 💡 **Tip**: Start with simple questions. Complex analysis might not work reliably.

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
- ✅ Time series analysis with trend calculation
- ✅ Interactive correlation heatmaps  
- ✅ Comprehensive export system (PDF, Markdown, Charts, JSON)
- ✅ Chart persistence and proper state management
- ✅ Fixed multiple time series generation conflicts

## 🤝 Contributing
This is an experimental project. Feel free to:
- Try it out and report issues
- Suggest improvements or new features
- Fork and experiment
- Share feedback and use cases

## ⚠️ Disclaimer

This is a learning project and proof-of-concept. It's not intended for production use or critical business decisions. Always validate any insights with proper data analysis tools.

## 📝 License

MIT License

---

*An experimental tool for exploring data through conversation.*
