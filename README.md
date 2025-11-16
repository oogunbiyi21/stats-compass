<div align="center">
  <img src="./assets/logo/logo1.png" alt="Stats Compass Logo" width="200"/>
  
  <h1>🧭 Stats Compass</h1>
  
  <p>An AI-powered data analysis tool that lets you explore datasets through natural language conversations. Upload CSV files and ask questions about your data.</p>
  
  <a href="https://stats-compass-beta.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Get%20Started-4CAF50?style=for-the-badge&logo=streamlit&logoColor=white" alt="Get Started"/>
  </a>
</div>

## 🎮 Features (v1.0.0)

### ✅ What's Included
- **50+ Analysis Tools**: From basic stats to advanced ML models
- **Natural Language Interface**: Ask questions in plain English
- **Data Exploration**: Schema, statistics, correlations, groupby, filtering
- **10+ Visualizations**: Line, scatter, bar, histogram, heatmap, box plot, and more
- **Data Cleaning**: Automated suggestions, imputation, outlier handling
- **Machine Learning**: Linear/logistic regression, ARIMA time series
- **Statistical Tests**: T-tests, z-tests, chi-square, ANOVA
- **Model Export**: Download trained models as .joblib files
- **Chart Export**: Save visualizations as PNG images
- **15+ Sample Datasets**: Practice with built-in data

### ⚠️ Limitations
- **Requires GPT-4 API key** (costs ~$0.01-0.05 per analysis)
- **Best with <100k rows** (memory constraints for larger datasets)
- **Tabular data only** (CSV, Excel, JSON, Parquet)
- **No SQL support** (yet - coming in v1.1.0)
- **English only** natural language processingral language conversations. Upload CSV files and ask questions about your data.

## 📦 **Stable Release vs Development**

### ✅ **For Users: Use v1.0.0 Release**
- **[📥 Download Stable Release](https://github.com/oogunbiyi21/ds-auto-insights/releases/tag/v1.0.0)**
- Production-ready with 50+ analysis tools
- Comprehensive testing and bug fixes
- Stable API and reliable performance
- Full ML capabilities (regression, classification, time series)

### 🔧 **For Developers: Main Branch**
- Stable v1.0.0 codebase
- Well-documented and production-ready
- Use for contributing or building on top

> 💡 **Recommendation**: Use v1.0.0 release for data analysis work

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

#### Option 1: Stable Release (Recommended)
```bash
# Download and extract the v1.0.0 release
wget https://github.com/oogunbiyi21/ds-auto-insights/archive/refs/tags/v1.0.0.zip
unzip v1.0.0.zip
cd ds-auto-insights-1.0.0
```

#### Option 2: Clone Repository
```bash
git clone https://github.com/oogunbiyi21/ds-auto-insights.git
cd stats-compass
git checkout v1.0.0
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
poetry run streamlit run app.py
```

## 🎮 Current Features

### ✅ What Works
- **File Upload**: CSV/XLSX files (small to medium datasets)
- **Basic Chat**: Ask simple questions about your data
- **Simple Charts**: Histograms, scatter plots, basic time series
- **Data Summaries**: Column info, missing values, basic stats
- **Export**: PDF and Markdown reports (experimental)


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

## 🙏 Acknowledgements

Special thanks to **ArtleSa** (u/ArtleSa on Reddit) for the landing page template inspiration.

---

*An experimental tool for exploring data through conversation.*
