<div align="center">
  <img src="./assets/logo/logo1.png" alt="Stats Compass Logo" width="200"/>
  
  <h1>🧭 Stats Compass</h1>
  
  <p>AI-powered data analysis through natural language. Upload datasets, ask questions in plain English, get insights.</p>
  
  <p>📖 <a href="https://medium.com/@olatunjiogunbiyi/stats-compass-an-open-source-ai-data-analyst-f457d5808946" target="_blank"><strong>Read the deep-dive blog post</strong></a></p>
  
  <a href="https://stats-compass-beta.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Get%20Started-4CAF50?style=for-the-badge&logo=streamlit&logoColor=white" alt="Get Started"/>
  </a>
</div>

---

<div align="center">
  <img src="./assets/images/demo.gif" alt="Stats Compass Demo" width="800"/>
  <p><em>Chat with your data: Ask questions, get insights, export results</em></p>
</div>

---

## ✨ Features

- **50+ Analysis Tools**: Statistics, ML models (regression, ARIMA), data cleaning
- **10+ Visualizations**: Charts, heatmaps, distributions, time series
- **Natural Language**: Ask questions in plain English
- **Statistical Tests**: Z-tests, T-tests, chi-square, correlations
- **Export Everything**: Models (.joblib), charts (PNG), reports (PDF/Markdown)

**Limitations**: Requires GPT-4 API key (~$0.01-0.05/analysis) • Best with <100k rows • Tabular data only

## 🎯 Why Stats Compass?

Traditional BI tools need setup and technical skills. AI chat tools hallucinate. Stats Compass actually operates on your data with 50+ validated analysis tools. No hallucinations, just real pandas operations guided by AI

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry
- OpenAI API key

### Installation

```bash
git clone https://github.com/oogunbiyi21/stats-compass.git
cd stats-compass
poetry install
cp .env.example .env  # Add your OpenAI API key
poetry run streamlit run stats_compass/app.py
```

## 💡 Usage Tips

Ask questions like:
- "Show correlation between sales and marketing spend"
- "Run linear regression predicting revenue from these features"
- "Create time series forecast for next 30 days using ARIMA"
- "What's the distribution of customer ages?"

**Pro tip**: Start simple, then build complexity. Export models and charts from the Reports tab.

## 📁 Example Analyses

See `/examples` for real-world use cases:
- **Stock Market**: Time series, trend analysis (TATA MOTORS)
- **Football Analytics**: Player performance (Bukayo Saka xG, assists)
- **Housing Market**: Price analysis, geographic trends

## 🤝 Contributing

Issues, suggestions, and PRs welcome. This is production-ready but actively maintained.

## 📝 License

MIT License

## 🙏 Credits

Landing page template by **ArtleSa** (u/ArtleSa)

---

*Data analysis through conversation.*
