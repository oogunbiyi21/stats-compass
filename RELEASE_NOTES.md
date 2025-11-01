# Stats Compass v1.0.0 - Release Notes

**Release Date**: October 29, 2025

First public release of **Stats Compass** - an AI-powered data analysis assistant.

## What is Stats Compass?

Stats Compass is a Streamlit-based application that lets you analyze data using natural language. Instead of writing code, just ask questions like:
- "Show me a correlation matrix"
- "Build a logistic regression model to predict customer churn"
- "Clean the missing data and suggest the best imputation strategy"

The AI agent automatically selects the right tools, executes the analysis, and provides insights in plain English.

## 🚀 Highlights

### Talk to Your Data
No more remembering pandas syntax or sklearn parameters. Just describe what you want:
```
"What's the average sales by region?"
"Plot revenue over time"
"Find outliers in the price column"
```

### End-to-End Workflow
From data cleaning to model training, Stats Compass handles the complete analysis pipeline:
1. **Import** data (CSV, Excel)
2. **Clean** with automated suggestions and one-click fixes
3. **Explore** with analysis tools and visualizations
4. **Model** with regression, classification, and time series
5. **Export** models and charts for production use

### Smart & Reliable
- **Auto-feature selection** filters out problematic columns
- **Data validation** catches issues before model training fails
- **Detailed errors** explain exactly what went wrong and how to fix it
- **Workflow tracking** prevents common mistakes (e.g., modeling before cleaning)

## 📊 What You Can Do

### Data Exploration
- Get schema, statistics, and data previews
- Group by categories and calculate aggregations
- Find correlations between variables
- Identify missing data patterns and outliers
- Filter and transform data with natural language

### Visualization (10+ Chart Types)
- Line, scatter, bar, histogram, box plot
- Correlation heatmaps and pair plots
- Distribution plots
- Time series plots
- All charts are interactive (Plotly) and downloadable

### Data Cleaning
- **Automated analysis**: Get AI-powered cleaning recommendations
- **Missing data**: Impute with mean/median/mode or forward fill
- **Outliers**: Detect with Z-score/IQR and remove/cap
- **Duplicates**: Find and remove duplicate rows
- **Type conversion**: Fix incorrect column types

### Machine Learning
- **Linear Regression**: Predict continuous outcomes with full diagnostics
- **Logistic Regression**: Binary and multiclass classification
- **ARIMA**: Time series forecasting with automatic parameter selection
- **Model Evaluation**: Metrics, plots, confusion matrices, ROC curves
- **Feature Engineering**: Encode categorical variables for ML
- **Model Export**: Download trained models for deployment

### Statistical Tests
- T-tests (one-sample, two-sample, paired)
- Z-tests for proportions
- Chi-square tests for independence
- ANOVA for multiple groups

## 🎯 Who Should Use This?

- **Data Analysts** who want to speed up exploratory analysis
- **Product Managers** who need quick insights without coding
- **Students** learning statistics and machine learning
- **Researchers** doing reproducible analysis
- **Anyone** who has data and questions.

## 🔐 Getting Started

### Requirements
- Python 3.11 or higher
- OpenAI API key (GPT-4 access)

### Installation
```bash
git clone https://github.com/oogunbiyi21/ds-auto-insights.git
cd stats-compass
pip install -r requirements.txt
```

### Quick Start
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run the app
streamlit run app.py
```

Navigate to `http://localhost:8501` and start analyzing.

### Try It Out
1. Upload your own CSV/Excel file, or
2. Try with a sample dataset (like Titanic from seaborn/sklearn)
3. Ask: *"What factors predict survival? Build a model."*

## 🐛 Major Fixes in This Release

This release includes several critical bug fixes discovered during internal testing:

### ✅ Fixed: Random Model Failures
**Problem**: Same command like "Build logistic regression to predict Symbol" would succeed sometimes and fail other times.

**Solution**: Auto-feature selection now excludes high-missing columns (>20% missing) and validates class counts after data cleaning.

### ✅ Fixed: DataFrame Ambiguity Errors
**Problem**: Certain queries crashed with "The truth value of a DataFrame is ambiguous".

**Solution**: Replaced unsafe boolean operators with explicit null checks throughout the codebase.

### ✅ Fixed: Model Download Not Working
**Problem**: "No trained models available" error even after successfully training models.

**Solution**: Models now correctly saved to both storage locations for export functionality.

## 🎓 Learning Resources

- **Tool Documentation**: Each tool has detailed descriptions and prerequisites
- **Agent Logs**: Watch what the AI agent is doing in real-time
- **Example workflows**: Try prompts like "Show correlation matrix", "Build a regression model", "Clean missing data"

## 🚧 Known Limitations

- Requires GPT-4 API access (no local LLM support yet)
- Best performance with datasets under 100k rows
- Only tabular data (no images, text, or audio)
- No SQL database connectivity (yet!)

## 🔮 What's Coming Next?

We're already working on v1.1.0 with:
- **Stricter workflows** with optional enforcement mode
- **More ML algorithms**: Random Forest, XGBoost, K-means clustering
- **SQL support**: Query databases directly
- **Custom tools**: Create your own analysis tools
- **Better prompts**: Improved agent reasoning and tool selection

## 📣 Feedback & Support

Found a bug? Have a feature request? Want to contribute?

- **GitHub Issues**: [github.com/oogunbiyi21/ds-auto-insights/issues](https://github.com/oogunbiyi21/ds-auto-insights/issues)
- **Discussions**: [github.com/oogunbiyi21/ds-auto-insights/discussions](https://github.com/oogunbiyi21/ds-auto-insights/discussions)

## 🙏 Acknowledgments

Stats Compass is built on:
- **LangChain**: Agent framework and tool orchestration
- **OpenAI**: GPT-4 language model
- **Streamlit**: Beautiful web app framework
- **Plotly**: Interactive visualizations
- **Scikit-learn, Statsmodels, Pandas**: Data science backbone

## 📜 License

MIT License - See LICENSE file for details

---

**Ready to explore your data? Download v1.0.0 and start asking questions**

[Download Release](https://github.com/oogunbiyi21/ds-auto-insights/releases/tag/v1.0.0) | [Documentation](https://github.com/oogunbiyi21/ds-auto-insights#readme) | [Report Issues](https://github.com/oogunbiyi21/ds-auto-insights/issues)
