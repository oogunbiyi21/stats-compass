# Changelog

All notable changes to Stats Compass will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-29

### 🎉 Initial Public Release

Stats Compass v1.0.0 is the first stable public release of the AI-powered data analysis assistant.

### ✨ Core Features

#### 🤖 AI-Powered Analysis
- **Natural Language Interface**: Ask questions about your data in plain English
- **Autonomous Agent**: LangChain-based agent that selects and chains the right tools automatically
- **Smart Suggestions**: Context-aware recommendations for next analysis steps
- **Workflow Guidance**: Built-in best practices for data cleaning, EDA, and modeling

#### 📊 Data Exploration Tools
- **Flexible Data Query**: Inspect, filter, and modify data with natural language
- **Statistical Summaries**: Schema inspection, descriptive statistics, correlation analysis
- **Groupby Aggregations**: Simple aggregations (mean, sum, count, median, etc.)
- **Missing Data Analysis**: Comprehensive missing data patterns and recommendations
- **Outlier Detection**: Z-score and IQR-based outlier identification

#### 📈 Visualization
- **10+ Chart Types**: Line, scatter, bar, histogram, box plot, heatmap, and more
- **Smart Defaults**: Automatic chart type selection based on data types
- **Interactive Charts**: Plotly-based interactive visualizations
- **Chart Export**: Download charts as PNG images

#### 🧹 Data Cleaning
- **Automated Cleaning**: `suggest_data_cleaning` analyzes and proposes fixes
- **Missing Data Handling**: Imputation with mean/median/mode/forward fill
- **Outlier Treatment**: Remove or cap outliers based on statistical thresholds
- **Duplicate Removal**: Identify and remove duplicate rows
- **Type Conversion**: Fix column data types automatically

#### 🤖 Machine Learning
- **Linear Regression**: Full diagnostics, coefficient interpretation, residual analysis
- **Logistic Regression**: Binary and multiclass classification with probability predictions
- **ARIMA Time Series**: Automated time series forecasting with stationarity tests
- **Model Evaluation**: Comprehensive metrics, confusion matrices, ROC curves
- **Feature Engineering**: Mean target encoding, rare category binning
- **Smart ML Guidance**: Validates data quality, suggests preprocessing, identifies issues

#### 📥 Data Import/Export
- **Multiple Formats**: CSV, Excel, JSON, Parquet support
- **Sample Datasets**: 15+ built-in datasets for testing and learning
- **Model Export**: Download trained models as `.joblib` files
- **Chart Export**: Save visualizations for presentations

### 🛠️ Technical Architecture

#### Tool System
- **50+ Specialized Tools**: Each tool handles a specific analysis task
- **Tool Categories**: Exploration, Charts, ML, Cleaning, Statistical Tests
- **Smart Tool Selection**: Agent automatically picks the right tool sequence

#### State Management
- **Persistent Session**: DataFrame state preserved across interactions
- **Workflow Metadata**: Tracks data cleaning, encoding, and modeling steps
- **Model Storage**: Trained models stored in session for evaluation and export

#### Performance
- **Streaming Responses**: Real-time token streaming for better UX
- **Efficient Processing**: Pandas-based operations for speed
- **Error Handling**: Comprehensive error messages with actionable solutions

### 🔧 Major Bug Fixes

#### DataFrame Boolean Ambiguity (Oct 2025)
- **Fixed**: "The truth value of a DataFrame is ambiguous" error
- **Root Cause**: Using `or` operator with pandas DataFrames triggers boolean evaluation
- **Solution**: Replaced with explicit `is not None` checks in 7 tool locations
- **Impact**: Eliminated common crash when using `inspect_data` and related tools

#### Logistic Regression Class Elimination (Oct 2025)
- **Fixed**: Inconsistent behavior where same command succeeds/fails randomly
- **Root Cause**: Missing data removal eliminated entire classes, but validation only happened before cleaning
- **Solution**: Added post-cleaning class validation with detailed error messages identifying high-missing features
- **Impact**: Users now get clear, actionable errors instead of cryptic sklearn failures

#### Model Download Missing (Oct 2025)
- **Fixed**: "No trained models available" error even after training models
- **Root Cause**: Models saved to `ml_model_results` but export utility looked in `trained_models`
- **Solution**: Models now saved to both locations for backward compatibility
- **Impact**: Download functionality now works for all trained models

#### Auto-Feature Selection Non-Determinism (Oct 2025)
- **Fixed**: Same command producing different results (random feature inclusion)
- **Root Cause**: Auto-selection included ALL numeric columns, including high-missing ones
- **Solution**: Auto-selection now filters out features with >20% missing data
- **Impact**: Consistent, predictable behavior; prevents class elimination during modeling

### 📖 Documentation

#### Tool Descriptions
- **Prerequisites Warnings**: ML tools now clearly state data requirements
- **Usage Examples**: Better examples in tool descriptions
- **Error Messages**: Detailed errors with specific solutions

#### Code Quality
- **Type Hints**: Comprehensive type annotations
- **Docstrings**: Detailed documentation for all major functions
- **Comments**: Inline comments explaining complex logic

### 🔐 Authentication (Branches)
- **main**: No authentication (local development only)
- **public-beta**: Users provide their own OpenAI API key
- **private-beta**: Password-protected access using developer's API key

### 📦 Dependencies
- Python 3.11+
- OpenAI API (GPT-4 required)
- Streamlit 1.29+
- LangChain 0.1.0+
- Pandas, NumPy, Scikit-learn, Statsmodels
- Plotly for visualizations

### 🎯 Known Limitations
- Requires OpenAI GPT-4 API access (no local LLM support yet)
- Large datasets (>100k rows) may cause memory issues
- No support for SQL databases (CSV/Excel only)
- Limited to tabular data (no image/text/audio analysis)

### 🚀 What's Next (v1.1.0)
- Strict mode for ML tools (enforce preprocessing workflows)
- Additional ML algorithms (Random Forest, XGBoost)
- SQL database connectivity
- Custom tool creation interface
- Improved prompt engineering for better agent performance

---

## Release History

### [1.0.0] - 2025-10-29
- Initial public release

---

**Note**: Version 1.0.0 represents the first stable public release. Previous development versions (0.1.0) were internal only.
