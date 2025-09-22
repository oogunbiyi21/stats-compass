# stats_compass/tools/chart_tools.py
"""
Chart and visualization tools for DS Auto Insights.
Provides comprehensive charting capabilities using Plotly.
"""

from typing import Type
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool


class CreateBarChartInput(BaseModel):
    column: str = Field(description="Column name to create bar chart for")
    top_n: int = Field(default=10, description="Number of top categories to show")
    title: str = Field(default="", description="Custom title for the chart")


class CreateBarChartTool(BaseTool):
    name: str = "create_bar_chart"
    description: str = "Creates a bar chart for categorical data showing top categories by count."
    args_schema: Type[BaseModel] = CreateBarChartInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, column: str, top_n: int = 10, title: str = "") -> str:
        try:
            if column not in self._df.columns:
                return f"❌ Column '{column}' not found. Available columns: {list(self._df.columns)}"

            value_counts = self._df[column].value_counts().head(top_n)
            
            if len(value_counts) == 0:
                return f"❌ No data found in column '{column}'"

            # Create chart data
            chart_data = pd.DataFrame({
                'category': value_counts.index.astype(str),
                'count': value_counts.values
            })

            chart_title = title or f"Top {top_n} {column} by Count"
            
            result = f"📊 {chart_title}\n\n"
            result += f"Total unique values: {self._df[column].nunique()}\n"
            result += f"Showing top {len(value_counts)} categories:\n\n"
            
            for i, (cat, count) in enumerate(value_counts.items(), 1):
                percentage = (count / len(self._df)) * 100
                result += f"  {i}. {cat}: {count} ({percentage:.1f}%)\n"
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'bar',
                    'data': chart_data,
                    'column': column,
                    'title': chart_title
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "\nChart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating bar chart: {e}"

    def _arun(self, column: str, top_n: int = 10, title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateScatterPlotInput(BaseModel):
    x_column: str = Field(description="Column name for x-axis")
    y_column: str = Field(description="Column name for y-axis")
    color_column: str = Field(default="", description="Optional column for color coding")
    title: str = Field(default="", description="Custom title for the chart")


class CreateScatterPlotTool(BaseTool):
    name: str = "create_scatter_plot"
    description: str = "Creates a scatter plot to show relationship between two numeric columns."
    args_schema: Type[BaseModel] = CreateScatterPlotInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, x_column: str, y_column: str, color_column: str = "", title: str = "") -> str:
        try:
            # Validate columns
            if x_column not in self._df.columns:
                return f"❌ X column '{x_column}' not found. Available columns: {list(self._df.columns)}"
            if y_column not in self._df.columns:
                return f"❌ Y column '{y_column}' not found. Available columns: {list(self._df.columns)}"
            
            if color_column and color_column not in self._df.columns:
                return f"❌ Color column '{color_column}' not found. Available columns: {list(self._df.columns)}"

            # Check if columns are numeric
            if not pd.api.types.is_numeric_dtype(self._df[x_column]):
                return f"❌ X column '{x_column}' must be numeric"
            if not pd.api.types.is_numeric_dtype(self._df[y_column]):
                return f"❌ Y column '{y_column}' must be numeric"

            # Prepare data
            plot_data = self._df[[x_column, y_column]].dropna()
            if color_column:
                plot_data = self._df[[x_column, y_column, color_column]].dropna()
            
            if len(plot_data) == 0:
                return f"❌ No complete data pairs found for {x_column} and {y_column}"

            chart_title = title or f"{y_column} vs {x_column}"
            
            # Calculate correlation
            correlation = plot_data[x_column].corr(plot_data[y_column])
            
            result = f"📊 {chart_title}\n\n"
            result += f"Data points: {len(plot_data)}\n"
            result += f"Correlation: {correlation:.3f}\n"
            
            if abs(correlation) > 0.7:
                result += "🔍 Strong correlation detected!\n"
            elif abs(correlation) > 0.3:
                result += "📈 Moderate correlation detected.\n"
            else:
                result += "📊 Weak correlation.\n"
            
            result += f"\n{x_column} range: {plot_data[x_column].min():.2f} to {plot_data[x_column].max():.2f}\n"
            result += f"{y_column} range: {plot_data[y_column].min():.2f} to {plot_data[y_column].max():.2f}\n"

            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'scatter',
                    'data': plot_data,
                    'x_column': x_column,
                    'y_column': y_column,
                    'color_column': color_column if color_column else None,
                    'title': chart_title,
                    'correlation': correlation
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "\nChart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating scatter plot: {e}"

    def _arun(self, x_column: str, y_column: str, color_column: str = "", title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateLineChartInput(BaseModel):
    x_column: str = Field(description="Column name for x-axis (typically time/date)")
    y_column: str = Field(description="Column name for y-axis")
    title: str = Field(default="", description="Custom title for the chart")


class CreateLineChartTool(BaseTool):
    name: str = "create_line_chart"
    description: str = "Creates a line chart to show trends over time or ordered data."
    args_schema: Type[BaseModel] = CreateLineChartInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, x_column: str, y_column: str, title: str = "") -> str:
        try:
            # Validate columns
            if x_column not in self._df.columns:
                return f"❌ X column '{x_column}' not found. Available columns: {list(self._df.columns)}"
            if y_column not in self._df.columns:
                return f"❌ Y column '{y_column}' not found. Available columns: {list(self._df.columns)}"

            # Check if y column is numeric
            if not pd.api.types.is_numeric_dtype(self._df[y_column]):
                return f"❌ Y column '{y_column}' must be numeric"

            # Prepare data
            plot_data = self._df[[x_column, y_column]].dropna().sort_values(x_column)
            
            if len(plot_data) == 0:
                return f"❌ No complete data found for {x_column} and {y_column}"

            chart_title = title or f"{y_column} over {x_column}"
            
            result = f"📊 {chart_title}\n\n"
            result += f"Data points: {len(plot_data)}\n"
            result += f"{y_column} range: {plot_data[y_column].min():.2f} to {plot_data[y_column].max():.2f}\n"
            
            # Calculate trend
            if len(plot_data) > 1:
                first_val = plot_data[y_column].iloc[0]
                last_val = plot_data[y_column].iloc[-1]
                change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                
                if change > 5:
                    result += f"📈 Upward trend: +{change:.1f}%\n"
                elif change < -5:
                    result += f"📉 Downward trend: {change:.1f}%\n"
                else:
                    result += f"➡️ Relatively stable: {change:.1f}%\n"

            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'line',
                    'data': plot_data,
                    'x_column': x_column,
                    'y_column': y_column,
                    'title': chart_title
                }
                st.session_state.current_response_charts.append(chart_info)
            
            result += "\nChart data prepared for display. 📈"
            return result

        except Exception as e:
            return f"❌ Error creating line chart: {e}"

    def _arun(self, x_column: str, y_column: str, title: str = ""):
        raise NotImplementedError("Async not supported")


class TimeSeriesAnalysisInput(BaseModel):
    date_column: str = Field(description="Name of the date/time column")
    value_column: str = Field(description="Name of the numeric column to analyze over time")
    freq: str = Field(default="D", description="Frequency for resampling (D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly)")
    agg_method: str = Field(default="mean", description="Aggregation method (mean, sum, count, min, max)")


class TimeSeriesAnalysisTool(BaseTool):
    name: str = "time_series_analysis"
    description: str = "Analyze trends and patterns in time series data. Automatically handles date parsing and resampling."
    args_schema: Type[BaseModel] = TimeSeriesAnalysisInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, date_column: str, value_column: str, freq: str = "D", agg_method: str = "mean") -> str:
        try:
            # Validate columns exist
            if date_column not in self._df.columns:
                return f"❌ Date column '{date_column}' not found. Available columns: {list(self._df.columns)}"
            
            if value_column not in self._df.columns:
                return f"❌ Value column '{value_column}' not found. Available columns: {list(self._df.columns)}"
            
            # Create a copy for time series analysis
            ts_df = self._df[[date_column, value_column]].copy()
            
            # Convert date column to datetime
            ts_df[date_column] = pd.to_datetime(ts_df[date_column], errors='coerce')
            
            # Drop rows with invalid dates or missing values
            ts_df = ts_df.dropna()
            
            if len(ts_df) == 0:
                return f"❌ No valid date-value pairs found after cleaning"
            
            # Sort by date
            ts_df = ts_df.sort_values(date_column)
            
            # Set date as index for resampling
            ts_df.set_index(date_column, inplace=True)
            
            # Resample based on frequency
            if agg_method == "mean":
                resampled = ts_df[value_column].resample(freq).mean()
            elif agg_method == "sum":
                resampled = ts_df[value_column].resample(freq).sum()
            elif agg_method == "count":
                resampled = ts_df[value_column].resample(freq).count()
            elif agg_method == "min":
                resampled = ts_df[value_column].resample(freq).min()
            elif agg_method == "max":
                resampled = ts_df[value_column].resample(freq).max()
            else:
                return f"❌ Invalid aggregation method '{agg_method}'. Use: mean, sum, count, min, max"
            
            # Remove NaN values from resampling
            resampled = resampled.dropna()
            
            if len(resampled) == 0:
                return f"❌ No data after resampling with frequency '{freq}'"
            
            # Calculate basic time series statistics
            trend_change = resampled.iloc[-1] - resampled.iloc[0] if len(resampled) > 1 else 0
            trend_pct = (trend_change / resampled.iloc[0] * 100) if resampled.iloc[0] != 0 else 0
            
            # Find peaks and troughs
            max_value = resampled.max()
            min_value = resampled.min()
            max_date = resampled.idxmax()
            min_date = resampled.idxmin()
            
            # Create chart data
            chart_data = pd.DataFrame({
                'Date': resampled.index,
                'Value': resampled.values
            })
            
            # Store chart info for export and persistence
            chart_info = {
                'type': 'time_series',
                'title': f"{value_column} over Time ({agg_method.title()})",
                'data': chart_data,
                'x_column': 'Date',
                'y_column': 'Value',
                'variable_name': value_column,
                'chart_config': {
                    'chart_type': 'line',
                    'x_col': 'Date',
                    'y_col': 'Value',
                    'line_width': 3,
                    'ylabel': value_column
                }
            }
            
            # Add to current response charts for persistence
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                st.session_state.current_response_charts.append(chart_info)
            
            # Summary statistics
            summary = f"""📈 Time Series Analysis: {value_column} over {date_column}

📊 Data Summary:
  • Time period: {resampled.index.min().strftime('%Y-%m-%d')} to {resampled.index.max().strftime('%Y-%m-%d')}
  • Frequency: {freq} ({agg_method})
  • Data points: {len(resampled)}

📈 Trend Analysis:
  • Overall change: {trend_change:.2f} ({trend_pct:+.1f}%)
  • Start value: {resampled.iloc[0]:.2f}
  • End value: {resampled.iloc[-1]:.2f}

🔍 Key Statistics:
  • Maximum: {max_value:.2f} on {max_date.strftime('%Y-%m-%d')}
  • Minimum: {min_value:.2f} on {min_date.strftime('%Y-%m-%d')}
  • Mean: {resampled.mean():.2f}
  • Std Dev: {resampled.std():.2f}

Chart data prepared for display. 📈"""
            
            return summary
            
        except Exception as e:
            return f"❌ Error in time series analysis: {str(e)}"

    def _arun(self, date_column: str, value_column: str, freq: str = "D", agg_method: str = "mean"):
        raise NotImplementedError("Async not supported")


class CreateColumnInput(BaseModel):
    column_name: str = Field(description="Name of the new column to create")
    operation: str = Field(description="The pandas operation to create the column. Examples: 'df[\"goals\"] * 2', 'df[\"goals\"].apply(lambda x: \"High\" if x > 10 else \"Low\")', 'df[\"h_team\"] + \" vs \" + df[\"a_team\"]'")
    description: str = Field(default="", description="Description of what this column represents")


class CreateColumnTool(BaseTool):
    name: str = "create_column"
    description: str = """Create a new column in the dataset using pandas operations.
    This tool allows you to add calculated columns, conditional columns, or transform existing data.
    Use this for complex data transformations that require creating new variables."""
    args_schema: Type[BaseModel] = CreateColumnInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _run(self, column_name: str, operation: str, description: str = "") -> str:
        try:
            # Safety checks
            if column_name in self._df.columns:
                return f"❌ Column '{column_name}' already exists. Choose a different name or use a different operation."
            
            # Check for dangerous operations
            dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file', 'system', 'os.', 'subprocess', 'globals', 'locals']
            if any(keyword in operation.lower() for keyword in dangerous_keywords):
                return f"❌ Operation contains potentially dangerous keywords. Please use only pandas operations."
            
            # Ensure operation is a valid pandas expression
            if not (operation.strip().startswith('df[') or operation.strip().startswith('df.') or 
                   any(func in operation for func in ['np.', 'pd.', 'lambda', '"', "'"])):
                return f"❌ Operation must start with 'df[' or 'df.' or use allowed functions (np., pd., lambda). Got: {operation}"
            
            # Create a safe namespace for evaluation
            safe_dict = {
                'df': self._df,
                'pd': pd,
                'np': np,
                '__builtins__': {'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool, 'max': max, 'min': min}
            }
            
            # Execute the operation
            try:
                result = eval(operation, safe_dict)
                
                # Validate result is appropriate for a column
                if hasattr(result, '__len__') and len(result) != len(self._df):
                    return f"❌ Operation result length ({len(result)}) doesn't match dataframe length ({len(self._df)})"
                
                # Add the new column
                self._df[column_name] = result
                
                # Update the dataframe in session state if available
                if hasattr(st, 'session_state') and 'uploaded_df' in st.session_state:
                    st.session_state.uploaded_df = self._df
                
                # Show preview of new column
                preview = self._df[[column_name]].head(10)
                
                success_msg = f"✅ Created column '{column_name}'"
                if description:
                    success_msg += f" - {description}"
                
                success_msg += f"\n\nPreview of new column (first 10 rows):\n{preview.to_string()}"
                
                # Show basic stats if numeric
                if pd.api.types.is_numeric_dtype(self._df[column_name]):
                    stats = self._df[column_name].describe()
                    success_msg += f"\n\nColumn statistics:\n{stats.to_string()}"
                else:
                    # Show value counts for categorical
                    value_counts = self._df[column_name].value_counts().head(5)
                    success_msg += f"\n\nTop 5 values:\n{value_counts.to_string()}"
                
                return success_msg
                
            except Exception as eval_error:
                return f"❌ Error executing operation: {str(eval_error)}. Please check your pandas syntax."
            
        except Exception as e:
            return f"❌ Error creating column: {str(e)}"
    
    def _arun(self, column_name: str, operation: str, description: str = ""):
        raise NotImplementedError("Async not supported")


# Create Correlation Heatmap Tool
class CreateCorrelationHeatmapInput(BaseModel):
    columns: list | None = Field(default=None, description="List of numeric columns to include. If None, uses all numeric columns.")
    method: str = Field(default="pearson", description="Correlation method: pearson, kendall, or spearman")
    title: str = Field(default="Correlation Heatmap", description="Chart title")


class CreateCorrelationHeatmapTool(BaseTool):
    name: str = "create_correlation_heatmap"
    description: str = "Create a visual correlation heatmap showing relationships between numeric variables."
    args_schema: Type[BaseModel] = CreateCorrelationHeatmapInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, columns: list = None, method: str = "pearson", title: str = "Correlation Heatmap") -> str:
        try:
            import streamlit as st
            import plotly.express as px
            import plotly.graph_objects as go
            import numpy as np
            
            # Get numeric columns
            numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return f"❌ Need at least 2 numeric columns for correlation analysis. Found: {len(numeric_cols)}"
            
            # Performance optimization: limit columns for large datasets
            if len(numeric_cols) > 20:
                return f"⚠️ Too many numeric columns ({len(numeric_cols)}) for heatmap visualization. Please specify up to 20 columns using the 'columns' parameter."
            
            # Use specified columns or all numeric columns
            if columns:
                # Validate specified columns
                invalid_cols = [col for col in columns if col not in numeric_cols]
                if invalid_cols:
                    return f"❌ Non-numeric or missing columns: {invalid_cols}. Numeric columns: {numeric_cols}"
                cols_to_use = columns
            else:
                cols_to_use = numeric_cols
            
            # Calculate correlation matrix
            corr_data = self._df[cols_to_use]
            
            if method == "pearson":
                corr_matrix = corr_data.corr(method='pearson')
            elif method == "kendall":
                corr_matrix = corr_data.corr(method='kendall')
            elif method == "spearman":
                corr_matrix = corr_data.corr(method='spearman')
            else:
                return f"❌ Invalid correlation method '{method}'. Use: pearson, kendall, or spearman"
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title=title,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1
            )
            
            # Enhance the heatmap
            fig.update_traces(
                texttemplate="%{text:.2f}",
                textfont_size=10,
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            )
            
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=16)),
                height=max(400, len(cols_to_use) * 40),  # Dynamic height based on number of variables
                width=max(400, len(cols_to_use) * 40)
            )
            
            # Store chart info for export and persistence  
            chart_info = {
                'type': 'correlation_heatmap',
                'title': title,
                'data': corr_matrix.reset_index(),  # Convert to DataFrame for storage
                'correlation_matrix': corr_matrix.to_dict(),  # Store as dict for JSON serialization
                'method': method,
                'columns': cols_to_use,
                # Store chart configuration for recreation
                'chart_config': {
                    'chart_type': 'heatmap',
                    'color_scale': 'RdBu_r',
                    'zmin': -1,
                    'zmax': 1,
                    'show_text': True
                }
            }
            
            # Add to current response charts for persistence
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                st.session_state.current_response_charts.append(chart_info)
            
            # Find strongest correlations
            # Get upper triangle of correlation matrix (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Show top correlations
            top_correlations = ""
            for i, pair in enumerate(corr_pairs[:5]):
                strength = "Very Strong" if abs(pair['correlation']) > 0.8 else \
                          "Strong" if abs(pair['correlation']) > 0.6 else \
                          "Moderate" if abs(pair['correlation']) > 0.4 else \
                          "Weak" if abs(pair['correlation']) > 0.2 else "Very Weak"
                
                direction = "Positive" if pair['correlation'] > 0 else "Negative"
                
                top_correlations += f"  {i+1}. {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f} ({direction}, {strength})\n"
            
            summary = f"""🔥 {title}

📊 Created correlation heatmap with {len(cols_to_use)} variables using {method} method.

🔍 Strongest correlations:
{top_correlations.strip()}

💡 The heatmap shows relationships between variables - blue indicates positive correlation, red indicates negative correlation."""

            return summary
            
        except Exception as e:
            return f"❌ Error creating correlation heatmap: {str(e)}"

    def _arun(self, columns: list = None, method: str = "pearson", title: str = "Correlation Heatmap"):
        raise NotImplementedError("Async not supported")


# ===== ML REGRESSION CHART TOOLS =====

class CreateRegressionPlotInput(BaseModel):
    model_key: str = Field(default="linear_regression", description="Key of the ML model results to visualize")
    title: str = Field(default="", description="Custom title for the chart")


class CreateRegressionPlotTool(BaseTool):
    name: str = "create_regression_plot"
    description: str = "Creates actual vs predicted scatter plot from ML regression results."
    args_schema: Type[BaseModel] = CreateRegressionPlotInput

    def _run(self, model_key: str = "linear_regression", title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a regression analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            y_train = results['y_train']
            y_train_pred = results['y_train_pred']
            y_test = results['y_test']
            y_test_pred = results['y_test_pred']
            target_column = results['target_column']
            train_r2 = results.get('train_r2', 0)
            test_r2 = results.get('test_r2', 0)
            
            # Create scatter plot
            fig = go.Figure()
            
            # Training data
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers',
                name='Training Data',
                marker=dict(color='blue', size=6, opacity=0.6),
                hovertemplate='<b>Training</b><br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
            ))
            
            # Test data (if different from training)
            if len(y_test) != len(y_train):
                fig.add_trace(go.Scatter(
                    x=y_test, y=y_test_pred,
                    mode='markers',
                    name='Test Data',
                    marker=dict(color='red', size=6, opacity=0.6),
                    hovertemplate='<b>Test</b><br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
                ))
            
            # Perfect prediction line
            min_val = min(min(y_train), min(y_train_pred))
            max_val = max(max(y_train), max(y_train_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash'),
                hoverinfo='skip'
            ))
            
            chart_title = title or f"Actual vs Predicted: {target_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title=f'Actual {target_column}',
                yaxis_title=f'Predicted {target_column}',
                hovermode='closest'
            )
            
            # Store chart data for Streamlit rendering (following the same pattern as other chart tools)
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'regression_plot',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'train_actual': y_train.tolist() if hasattr(y_train, 'tolist') else y_train,
                        'train_predicted': y_train_pred.tolist() if hasattr(y_train_pred, 'tolist') else y_train_pred,
                        'test_actual': y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
                        'test_predicted': y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else y_test_pred
                    },
                    'target_column': target_column,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
                st.session_state.current_response_charts.append(chart_info)
            
            summary = f"""🎯 {chart_title}

📊 Created regression scatter plot showing model predictions vs actual values.

📈 Model Performance:
  • Training R² = {train_r2:.3f} ({train_r2*100:.1f}% variance explained)
  • Test R² = {test_r2:.3f} ({test_r2*100:.1f}% variance explained)

💡 Points closer to the diagonal line indicate better predictions. Systematic deviations from the line suggest model issues."""

            return summary
            
        except Exception as e:
            return f"❌ Error creating regression plot: {str(e)}"

    def _arun(self, model_key: str = "linear_regression", title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateResidualPlotInput(BaseModel):
    model_key: str = Field(default="linear_regression", description="Key of the ML model results to visualize")
    title: str = Field(default="", description="Custom title for the chart")


class CreateResidualPlotTool(BaseTool):
    name: str = "create_residual_plot"
    description: str = "Creates residual plot from ML regression results to check model assumptions."
    args_schema: Type[BaseModel] = CreateResidualPlotInput

    def _run(self, model_key: str = "linear_regression", title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a regression analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            y_train_pred = results['y_train_pred']
            residuals = results['residuals']
            target_column = results['target_column']
            assumptions = results.get('assumptions', {})
            
            # Create residual plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=y_train_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue', size=6, opacity=0.6),
                hovertemplate='<b>Residual Analysis</b><br>Predicted: %{x}<br>Residual: %{y}<extra></extra>'
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Zero Line")
            
            chart_title = title or f"Residual Plot: {target_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title=f'Predicted {target_column}',
                yaxis_title='Residuals',
                hovermode='closest'
            )
            
            # Store chart data for Streamlit rendering (following the same pattern as other chart tools)
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'residual_plot',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'predicted': y_train_pred.tolist() if hasattr(y_train_pred, 'tolist') else y_train_pred,
                        'residuals': residuals.tolist() if hasattr(residuals, 'tolist') else residuals
                    },
                    'target_column': target_column,
                    'assumptions': assumptions
                }
                st.session_state.current_response_charts.append(chart_info)
            
            # Analyze residual patterns
            pattern_analysis = ""
            if 'homoscedasticity' in assumptions:
                homo_status = assumptions['homoscedasticity']['status']
                pattern_analysis += f"  • Homoscedasticity (constant variance): {homo_status}\n"
            
            if 'normality' in assumptions:
                norm_status = assumptions['normality']['status']
                pattern_analysis += f"  • Normality of residuals: {norm_status}\n"
            
            summary = f"""🔍 {chart_title}

📊 Created residual plot to validate model assumptions.

⚠️ Model Assumptions Check:
{pattern_analysis.strip()}

💡 Good residuals should:
  • Be randomly scattered around zero line
  • Show constant variance (no funnel pattern)
  • Have no obvious patterns or trends"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating residual plot: {str(e)}"

    def _arun(self, model_key: str = "linear_regression", title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateCoefficientChartInput(BaseModel):
    model_key: str = Field(default="linear_regression", description="Key of the ML model results to visualize")
    top_n: int = Field(default=10, description="Number of top features to show")
    title: str = Field(default="", description="Custom title for the chart")


class CreateCoefficientChartTool(BaseTool):
    name: str = "create_coefficient_chart"
    description: str = "Creates coefficient importance chart from ML regression results."
    args_schema: Type[BaseModel] = CreateCoefficientChartInput

    def _run(self, model_key: str = "linear_regression", top_n: int = 10, title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a regression analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            coefficients = results['coefficients']
            target_column = results['target_column']
            standardized = results.get('standardized', False)
            
            # Sort by absolute coefficient value and take top N
            coef_sorted = coefficients.sort_values('abs_coefficient', ascending=True).tail(top_n)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Color bars based on positive/negative coefficients
            colors = ['green' if x > 0 else 'red' for x in coef_sorted['coefficient']]
            
            fig.add_trace(go.Bar(
                y=coef_sorted['feature'],
                x=coef_sorted['coefficient'],
                orientation='h',
                marker=dict(color=colors, opacity=0.7),
                hovertemplate='<b>%{y}</b><br>Coefficient: %{x}<br>95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>',
                customdata=coef_sorted[['conf_int_lower', 'conf_int_upper']].values
            ))
            
            chart_title = title or f"Feature Coefficients: {target_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title='Coefficient Value',
                yaxis_title='Features',
                hovermode='closest'
            )
            
            # Store chart data for Streamlit rendering (following the same pattern as other chart tools)
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'coefficient_chart',
                    'title': chart_title,
                    'figure': fig,
                    'data': coef_sorted.to_dict('records') if hasattr(coef_sorted, 'to_dict') else coef_sorted,
                    'target_column': target_column,
                    'standardized': standardized,
                    'top_n': top_n
                }
                st.session_state.current_response_charts.append(chart_info)
            
            # Create interpretation
            top_features = coef_sorted.tail(3)  # Most important 3
            feature_interpretation = ""
            for _, row in top_features.iterrows():
                direction = "increases" if row['coefficient'] > 0 else "decreases"
                significance = "✓ Significant" if row['significant'] else "Not significant"
                
                if standardized:
                    feature_interpretation += f"  • **{row['feature']}**: 1 std dev increase {direction} {target_column} by {abs(row['coefficient']):.3f} units ({significance})\n"
                else:
                    feature_interpretation += f"  • **{row['feature']}**: 1 unit increase {direction} {target_column} by {abs(row['coefficient']):.3f} units ({significance})\n"
            
            summary = f"""📊 {chart_title}

🎯 Created coefficient importance chart showing top {min(top_n, len(coefficients))} features.

🔍 Key Feature Effects:
{feature_interpretation.strip()}

💡 Interpretation:
  • Green bars = positive effect on {target_column}
  • Red bars = negative effect on {target_column}
  • Longer bars = stronger effect
  • Focus on significant features (✓) for decisions"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating coefficient chart: {str(e)}"

    def _arun(self, model_key: str = "linear_regression", top_n: int = 10, title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateFeatureImportanceChartInput(BaseModel):
    model_key: str = Field(default="logistic_regression", description="Key of the ML model results to visualize")
    top_n: int = Field(default=10, description="Number of top features to show")
    show_odds_ratios: bool = Field(default=True, description="Show odds ratios instead of raw coefficients for logistic regression")
    title: str = Field(default="", description="Custom title for the chart")


class CreateFeatureImportanceChartTool(BaseTool):
    name: str = "create_feature_importance_chart"
    description: str = "Creates feature importance chart from ML model results. Supports both linear and logistic regression with appropriate visualizations."
    args_schema: Type[BaseModel] = CreateFeatureImportanceChartInput

    def _run(self, model_key: str = "logistic_regression", top_n: int = 10, show_odds_ratios: bool = True, title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a regression analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            model_type = results['model_type']
            coefficients = results['coefficients']
            target_column = results['target_column']
            
            # Handle different model types
            if model_type == 'logistic_regression':
                # For logistic regression, we can show either coefficients or odds ratios
                if show_odds_ratios and 'odds_ratio' in coefficients.columns:
                    # Sort by absolute coefficient value to get most important features
                    chart_data = coefficients.sort_values('abs_coefficient', ascending=True).tail(top_n)
                    
                    # Create horizontal bar chart showing coefficients (not odds ratios) for better visualization
                    fig = go.Figure()
                    
                    # Color bars based on positive/negative coefficients  
                    colors = ['green' if x > 0 else 'red' for x in chart_data['coefficient']]
                    
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],  # Use coefficient values so negatives go left
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<br>Odds Ratio: %{customdata[0]:.3f}<extra></extra>',
                        customdata=chart_data[['odds_ratio']].values
                    ))
                    
                    chart_title = title or f"Feature Importance (Odds Ratios): {target_column}"
                    
                    # Calculate symmetric range around zero for better visualization
                    max_abs_coef = chart_data['coefficient'].abs().max()
                    x_range = [-max_abs_coef * 1.1, max_abs_coef * 1.1]
                    
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title='Coefficient Value (Log Odds)',
                        yaxis_title='Features',
                        hovermode='closest',
                        xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='black')
                    )
                    
                    # Add vertical line at zero (no effect)
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.8)
                    
                    value_column = 'coefficient'
                    interpretation_type = "odds ratios"
                    
                else:
                    # Show raw coefficients for logistic regression
                    chart_data = coefficients.sort_values('abs_coefficient', ascending=True).tail(top_n)
                    
                    # Create horizontal bar chart
                    fig = go.Figure()
                    
                    # Color bars based on positive/negative coefficients
                    colors = ['green' if x > 0 else 'red' for x in chart_data['coefficient']]
                    
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<br>Odds Ratio: %{customdata[0]:.3f}<extra></extra>',
                        customdata=chart_data[['odds_ratio']].values if 'odds_ratio' in chart_data.columns else [[0]] * len(chart_data)
                    ))
                    
                    chart_title = title or f"Feature Importance (Coefficients): {target_column}"
                    
                    # Calculate symmetric range around zero for better visualization
                    max_abs_coef = chart_data['coefficient'].abs().max()
                    x_range = [-max_abs_coef * 1.1, max_abs_coef * 1.1]
                    
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title='Coefficient Value',
                        yaxis_title='Features',
                        hovermode='closest',
                        xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='black')
                    )
                    
                    # Add vertical line at zero for reference
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.8)
                    
                    value_column = 'coefficient'
                    interpretation_type = "coefficients"
                    
            elif model_type == 'linear_regression':
                # For linear regression, show coefficients
                chart_data = coefficients.sort_values('abs_coefficient', ascending=True).tail(top_n)
                
                # Create horizontal bar chart
                fig = go.Figure()
                
                # Color bars based on positive/negative coefficients
                colors = ['green' if x > 0 else 'red' for x in chart_data['coefficient']]
                
                # Check if we have confidence intervals
                has_confidence_intervals = 'conf_int_lower' in chart_data.columns and 'conf_int_upper' in chart_data.columns
                
                if has_confidence_intervals:
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<br>95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>',
                        customdata=chart_data[['conf_int_lower', 'conf_int_upper']].values
                    ))
                else:
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>'
                    ))
                
                chart_title = title or f"Feature Importance: {target_column}"
                
                # Calculate symmetric range around zero for better visualization
                max_abs_coef = chart_data['coefficient'].abs().max()
                x_range = [-max_abs_coef * 1.1, max_abs_coef * 1.1]
                
                fig.update_layout(
                    title=chart_title,
                    xaxis_title='Coefficient Value',
                    yaxis_title='Features',
                    hovermode='closest',
                    xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='black')
                )
                
                # Add vertical line at zero for reference
                fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.8)
                
                value_column = 'coefficient'
                interpretation_type = "coefficients"
                
            else:
                return f"❌ Unsupported model type: {model_type}. Supported types: linear_regression, logistic_regression"
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'feature_importance_chart',
                    'title': chart_title,
                    'figure': fig,
                    'data': chart_data.to_dict('records') if hasattr(chart_data, 'to_dict') else chart_data,
                    'target_column': target_column,
                    'model_type': model_type,
                    'interpretation_type': interpretation_type,
                    'top_n': top_n
                }
                st.session_state.current_response_charts.append(chart_info)
            
            # Create more actionable interpretation
            top_features = chart_data.tail(3)  # Most important 3
            feature_interpretation = ""
            recommendations = ""
            
            if model_type == 'logistic_regression' and show_odds_ratios and 'odds_ratio' in coefficients.columns:
                # Get the most impactful feature for recommendations
                most_impactful = chart_data.tail(1).iloc[0]
                most_impactful_odds = most_impactful['odds_ratio']
                
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    odds_ratio = row['odds_ratio']
                    if odds_ratio > 1:
                        effect = f"increases odds by {((odds_ratio - 1) * 100):.1f}%"
                        action = "increase" if odds_ratio > 1.2 else "consider increasing"
                    else:
                        effect = f"decreases odds by {((1 - odds_ratio) * 100):.1f}%"
                        action = "decrease" if odds_ratio < 0.8 else "consider decreasing"
                    
                    feature_interpretation += f"  • **{row['feature']}**: 1 unit increase {effect} (OR: {odds_ratio:.3f})\n"
                
                # Add specific recommendation
                if most_impactful_odds > 1.5:
                    recommendations = f"\n🎯 **Primary Recommendation**: Focus on maximizing {most_impactful['feature']} - it has the strongest positive impact (OR: {most_impactful_odds:.3f})."
                elif most_impactful_odds < 0.67:
                    recommendations = f"\n🎯 **Primary Recommendation**: Focus on minimizing {most_impactful['feature']} - it has the strongest negative impact (OR: {most_impactful_odds:.3f})."
                else:
                    recommendations = f"\n🎯 **Primary Recommendation**: {most_impactful['feature']} shows the strongest effect. Consider strategic adjustments."
                    
            else:
                # For linear regression or coefficient view
                most_impactful = chart_data.tail(1).iloc[0]
                most_impactful_coef = most_impactful['coefficient']
                
                for _, row in top_features.iterrows():
                    direction = "increases" if row['coefficient'] > 0 else "decreases"
                    significance = ""
                    if 'significant' in row and row['significant']:
                        significance = " (✓ Significant)"
                    elif 'significant' in row:
                        significance = " (Not significant)"
                    
                    feature_interpretation += f"  • **{row['feature']}**: 1 unit increase {direction} {target_column} by {abs(row['coefficient']):.3f} units{significance}\n"
                
                # Add specific recommendation with coefficient comparison
                if len(chart_data) > 1:
                    second_most = chart_data.tail(2).iloc[0]
                    ratio = abs(most_impactful_coef) / abs(second_most['coefficient'])
                    action = "increase" if most_impactful_coef > 0 else "decrease"
                    
                    recommendations = f"\n🎯 **Primary Recommendation**: Prioritize {most_impactful['feature']} - it has {ratio:.1f}x more impact than {second_most['feature']}. Focus on {action}ing {most_impactful['feature']} for maximum {target_column} improvement."
                else:
                    action = "increase" if most_impactful_coef > 0 else "decrease"
                    recommendations = f"\n🎯 **Primary Recommendation**: Focus on {action}ing {most_impactful['feature']} for maximum {target_column} impact."
            
            chart_type_desc = "Odds Ratios" if (model_type == 'logistic_regression' and show_odds_ratios) else "Coefficients"
            
            summary = f"""📊 {chart_title}

🎯 Created feature importance chart showing top {min(top_n, len(coefficients))} features using {interpretation_type}.

🔍 Key Feature Effects:
{feature_interpretation.strip()}{recommendations}

💡 Chart Interpretation:
"""
            
            if model_type == 'logistic_regression' and show_odds_ratios:
                summary += """  • Green bars = increases odds of positive outcome
  • Red bars = decreases odds of positive outcome
  • Values > 1 = increases probability, < 1 = decreases probability
  • Distance from 1 indicates strength of effect"""
            else:
                summary += f"""  • Green bars = positive effect on {target_column}
  • Red bars = negative effect on {target_column}
  • Longer bars = stronger effect"""
                if model_type == 'linear_regression':
                    summary += "\n  • Focus on significant features (✓) for decisions"

            return summary
            
        except Exception as e:
            return f"❌ Error creating feature importance chart: {str(e)}"

    def _arun(self, model_key: str = "logistic_regression", top_n: int = 10, show_odds_ratios: bool = True, title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateROCCurveInput(BaseModel):
    model_key: str = Field(default="logistic_regression", description="Key of the classification model results to visualize")
    title: str = Field(default="", description="Custom title for the chart")


class CreateROCCurveTool(BaseTool):
    """
    Create ROC (Receiver Operating Characteristic) curve for binary classification models.
    Shows true positive rate vs false positive rate at various threshold settings.
    """
    
    name: str = "create_roc_curve"
    description: str = "Create ROC curve visualization for binary classification models showing model discrimination ability"
    args_schema: Type[BaseModel] = CreateROCCurveInput

    def _run(self, model_key: str = "logistic_regression", title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a classification analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            
            # Check if this is a classification model
            if 'y_train_proba' not in results:
                return f"❌ Model '{model_key}' does not appear to be a classification model. ROC curves require probability predictions."
            
            # Extract data
            y_train = results['y_train']
            y_train_proba = results['y_train_proba']
            y_test = results['y_test']
            y_test_proba = results['y_test_proba']
            target_column = results['target_column']
            
            # Import sklearn metrics
            from sklearn.metrics import roc_curve, auc
            
            # Calculate ROC curves
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
            roc_auc_train = auc(fpr_train, tpr_train)
            
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
            roc_auc_test = auc(fpr_test, tpr_test)
            
            # Create ROC curve plot
            fig = go.Figure()
            
            # Training ROC curve
            fig.add_trace(go.Scatter(
                x=fpr_train, y=tpr_train,
                mode='lines',
                name=f'Training ROC (AUC = {roc_auc_train:.3f})',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Training</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
            
            # Test ROC curve (if different from training)
            if len(y_test) != len(y_train) or not np.array_equal(y_test, y_train):
                fig.add_trace(go.Scatter(
                    x=fpr_test, y=tpr_test,
                    mode='lines',
                    name=f'Test ROC (AUC = {roc_auc_test:.3f})',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Test</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                ))
            
            # Random classifier line (diagonal)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier (AUC = 0.5)',
                line=dict(color='gray', dash='dash', width=1),
                hoverinfo='skip'
            ))
            
            chart_title = title or f"ROC Curve: {target_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title='False Positive Rate (1 - Specificity)',
                yaxis_title='True Positive Rate (Sensitivity)',
                hovermode='closest',
                width=600,
                height=500,
                showlegend=True
            )
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'roc_curve',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'train_fpr': fpr_train.tolist(),
                        'train_tpr': tpr_train.tolist(),
                        'train_auc': roc_auc_train,
                        'test_fpr': fpr_test.tolist(),
                        'test_tpr': tpr_test.tolist(),
                        'test_auc': roc_auc_test
                    },
                    'target_column': target_column,
                    'model_key': model_key
                }
                st.session_state.current_response_charts.append(chart_info)
            
            summary = f"""📈 {chart_title}

📊 Created ROC curve showing model's discrimination ability across all classification thresholds.

📈 Model Performance:
  • Training AUC = {roc_auc_train:.3f} ({self._interpret_auc(roc_auc_train)})
  • Test AUC = {roc_auc_test:.3f} ({self._interpret_auc(roc_auc_test)})

💡 AUC Interpretation:
  • AUC = 1.0: Perfect classifier
  • AUC = 0.5: Random classifier (diagonal line)
  • AUC > 0.8: Excellent performance
  • AUC 0.7-0.8: Good performance
  • AUC 0.6-0.7: Fair performance"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating ROC curve: {str(e)}"

    def _interpret_auc(self, auc_score):
        """Interpret AUC score"""
        if auc_score >= 0.9:
            return "Excellent"
        elif auc_score >= 0.8:
            return "Good"
        elif auc_score >= 0.7:
            return "Fair"
        elif auc_score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"

    def _arun(self, model_key: str = "logistic_regression", title: str = ""):
        raise NotImplementedError("Async not supported")


class CreatePrecisionRecallCurveInput(BaseModel):
    model_key: str = Field(default="logistic_regression", description="Key of the classification model results to visualize")
    title: str = Field(default="", description="Custom title for the chart")


class CreatePrecisionRecallCurveTool(BaseTool):
    """
    Create Precision-Recall curve for binary classification models.
    Useful for imbalanced datasets where positive class is rare.
    """
    
    name: str = "create_precision_recall_curve"
    description: str = "Create precision-recall curve for binary classification, especially useful for imbalanced datasets"
    args_schema: Type[BaseModel] = CreatePrecisionRecallCurveInput

    def _run(self, model_key: str = "logistic_regression", title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a classification analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            
            # Check if this is a classification model
            if 'y_train_proba' not in results:
                return f"❌ Model '{model_key}' does not appear to be a classification model. PR curves require probability predictions."
            
            # Extract data
            y_train = results['y_train']
            y_train_proba = results['y_train_proba']
            y_test = results['y_test']
            y_test_proba = results['y_test_proba']
            target_column = results['target_column']
            
            # Import sklearn metrics
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            # Calculate Precision-Recall curves
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
            ap_score_train = average_precision_score(y_train, y_train_proba)
            
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
            ap_score_test = average_precision_score(y_test, y_test_proba)
            
            # Create Precision-Recall curve plot
            fig = go.Figure()
            
            # Training PR curve
            fig.add_trace(go.Scatter(
                x=recall_train, y=precision_train,
                mode='lines',
                name=f'Training PR (AP = {ap_score_train:.3f})',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Training</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
            ))
            
            # Test PR curve (if different from training)
            if len(y_test) != len(y_train) or not np.array_equal(y_test, y_train):
                fig.add_trace(go.Scatter(
                    x=recall_test, y=precision_test,
                    mode='lines',
                    name=f'Test PR (AP = {ap_score_test:.3f})',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Test</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
                ))
            
            # Baseline (random classifier for imbalanced data)
            positive_ratio = np.mean(y_train)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[positive_ratio, positive_ratio],
                mode='lines',
                name=f'Random Classifier (AP = {positive_ratio:.3f})',
                line=dict(color='gray', dash='dash', width=1),
                hoverinfo='skip'
            ))
            
            chart_title = title or f"Precision-Recall Curve: {target_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title='Recall (Sensitivity)',
                yaxis_title='Precision',
                hovermode='closest',
                width=600,
                height=500,
                showlegend=True
            )
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'precision_recall_curve',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'train_precision': precision_train.tolist(),
                        'train_recall': recall_train.tolist(),
                        'train_ap': ap_score_train,
                        'test_precision': precision_test.tolist(),
                        'test_recall': recall_test.tolist(),
                        'test_ap': ap_score_test,
                        'positive_ratio': positive_ratio
                    },
                    'target_column': target_column,
                    'model_key': model_key
                }
                st.session_state.current_response_charts.append(chart_info)
            
            summary = f"""📈 {chart_title}

📊 Created Precision-Recall curve showing trade-off between precision and recall.

📈 Model Performance:
  • Training AP = {ap_score_train:.3f} ({self._interpret_ap(ap_score_train, positive_ratio)})
  • Test AP = {ap_score_test:.3f} ({self._interpret_ap(ap_score_test, positive_ratio)})

💡 Interpretation:
  • Average Precision (AP) summarizes the curve
  • Higher AP = better performance
  • Baseline (random) = {positive_ratio:.3f} (proportion of positive class)
  • PR curves are better than ROC for imbalanced datasets"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating precision-recall curve: {str(e)}"

    def _interpret_ap(self, ap_score, baseline):
        """Interpret Average Precision score"""
        if ap_score >= baseline * 2:
            return "Excellent"
        elif ap_score >= baseline * 1.5:
            return "Good"
        elif ap_score >= baseline * 1.2:
            return "Fair"
        else:
            return "Poor"

    def _arun(self, model_key: str = "logistic_regression", title: str = ""):
        raise NotImplementedError("Async not supported")
