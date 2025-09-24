"""
Machine Learning Regression Tools for Stats Compass

This module provides comprehensive linear and logistic regression capabilities
with PM-friendly interpretations and professional visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import streamlit as st
from typing import Type, Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool


class LinearRegressionInput(BaseModel):
    target_column: str = Field(description="Column to predict (dependent variable)")
    feature_columns: Optional[List[str]] = Field(default=None, description="Columns to use as predictors (if None, use all numeric columns except target)")
    test_size: float = Field(default=0.2, description="Proportion of data for testing (0.0-0.5)")
    include_intercept: bool = Field(default=True, description="Whether to include intercept term")
    standardize_features: bool = Field(default=False, description="Whether to standardize features before fitting")


class RunLinearRegressionTool(BaseTool):
    """
    Comprehensive linear regression analysis tool for predictive modeling.
    
    Supports both simple and multiple regression with:
    - PM-friendly coefficient interpretation
    - Assumption checking and diagnostics
    - Professional visualizations
    - Business-focused insights
    """
    
    name: str = "run_linear_regression"
    description: str = "Fit linear regression models to predict continuous outcomes with comprehensive diagnostics and PM-friendly interpretations"
    args_schema: Type[BaseModel] = LinearRegressionInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        
    def _run(self, target_column: str, 
             feature_columns: Optional[List[str]] = None,
             test_size: float = 0.2, 
             include_intercept: bool = True,
             standardize_features: bool = False) -> str:
        """
        Execute linear regression analysis.
        
        Args:
            target_column: Column to predict (dependent variable)
            feature_columns: Columns to use as predictors (if None, use all numeric columns)
            test_size: Proportion of data for testing (0.0-0.5)
            include_intercept: Whether to include intercept term
            standardize_features: Whether to standardize features before fitting
            
        Returns:
            String containing formatted results for display
        """
        try:
            # Input validation
            if target_column not in self._df.columns:
                return f"❌ Target column '{target_column}' not found in dataset. Available columns: {list(self._df.columns)}"
                
            if not pd.api.types.is_numeric_dtype(self._df[target_column]):
                return f"❌ Target column '{target_column}' must be numeric for regression"
                
            if test_size < 0 or test_size > 0.5:
                return f"❌ test_size must be between 0 and 0.5"
                
            # Prepare features
            if feature_columns is None:
                # Use all numeric columns except target
                numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_cols if col != target_column]
                
            if not feature_columns:
                return "❌ No numeric feature columns available for regression"
                
            # Check for missing feature columns
            missing_cols = [col for col in feature_columns if col not in self._df.columns]
            if missing_cols:
                return f"❌ Feature columns not found: {missing_cols}"
                
            # Prepare data
            X = self._df[feature_columns].copy()
            y = self._df[target_column].copy()
            
            # Remove rows with missing values
            missing_mask = X.isnull().any(axis=1) | y.isnull()
            if missing_mask.sum() > 0:
                X = X[~missing_mask]
                y = y[~missing_mask]
                missing_count = missing_mask.sum()
                
            # Check if we have enough data
            if len(X) < 10:
                return "❌ Insufficient data for regression analysis (need at least 10 complete rows)"
                
            if len(X) < len(feature_columns) * 5:
                return f"⚠️ Warning: Limited data: {len(X)} rows for {len(feature_columns)} features. Consider using fewer features."
                
            # Split data
            if test_size > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
                
            # Standardize features if requested
            scaler = None
            if standardize_features:
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                
            # Fit model
            model = LinearRegression(fit_intercept=include_intercept)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate performance metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Calculate residuals
            residuals = y_train - y_train_pred
            
            # Coefficient analysis
            coefficients = pd.DataFrame({
                'feature': feature_columns,
                'coefficient': model.coef_,
                'abs_coefficient': np.abs(model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            # Calculate confidence intervals (approximate)
            mse = np.mean(residuals**2)
            var_coef = mse * np.linalg.inv(X_train_scaled.T @ X_train_scaled).diagonal()
            se_coef = np.sqrt(var_coef)
            t_val = stats.t.ppf(0.975, len(X_train) - len(feature_columns) - 1)
            
            coefficients['std_error'] = se_coef
            coefficients['conf_int_lower'] = coefficients['coefficient'] - t_val * se_coef
            coefficients['conf_int_upper'] = coefficients['coefficient'] + t_val * se_coef
            coefficients['significant'] = (
                (coefficients['conf_int_lower'] > 0) | 
                (coefficients['conf_int_upper'] < 0)
            )
            
            # Store comprehensive results in session state for evaluation and chart tools
            if hasattr(st, 'session_state'):
                if 'ml_model_results' not in st.session_state:
                    st.session_state.ml_model_results = {}
                
                # Store results for evaluation and chart tools to access
                st.session_state.ml_model_results['linear_regression'] = {
                    'model_type': 'linear_regression',
                    'model': model,
                    'scaler': scaler,
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred,
                    'residuals': residuals,
                    'coefficients': coefficients,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'standardized': standardize_features,
                }
            
            # Format results as string for display
            result_lines = [f"📊 **Linear Regression Analysis: {target_column}**\n"]
            
            # Model summary
            result_lines.extend([
                f"**Model Type:** Linear Regression",
                f"**Target Variable:** {target_column}",
                f"**Features Used:** {', '.join(feature_columns)}",
                f"**Observations:** {len(X):,} (after removing missing values)",
                f"**Train/Test Split:** {int((1-test_size)*100)}/{int(test_size*100)}%" if test_size > 0 else "No split",
                f"**Standardized Features:** {'Yes' if standardize_features else 'No'}",
                f"",
                f"📈 **Model Performance:**",
                f"  • Training R² = {train_r2:.3f} ({train_r2*100:.1f}% variance explained)",
                f"  • Test R² = {test_r2:.3f} ({test_r2*100:.1f}% variance explained)",
                f"  • Training RMSE = {train_rmse:.2f}",
                f"  • Test RMSE = {test_rmse:.2f}",
                f""
            ])
            
            
            # Feature importance
            result_lines.extend([
                f"🎯 **Feature Importance (Top 5):**"
            ])
            
            top_features = coefficients.head(5)
            for _, row in top_features.iterrows():
                direction = "increases" if row['coefficient'] > 0 else "decreases"
                significance = "✓ Significant" if row['significant'] else "Not significant"
                
                if standardize_features:
                    result_lines.append(f"  • **{row['feature']}**: 1 std dev increase {direction} {target_column} by {abs(row['coefficient']):.4f} units ({significance})")
                else:
                    result_lines.append(f"  • **{row['feature']}**: 1 unit increase {direction} {target_column} by {abs(row['coefficient']):.4f} units ({significance})")
            
            result_lines.append("")
            
            # Model assumptions
            result_lines.extend([
                f"⚠️ **Model Assumptions Check:**"
            ])
            
            
            result_lines.extend([
                f"",
                f"📊 **Business Interpretation:**",
                f"The model shows the following feature impacts (use these specific numbers for decisions):",
                f""
            ])
            
            # Show top 3 features with their exact coefficients for LLM to interpret
            top_3_features = coefficients.head(3)
            for _, row in top_3_features.iterrows():
                direction = "increases" if row['coefficient'] > 0 else "decreases"
                significance = "✓ Significant" if row['significant'] else "⚠️ Not significant"
                
                result_lines.append(f"• **{row['feature']}**: Coefficient = {row['coefficient']:.4f} ({significance})")
                if standardize_features:
                    result_lines.append(f"  → 1 std dev increase {direction} {target_column} by {abs(row['coefficient']):.4f} units")
                else:
                    result_lines.append(f"  → 1 unit increase {direction} {target_column} by {abs(row['coefficient']):.4f} units")
            
            # Provide model performance context for reliability assessment
            result_lines.extend([
                f"",
                f"**Model Reliability:** R² = {train_r2:.3f} (explains {train_r2*100:.1f}% of variance)",
                f"**Use for decisions:** {'✅ Model shows good predictive power' if train_r2 > 0.7 else '⚠️ Model has limited predictive power - use results cautiously' if train_r2 > 0.3 else '❌ Model has poor predictive power - collect better data before making decisions'}",
                f"",
                f"**Model Equation:** {target_column} = {model.intercept_:.4f}" if include_intercept else f"**Model Equation:** {target_column} = 0"
            ])
            
            # Add coefficient equation
            for _, row in coefficients.iterrows():
                sign = "+" if row['coefficient'] >= 0 else ""
                result_lines[-1] += f" {sign} {row['coefficient']:.4f} × {row['feature']}"
            
            if missing_mask.sum() > 0:
                result_lines.append(f"\n📝 **Data Notes:** Removed {missing_count} rows with missing values")
                
            result_lines.append(f"\n📊 **Next Steps:** Use evaluate_regression_model for comprehensive model assessment, then create charts to visualize regression results, residuals, and feature importance.")
                
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in linear regression analysis: {str(e)}"

    def _arun(self, target_column: str, feature_columns: Optional[List[str]] = None,
              test_size: float = 0.2, include_intercept: bool = True,
              standardize_features: bool = False):
        raise NotImplementedError("Async not supported")


class LogisticRegressionInput(BaseModel):
    target_column: str = Field(description="Binary column to predict (0/1 or True/False)")
    feature_columns: Optional[List[str]] = Field(default=None, description="Columns to use as predictors (if None, use all numeric columns except target)")
    test_size: float = Field(default=0.2, description="Proportion of data for testing (0.0-0.5)")
    standardize_features: bool = Field(default=False, description="Whether to standardize features before fitting")
    class_weight: Optional[str] = Field(default=None, description="Handle class imbalance ('balanced' or None)")


class RunLogisticRegressionTool(BaseTool):
    """
    Comprehensive logistic regression analysis tool for binary classification.
    
    Supports:
    - Binary classification with probability predictions
    - PM-friendly odds ratio interpretation  
    - Model diagnostics and assumption checking
    - Professional visualizations
    """
    
    name: str = "run_logistic_regression"
    description: str = "Fit logistic regression models for binary classification with probability predictions and odds ratio interpretation"
    args_schema: Type[BaseModel] = LogisticRegressionInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        
    def _run(self, target_column: str,
             feature_columns: Optional[List[str]] = None,
             test_size: float = 0.2,
             standardize_features: bool = False,
             class_weight: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute logistic regression analysis.
        
        Args:
            target_column: Binary column to predict (0/1 or True/False)
            feature_columns: Columns to use as predictors
            test_size: Proportion of data for testing
            standardize_features: Whether to standardize features
            class_weight: Handle class imbalance ('balanced' or None)
            
        Returns:
            Dictionary containing model results, metrics, and visualizations
        """
        try:
            # Input validation
            if target_column not in self._df.columns:
                return {"error": f"Target column '{target_column}' not found in dataset"}
            
            # Check if target is binary
            unique_values = self._df[target_column].dropna().unique()
            if len(unique_values) != 2:
                return {"error": f"Target column '{target_column}' must be binary (2 unique values). Found: {unique_values}"}
            
            # Convert target to 0/1 if needed
            y_original = self._df[target_column].copy()
            if set(unique_values) == {True, False}:
                y = y_original.astype(int)
            elif set(unique_values) == {0, 1}:
                y = y_original
            else:
                # Map to 0/1
                y = (y_original == unique_values[1]).astype(int)
                
            # Prepare features
            if feature_columns is None:
                numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_cols if col != target_column]
                
            if not feature_columns:
                return {"error": "No numeric feature columns available for logistic regression"}
                
            missing_cols = [col for col in feature_columns if col not in self._df.columns]
            if missing_cols:
                return {"error": f"Feature columns not found: {missing_cols}"}
                
            X = self._df[feature_columns].copy()
            
            # Remove rows with missing values
            missing_mask = X.isnull().any(axis=1) | y.isnull()
            if missing_mask.sum() > 0:
                X = X[~missing_mask]
                y = y[~missing_mask]
                missing_count = missing_mask.sum()
                
            # Check data sufficiency
            if len(X) < 20:
                return {"error": "Insufficient data for logistic regression (need at least 20 complete rows)"}
                
            # Check class balance
            class_counts = y.value_counts()
            minority_class_pct = class_counts.min() / len(y) * 100
            
            if minority_class_pct < 5:
                return {"warning": f"Severely imbalanced classes: {class_counts.to_dict()}. Consider using class_weight='balanced'"}
            elif minority_class_pct < 20:
                if class_weight is None:
                    class_weight = 'balanced'
                    
            # Split data
            if test_size > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
                
            # Standardize features if requested
            scaler = None
            if standardize_features:
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                
            # Fit model
            model = LogisticRegression(
                class_weight=class_weight,
                random_state=42,
                max_iter=1000
            )
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            
            # Coefficient analysis with odds ratios
            coefficients = pd.DataFrame({
                'feature': feature_columns,
                'coefficient': model.coef_[0],
                'odds_ratio': np.exp(model.coef_[0]),
                'abs_coefficient': np.abs(model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            
            # Store comprehensive results in session state for evaluation and chart tools
            if hasattr(st, 'session_state'):
                if 'ml_model_results' not in st.session_state:
                    st.session_state.ml_model_results = {}
                
                # Store results for evaluation and chart tools to access
                st.session_state.ml_model_results['logistic_regression'] = {
                    'model_type': 'logistic_regression',
                    'model': model,
                    'scaler': scaler,
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred,
                    'y_train_proba': y_train_proba,
                    'y_test_proba': y_test_proba,
                    'coefficients': coefficients,
                    'standardized': standardize_features,
                    'class_weight': class_weight,
                    'unique_values': unique_values
                }
            
            # Format results as string for display (like linear regression)
            result_lines = [f"📊 **Logistic Regression Analysis: {target_column}**\n"]
            
            # Model summary
            result_lines.extend([
                f"**Model Type:** Logistic Regression",
                f"**Target Variable:** {target_column} (Classes: {unique_values[0]}, {unique_values[1]})",
                f"**Features Used:** {', '.join(feature_columns)}",
                f"**Observations:** {len(X):,} (after removing missing values)",
                f"**Train/Test Split:** {int((1-test_size)*100)}/{int(test_size*100)}%" if test_size > 0 else "No split",
                f"**Standardized Features:** {'Yes' if standardize_features else 'No'}",
                f"**Class Balance:** {'Balanced' if class_weight == 'balanced' else 'Natural'}",
                f"",
                f"📈 **Model Performance:**",
                f"  • Training Accuracy = {accuracy_score(y_train, y_train_pred):.3f} ({accuracy_score(y_train, y_train_pred)*100:.1f}%)",
                f"  • Test Accuracy = {accuracy_score(y_test, y_test_pred):.3f} ({accuracy_score(y_test, y_test_pred)*100:.1f}%)",
                f"  • Training AUC = {roc_auc_score(y_train, y_train_proba):.3f}",
                f"  • Test AUC = {roc_auc_score(y_test, y_test_proba):.3f}",
                f""
            ])
            
            # Feature importance (odds ratios)
            result_lines.extend([
                f"🎯 **Feature Importance (Top 5 Odds Ratios):**"
            ])
            
            top_features = coefficients.head(5)
            for _, row in top_features.iterrows():
                odds_ratio = row['odds_ratio']
                if odds_ratio > 1:
                    effect = f"increases odds by {((odds_ratio - 1) * 100):.1f}%"
                else:
                    effect = f"decreases odds by {((1 - odds_ratio) * 100):.1f}%"
                
                result_lines.append(f"  • **{row['feature']}**: Each unit increase {effect} (OR: {odds_ratio:.3f})")
            
            result_lines.extend([
                f"",
                f"📊 **Business Interpretation:**",
                f"The model shows the following feature impacts (use these specific numbers for decisions):",
                f""
            ])
            
            # Show top 3 features with their exact odds ratios and coefficients
            top_3_features = coefficients.head(3)
            for _, row in top_3_features.iterrows():
                odds_ratio = row['odds_ratio']
                if odds_ratio > 1:
                    effect = f"increases odds by {((odds_ratio - 1) * 100):.1f}%"
                else:
                    effect = f"decreases odds by {((1 - odds_ratio) * 100):.1f}%"
                
                result_lines.append(f"• **{row['feature']}**: Odds Ratio = {odds_ratio:.3f}, Coefficient = {row['coefficient']:.4f}")
                result_lines.append(f"  → 1 unit increase {effect}")
            
            # Provide model performance context for reliability assessment
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            result_lines.extend([
                f"",
                f"**Model Reliability:** AUC = {test_auc:.3f} (discrimination ability)",
                f"**Use for decisions:** {'✅ Model shows excellent discrimination' if test_auc > 0.8 else '⚠️ Model has moderate discrimination - validate results' if test_auc > 0.6 else '❌ Model has poor discrimination - collect better data before making decisions'}",
                f"",
                f"🎯 **Classification Equation:** Probability = 1 / (1 + e^(-({model.intercept_[0]:.4f}",
            ])
            
            # Add coefficient equation
            for _, row in coefficients.iterrows():
                sign = "+" if row['coefficient'] >= 0 else ""
                result_lines[-1] += f" {sign} {row['coefficient']:.4f} × {row['feature']}"
            
            result_lines[-1] += ")))"
            
            if missing_mask.sum() > 0:
                result_lines.append(f"\n📝 **Data Notes:** Removed {missing_count} rows with missing values")
                
            result_lines.append(f"\n📊 **Next Steps:** Use evaluate_classification_model for comprehensive model assessment, then create ROC curves, precision-recall curves, and feature importance charts to visualize model performance.")
                
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in logistic regression analysis: {str(e)}"


class ARIMAInput(BaseModel):
    time_column: str = Field(description="Column containing time/date values")
    value_column: str = Field(description="Column containing the time series values to model")
    p: int = Field(default=1, description="Autoregressive (AR) order - number of lag observations")
    d: int = Field(default=1, description="Differencing order - degree of differencing")
    q: int = Field(default=1, description="Moving average (MA) order - size of moving average window")
    forecast_periods: int = Field(default=12, description="Number of periods to forecast into the future")


class RunARIMATool(BaseTool):
    """
    Simple ARIMA (AutoRegressive Integrated Moving Average) time series analysis tool.
    
    Fits ARIMA(p,d,q) model to time series data for:
    - Time series forecasting
    - Trend analysis
    - Model fit assessment
    - Simple forecasting without complex seasonal adjustments
    """
    
    name: str = "run_arima_analysis"
    description: str = "Fit ARIMA time series model for forecasting and trend analysis with simple parameterization"
    args_schema: Type[BaseModel] = ARIMAInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, time_column: str, value_column: str, p: int = 1, d: int = 1, q: int = 1, forecast_periods: int = 12) -> str:
        try:
            # Validate inputs
            if time_column not in self._df.columns:
                return f"❌ Time column '{time_column}' not found. Available columns: {list(self._df.columns)}"
            
            if value_column not in self._df.columns:
                return f"❌ Value column '{value_column}' not found. Available columns: {list(self._df.columns)}"
                        
            # Prepare data
            df_work = self._df[[time_column, value_column]].copy()
            df_work = df_work.dropna()
            
            if len(df_work) < 10:
                return f"❌ Insufficient data for ARIMA analysis. Need at least 10 observations, got {len(df_work)}"
            
            # Convert time column to datetime and sort
            try:
                df_work[time_column] = pd.to_datetime(df_work[time_column])
            except:
                return f"❌ Could not convert '{time_column}' to datetime format"
            
            df_work = df_work.sort_values(time_column).reset_index(drop=True)
            
            # Ensure numeric values
            try:
                df_work[value_column] = pd.to_numeric(df_work[value_column])
            except:
                return f"❌ Could not convert '{value_column}' to numeric values"
            
            time_series = df_work[value_column].values
            
            # Test for stationarity
            adf_test = adfuller(time_series)
            is_stationary = adf_test[1] <= 0.05
            
            # Fit ARIMA model
            model = ARIMA(time_series, order=(p, d, q))
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=forecast_periods)
            forecast_conf_int = fitted_model.get_forecast(steps=forecast_periods).conf_int()
            
            # Calculate fit statistics
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid
            
            # Create forecast dates
            last_date = df_work[time_column].iloc[-1]
            
            # Try to infer frequency
            if len(df_work) > 1:
                time_diff = df_work[time_column].iloc[1] - df_work[time_column].iloc[0]
                forecast_dates = pd.date_range(start=last_date + time_diff, periods=forecast_periods, freq=time_diff)
            else:
                # Default to daily if can't infer
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
            
            # Store results in session state
            if hasattr(st, 'session_state'):
                if 'arima_results' not in st.session_state:
                    st.session_state.arima_results = {}
                
                arima_key = f"arima_{value_column}"
                st.session_state.arima_results[arima_key] = {
                    'model': fitted_model,
                    'time_series': time_series,
                    'fitted_values': fitted_values,
                    'residuals': residuals,
                    'forecast': forecast_result.values if hasattr(forecast_result, 'values') else forecast_result,
                    'forecast_conf_int': forecast_conf_int.values if hasattr(forecast_conf_int, 'values') else forecast_conf_int,
                    'time_column': time_column,
                    'value_column': value_column,
                    'original_dates': df_work[time_column].values,
                    'forecast_dates': forecast_dates,
                    'order': (p, d, q),
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'is_stationary': is_stationary,
                    'adf_pvalue': adf_test[1]
                }
            
            # Calculate performance metrics
            mse = np.mean(residuals**2)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(mse)
            
            # Generate summary
            result_lines = [
                f"🔮 **ARIMA({p},{d},{q}) Time Series Analysis: {value_column}**",
                f"",
                f"📊 **Model Performance:**",
                f"  • AIC (Akaike Information Criterion): {fitted_model.aic:.2f}",
                f"  • BIC (Bayesian Information Criterion): {fitted_model.bic:.2f}",
                f"  • Root Mean Squared Error: {rmse:.4f}",
                f"  • Mean Absolute Error: {mae:.4f}",
                f"",
                f"📈 **Time Series Properties:**",
                f"  • Data points: {len(time_series)}",
                f"  • Stationarity test (ADF): {'✅ Stationary' if is_stationary else '⚠️ Non-stationary'} (p-value: {adf_test[1]:.4f})",
                f"  • Differencing applied: {d} {'time' if d == 1 else 'times'}",
                f"",
                f"🔮 **Forecast Summary:**",
                f"  • Forecast periods: {forecast_periods}",
                f"  • Forecast range: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}",
                f"  • Average forecast value: {np.mean(forecast_result):.2f}",
                f"",
                f"📊 **Model Interpretation:**"
            ]
            
            # Add model interpretation based on performance
            if fitted_model.aic < 100:
                result_lines.append("  • Excellent model fit - AIC indicates strong predictive capability")
            elif fitted_model.aic < 200:
                result_lines.append("  • Good model fit - reasonable predictive performance")  
            else:
                result_lines.append("  • Moderate model fit - consider adjusting ARIMA parameters")
            
            if is_stationary:
                result_lines.append("  • Time series is stationary - good for ARIMA modeling")
            else:
                result_lines.append(f"  • Time series shows non-stationarity - differencing ({d}) applied to stabilize")
            
            # Add forecast insight
            current_avg = np.mean(time_series[-min(12, len(time_series)):])  # Last 12 periods or available
            forecast_avg = np.mean(forecast_result)
            trend_direction = "increasing" if forecast_avg > current_avg else "decreasing" if forecast_avg < current_avg else "stable"
            trend_magnitude = abs((forecast_avg - current_avg) / current_avg * 100)
            
            result_lines.extend([
                f"",
                f"🎯 **Forecast Insights:**",
                f"  • Trend direction: {trend_direction.title()}",
                f"  • Expected change: {trend_magnitude:.1f}% vs recent average",
                f"  • Use create_arima_plot and create_arima_forecast_plot for visualization"
            ])
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in ARIMA analysis: {str(e)}"

    def _arun(self, time_column: str, value_column: str, p: int = 1, d: int = 1, q: int = 1, forecast_periods: int = 12):
        raise NotImplementedError("Async not supported")
