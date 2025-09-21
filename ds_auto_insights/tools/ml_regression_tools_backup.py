"""
Machine Learning Regression Tools for DS Auto Insights

This module provides comprehensive linear and logistic regression capabilities
with PM-friendly interpretations and professional visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
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
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Coefficient analysis
            coefficients = pd.DataFrame({
                'feature': feature_columns,
                'coefficient': model.coef_,
                'abs_coefficient': np.abs(model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            # Calculate confidence intervals (approximate)
            residuals = y_train - y_train_pred
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
            
            # Model assumptions check
            assumptions = self._check_assumptions(X_train_scaled, y_train, y_train_pred, residuals)
            
            # Create visualizations
            # Store model results for chart tools to use later
            if hasattr(st, 'session_state'):
                if 'ml_model_results' not in st.session_state:
                    st.session_state.ml_model_results = {}
                
                # Store results for chart tools to access
                st.session_state.ml_model_results['linear_regression'] = {
                    'model_type': 'linear_regression',
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'X_train': X_train,
                    'y_train': y_train,
                    'y_train_pred': y_train_pred,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_test_pred': y_test_pred,
                    'residuals': residuals,
                    'coefficients': coefficients,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'assumptions': assumptions
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
                f""
            ])
            
            # Model performance
            result_lines.extend([
                f"📈 **Model Performance:**",
                f"  • **Training R² = {train_r2:.4f}** ({train_r2*100:.1f}% of variance explained)",
                f"  • **Test R² = {test_r2:.4f}** ({test_r2*100:.1f}% of variance explained)",
                f"  • **Training RMSE:** {train_rmse:.4f}",
                f"  • **Test RMSE:** {test_rmse:.4f}",
                f"  • **Training MAE:** {train_mae:.4f}",
                f"  • **Test MAE:** {test_mae:.4f}",
                f""
            ])
            
            # Model quality assessment
            if train_r2 > 0.7:
                result_lines.append("  • 🟢 **Strong model** - explains most of the variance")
            elif train_r2 > 0.5:
                result_lines.append("  • 🟡 **Moderate model** - explains some variance, room for improvement")
            else:
                result_lines.append("  • 🔴 **Weak model** - limited predictive power")
                
            if abs(train_r2 - test_r2) > 0.1:
                result_lines.append("  • ⚠️ **Warning:** Large difference between training and test performance suggests overfitting")
            
            result_lines.append("")
            
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
            
            if 'linearity' in assumptions:
                status = assumptions['linearity']['status']
                result_lines.append(f"  • **Linearity**: {status}")
                
            if 'normality' in assumptions:
                status = assumptions['normality']['status']
                result_lines.append(f"  • **Normality of residuals**: {status}")
                
            if 'homoscedasticity' in assumptions:
                status = assumptions['homoscedasticity']['status']
                result_lines.append(f"  • **Constant variance**: {status}")
                
            if 'independence' in assumptions:
                status = assumptions['independence']['status']
                result_lines.append(f"  • **Independence**: {status}")
                
            if 'multicollinearity' in assumptions:
                status = assumptions['multicollinearity']['status']
                result_lines.append(f"  • **Multicollinearity**: {status}")
            
            result_lines.extend([
                f"",
                f"📊 **Business Interpretation:**",
                f"Use the coefficient values to understand how changes in each feature",
                f"affect the predicted {target_column}. Focus on significant features",
                f"(marked with ✓) for business decision-making.",
                f"",
                f"**Model Equation:** {target_column} = {model.intercept_:.4f}" if include_intercept else f"**Model Equation:** {target_column} = 0"
            ])
            
            # Add coefficient equation
            for _, row in coefficients.iterrows():
                sign = "+" if row['coefficient'] >= 0 else ""
                result_lines[-1] += f" {sign} {row['coefficient']:.4f} × {row['feature']}"
            
            if missing_mask.sum() > 0:
                result_lines.append(f"\n📝 **Data Notes:** Removed {missing_count} rows with missing values")
                
            result_lines.append(f"\n� **Next Steps:** Use chart tools to visualize regression results, residuals, and feature importance.")
                
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in linear regression analysis: {str(e)}"

    def _arun(self, target_column: str, feature_columns: Optional[List[str]] = None,
              test_size: float = 0.2, include_intercept: bool = True,
              standardize_features: bool = False):
        raise NotImplementedError("Async not supported")
    
    def _check_assumptions(self, X: pd.DataFrame, y_actual: pd.Series, 
                          y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Check linear regression assumptions."""
        assumptions = {}
        
        try:
            # 1. Linearity (correlation between actual and predicted)
            linearity_corr = np.corrcoef(y_actual, y_pred)[0, 1]
            assumptions["linearity"] = {
                "correlation": round(linearity_corr, 4),
                "status": "Good" if linearity_corr > 0.8 else "Check required" if linearity_corr > 0.6 else "Poor"
            }
            
            # 2. Normality of residuals (Shapiro-Wilk test for small samples)
            if len(residuals) <= 5000:
                _, normality_p = stats.shapiro(residuals)
                assumptions["normality"] = {
                    "shapiro_wilk_p": round(normality_p, 4),
                    "status": "Good" if normality_p > 0.05 else "Violated"
                }
            else:
                # Use Kolmogorov-Smirnov for large samples
                _, normality_p = stats.kstest(residuals, 'norm')
                assumptions["normality"] = {
                    "ks_test_p": round(normality_p, 4),
                    "status": "Good" if normality_p > 0.05 else "Violated"
                }
            
            # 3. Homoscedasticity (constant variance)
            # Split residuals into groups and compare variances
            mid_point = len(residuals) // 2
            sorted_indices = np.argsort(y_pred)
            first_half_residuals = residuals[sorted_indices[:mid_point]]
            second_half_residuals = residuals[sorted_indices[mid_point:]]
            
            _, homoscedasticity_p = stats.levene(first_half_residuals, second_half_residuals)
            assumptions["homoscedasticity"] = {
                "levene_p": round(homoscedasticity_p, 4),
                "status": "Good" if homoscedasticity_p > 0.05 else "Violated"
            }
            
            # 4. Independence (Durbin-Watson test)
            # Simple autocorrelation check
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            assumptions["independence"] = {
                "autocorrelation": round(autocorr, 4),
                "status": "Good" if abs(autocorr) < 0.2 else "Check required" if abs(autocorr) < 0.4 else "Violated"
            }
            
            # 5. Multicollinearity (VIF for multiple regression)
            if X.shape[1] > 1:
                correlations = X.corr()
                max_corr = correlations.abs().values
                np.fill_diagonal(max_corr, 0)
                max_feature_corr = np.max(max_corr)
                
                assumptions["multicollinearity"] = {
                    "max_feature_correlation": round(max_feature_corr, 4),
                    "status": "Good" if max_feature_corr < 0.7 else "Check required" if max_feature_corr < 0.9 else "High"
                }
            
        except Exception as e:
            assumptions["error"] = f"Could not complete all assumption checks: {str(e)}"
            
        return assumptions


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
    
    def __init__(self):
        super().__init__()
        self.name = "run_logistic_regression"
        self.description = "Fit logistic regression models for binary classification"
        
    def execute(self, df: pd.DataFrame, target_column: str,
                feature_columns: Optional[List[str]] = None,
                test_size: float = 0.2,
                standardize_features: bool = False,
                class_weight: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute logistic regression analysis.
        
        Args:
            df: Input DataFrame
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
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found in dataset"}
            
            # Check if target is binary
            unique_values = df[target_column].dropna().unique()
            if len(unique_values) != 2:
                return {"error": f"Target column '{target_column}' must be binary (2 unique values). Found: {unique_values}"}
            
            # Convert target to 0/1 if needed
            y_original = df[target_column].copy()
            if set(unique_values) == {True, False}:
                y = y_original.astype(int)
            elif set(unique_values) == {0, 1}:
                y = y_original
            else:
                # Map to 0/1
                y = (y_original == unique_values[1]).astype(int)
                
            # Prepare features
            if feature_columns is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_cols if col != target_column]
                
            if not feature_columns:
                return {"error": "No numeric feature columns available for logistic regression"}
                
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                return {"error": f"Feature columns not found: {missing_cols}"}
                
            X = df[feature_columns].copy()
            
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
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred)
            test_precision = precision_score(y_test, y_test_pred)
            train_recall = recall_score(y_train, y_train_pred)
            test_recall = recall_score(y_test, y_test_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            # Coefficient analysis with odds ratios
            coefficients = pd.DataFrame({
                'feature': feature_columns,
                'coefficient': model.coef_[0],
                'odds_ratio': np.exp(model.coef_[0]),
                'abs_coefficient': np.abs(model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            
            # Create visualizations
            charts = self._create_logistic_charts(
                X_train, y_train, y_train_proba, X_test, y_test, y_test_proba,
                coefficients, target_column, feature_columns, unique_values
            )
            
            # PM-friendly interpretation
            interpretation = self._create_logistic_interpretation(
                coefficients, train_accuracy, test_accuracy, train_auc, test_auc,
                target_column, feature_columns, class_counts, unique_values
            )
            
            # Compile results
            results = {
                "model_type": "Logistic Regression",
                "target_variable": target_column,
                "target_classes": unique_values.tolist(),
                "features_used": feature_columns,
                "n_observations": len(X),
                "n_features": len(feature_columns),
                "class_distribution": class_counts.to_dict(),
                "standardized": standardize_features,
                "class_weight": class_weight,
                
                # Model performance
                "performance_metrics": {
                    "train_accuracy": round(train_accuracy, 4),
                    "test_accuracy": round(test_accuracy, 4),
                    "train_precision": round(train_precision, 4),
                    "test_precision": round(test_precision, 4),
                    "train_recall": round(train_recall, 4),
                    "test_recall": round(test_recall, 4),
                    "train_f1": round(train_f1, 4),
                    "test_f1": round(test_f1, 4),
                    "train_auc": round(train_auc, 4),
                    "test_auc": round(test_auc, 4)
                },
                
                # Coefficient analysis
                "coefficients": coefficients.round(4).to_dict('records'),
                "intercept": round(model.intercept_[0], 4),
                
                # Model summary
                "model_summary": interpretation,
                
                # Charts
                "charts": charts,
                
                # Predictions
                "predictions": {
                    "train_actual": y_train.tolist(),
                    "train_predicted": y_train_pred.tolist(),
                    "train_probabilities": y_train_proba.tolist(),
                    "test_actual": y_test.tolist(),
                    "test_predicted": y_test_pred.tolist(),
                    "test_probabilities": y_test_proba.tolist()
                }
            }
            
            if missing_mask.sum() > 0:
                results["data_notes"] = f"Removed {missing_count} rows with missing values"
                
            return results
            
        except Exception as e:
            return {"error": f"Logistic regression analysis failed: {str(e)}"}
    
    def _create_logistic_charts(self, X_train: pd.DataFrame, y_train: pd.Series,
                               y_train_proba: np.ndarray, X_test: pd.DataFrame,
                               y_test: pd.Series, y_test_proba: np.ndarray,
                               coefficients: pd.DataFrame, target_column: str,
                               feature_columns: List[str], unique_values: np.ndarray) -> List[Dict[str, Any]]:
        """Create comprehensive logistic regression visualizations."""
        charts = []
        
        try:
            # 1. ROC Curve
            from sklearn.metrics import roc_curve
            
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
            
            fig_roc = go.Figure()
            
            fig_roc.add_trace(go.Scatter(
                x=fpr_train, y=tpr_train,
                mode='lines',
                name=f'Training ROC',
                line=dict(color='blue', width=2)
            ))
            
            if len(X_test) != len(X_train):
                fig_roc.add_trace(go.Scatter(
                    x=fpr_test, y=tpr_test,
                    mode='lines',
                    name=f'Test ROC',
                    line=dict(color='red', width=2)
                ))
            
            # Random classifier line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', dash='dash'),
                hoverinfo='skip'
            ))
            
            fig_roc.update_layout(
                title='ROC Curve - Model Performance',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            charts.append({
                "type": "logistic_roc",
                "title": "ROC Curve",
                "figure": fig_roc
            })
            
            # 2. Probability distribution
            fig_prob = go.Figure()
            
            # Class 0 probabilities
            class_0_probs = y_train_proba[y_train == 0]
            class_1_probs = y_train_proba[y_train == 1]
            
            fig_prob.add_trace(go.Histogram(
                x=class_0_probs,
                name=f'{unique_values[0]} (Class 0)',
                opacity=0.7,
                nbinsx=30,
                histnorm='probability'
            ))
            
            fig_prob.add_trace(go.Histogram(
                x=class_1_probs,
                name=f'{unique_values[1]} (Class 1)',
                opacity=0.7,
                nbinsx=30,
                histnorm='probability'
            ))
            
            fig_prob.update_layout(
                title='Predicted Probability Distribution by Actual Class',
                xaxis_title='Predicted Probability',
                yaxis_title='Density',
                barmode='overlay'
            )
            
            charts.append({
                "type": "logistic_probability",
                "title": "Probability Distribution",
                "figure": fig_prob
            })
            
            # 3. Odds ratio chart
            fig_odds = go.Figure()
            
            # Sort by odds ratio for better visualization
            odds_sorted = coefficients.sort_values('odds_ratio', ascending=True)
            
            colors = ['red' if x < 1 else 'green' for x in odds_sorted['odds_ratio']]
            
            fig_odds.add_trace(go.Bar(
                y=odds_sorted['feature'],
                x=odds_sorted['odds_ratio'],
                orientation='h',
                marker=dict(color=colors, opacity=0.7),
                hovertemplate='<b>%{y}</b><br>Odds Ratio: %{x:.3f}<br>Coefficient: %{customdata:.3f}<extra></extra>',
                customdata=odds_sorted['coefficient']
            ))
            
            # Add vertical line at odds ratio = 1 (no effect)
            fig_odds.add_vline(x=1, line_dash="dash", line_color="black",
                              annotation_text="No Effect (OR=1)")
            
            fig_odds.update_layout(
                title='Odds Ratios (Effect on Likelihood of Positive Class)',
                xaxis_title='Odds Ratio',
                yaxis_title='Features',
                xaxis_type='log'
            )
            
            charts.append({
                "type": "logistic_odds_ratio",
                "title": "Odds Ratios",
                "figure": fig_odds
            })
            
        except Exception as e:
            charts.append({
                "type": "error",
                "title": "Chart Generation Error",
                "error": f"Could not create all charts: {str(e)}"
            })
            
        return charts
    
    def _create_logistic_interpretation(self, coefficients: pd.DataFrame, 
                                       train_accuracy: float, test_accuracy: float,
                                       train_auc: float, test_auc: float,
                                       target_column: str, feature_columns: List[str],
                                       class_counts: pd.Series, unique_values: np.ndarray) -> str:
        """Create PM-friendly interpretation of logistic regression results."""
        
        interpretation = []
        
        # Model summary
        interpretation.append(f"## Logistic Regression Model Summary")
        interpretation.append(f"**Target Variable:** {target_column}")
        interpretation.append(f"**Predicting:** {unique_values[1]} vs {unique_values[0]}")
        interpretation.append(f"**Features Used:** {', '.join(feature_columns)}")
        interpretation.append("")
        
        # Model performance
        interpretation.append(f"### Model Performance")
        interpretation.append(f"- **Training Accuracy = {train_accuracy:.1%}** ({train_accuracy:.3f})")
        interpretation.append(f"- **Test Accuracy = {test_accuracy:.1%}** ({test_accuracy:.3f})")
        interpretation.append(f"- **Training AUC = {train_auc:.3f}** (Area Under ROC Curve)")
        interpretation.append(f"- **Test AUC = {test_auc:.3f}** (Area Under ROC Curve)")
        
        if test_auc > 0.8:
            interpretation.append("- 🟢 **Excellent model** - strong predictive power")
        elif test_auc > 0.7:
            interpretation.append("- 🟡 **Good model** - reasonable predictive power")
        elif test_auc > 0.6:
            interpretation.append("- 🟠 **Fair model** - limited but useful predictive power")
        else:
            interpretation.append("- 🔴 **Poor model** - weak predictive power")
            
        interpretation.append("")
        
        # Class distribution
        interpretation.append(f"### Data Overview")
        total = class_counts.sum()
        for class_val, count in class_counts.items():
            class_name = unique_values[1] if class_val == 1 else unique_values[0]
            pct = count / total * 100
            interpretation.append(f"- **{class_name}**: {count:,} observations ({pct:.1f}%)")
        interpretation.append("")
        
        # Feature effects (odds ratios)
        interpretation.append(f"### Key Factors (Odds Ratios)")
        interpretation.append("*How much each factor increases/decreases the odds of the positive outcome:*")
        interpretation.append("")
        
        # Top features by odds ratio effect
        top_features = coefficients.head(3)
        for _, row in top_features.iterrows():
            odds_ratio = row['odds_ratio']
            
            if odds_ratio > 1:
                effect = f"**increases** odds by {(odds_ratio-1)*100:.1f}%"
                interpretation.append(f"- **{row['feature']}**: {effect} (OR = {odds_ratio:.3f})")
            else:
                effect = f"**decreases** odds by {(1-odds_ratio)*100:.1f}%"
                interpretation.append(f"- **{row['feature']}**: {effect} (OR = {odds_ratio:.3f})")
        
        interpretation.append("")
        interpretation.append("### Business Interpretation")
        interpretation.append(f"This model predicts the likelihood of **{unique_values[1]}** occurring.")
        interpretation.append("Use the odds ratios to understand which factors most strongly")
        interpretation.append(f"influence the chance of **{unique_values[1]}** vs **{unique_values[0]}**.")
        interpretation.append("")
        interpretation.append("**Odds Ratio Guide:**")
        interpretation.append("- OR > 1: Factor increases likelihood of positive outcome")
        interpretation.append("- OR < 1: Factor decreases likelihood of positive outcome")
        interpretation.append("- OR = 1: Factor has no effect")
        
        return "\n".join(interpretation)
