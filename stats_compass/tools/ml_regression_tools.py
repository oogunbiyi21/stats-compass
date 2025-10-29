"""
Machine Learning Regression Tools for Stats Compass

This module provides comprehensive linear and logistic regression capabilities
with PM-friendly interpretations and professional visualizations.
"""

import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import streamlit as st
from typing import Type, List, Optional
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool
from stats_compass.tools.ml_guidance import SmartMLToolMixin


class LinearRegressionInput(BaseModel):
    target_column: str = Field(description="Column to predict (dependent variable)")
    feature_columns: Optional[List[str]] = Field(default=None, description="Columns to use as predictors (if None, use all numeric columns except target)")
    test_size: float = Field(default=0.2, description="Proportion of data for testing (0.0-0.5)")
    include_intercept: bool = Field(default=True, description="Whether to include intercept term")
    standardize_features: bool = Field(default=False, description="Whether to standardize features before fitting")
    regularization_type: Optional[str] = Field(default=None, description="Type of regularization: None, 'lasso' (L1), 'ridge' (L2), or 'elasticnet' (L1+L2)")
    alpha: float = Field(default=1.0, description="Regularization strength (higher = more regularization). Only used if regularization_type is set")
    l1_ratio: float = Field(default=0.5, description="Mix of L1 and L2 for elasticnet (0=ridge, 1=lasso). Only used if regularization_type='elasticnet'")


class RunLinearRegressionTool(BaseTool, SmartMLToolMixin):
    """
    Comprehensive linear regression analysis tool for predictive modeling.
    
    Supports both simple and multiple regression with:
    - PM-friendly coefficient interpretation
    - Assumption checking and diagnostics
    - Professional visualizations
    - Business-focused insights
    - Smart workflow guidance via SmartMLToolMixin
    """
    
    name: str = "run_linear_regression"
    description: str = """Fit linear regression models to predict continuous outcomes with comprehensive diagnostics.
    
- Target must be numeric and continuous
- For data with >10% missing: Run apply_imputation first or specify clean features explicitly
- For categorical features: Run mean_target_encoding first (auto-includes encoded columns if available)
- Auto-selection: Includes only numeric features with <20% missing data

📊 Best for: Predicting sales, prices, quantities, measurements, etc."""
    args_schema: Type[BaseModel] = LinearRegressionInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _get_current_df(self) -> pd.DataFrame:
        """Get the most current dataframe, checking session state for updates"""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state.df
        return self._df

    def _run(self, target_column: str, 
             feature_columns: Optional[List[str]] = None,
             test_size: float = 0.2,
             include_intercept: bool = True,
             standardize_features: bool = False,
             regularization_type: Optional[str] = None,
             alpha: float = 1.0,
             l1_ratio: float = 0.5) -> str:
        """
        Execute linear regression analysis with smart workflow guidance.
        
        Args:
            target_column: Column to predict (dependent variable)
            feature_columns: Columns to use as predictors (if None, use all numeric + encoded columns)
            test_size: Proportion of data for testing (0.0-0.5)
            include_intercept: Whether to include intercept term
            standardize_features: Whether to standardize features before fitting
            regularization_type: Type of regularization (None, 'lasso', 'ridge', 'elasticnet')
            alpha: Regularization strength (higher = more regularization)
            l1_ratio: L1/L2 mix for elasticnet (0=ridge, 1=lasso)
            include_intercept: Whether to include intercept term
            standardize_features: Whether to standardize features before fitting
            
        Returns:
            String containing formatted results with quality assessment and workflow suggestions
        """
        try:
            # Get the most current dataframe
            df = self._get_current_df()
            
            # ============================================
            # Dataset Inspection & Feature Analysis (using mixin)
            # ============================================
            dataset_context = self._analyze_ml_features(df, target_column, feature_columns)
            
            # Validate test_size
            if test_size < 0 or test_size > 0.5:
                return f"❌ test_size must be between 0 and 0.5"
            
            # Prepare and validate data using shared helper
            # NOTE: feature_columns may be auto-populated with encoded columns
            if feature_columns is None and dataset_context.get('auto_included_encoded'):
                # Auto-include encoded features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != target_column]
                feature_columns = numeric_cols + dataset_context['auto_included_encoded']
            
            X, y, feature_columns, error, missing_count, auto_exclusion_warning = self._prepare_regression_data(
                df, target_column, feature_columns, allow_non_numeric_target=False
            )
            if error:
                return error
            
            # Store missing count in context for suggestions
            dataset_context['missing_removed'] = missing_count
            
            # Check if we have enough data (need at least 10x the number of features)
            min_required_samples = max(20, len(feature_columns) * 10)
            if len(X) < min_required_samples:
                return f"❌ Insufficient data for reliable regression. Need at least {min_required_samples} rows ({len(feature_columns)} features × 10), got {len(X)}. Consider using fewer features."
                
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
                
            # Fit model with optional regularization
            if regularization_type is None:
                model = LinearRegression(fit_intercept=include_intercept)
            elif regularization_type.lower() == 'lasso':
                model = Lasso(alpha=alpha, fit_intercept=include_intercept, random_state=42)
            elif regularization_type.lower() == 'ridge':
                model = Ridge(alpha=alpha, fit_intercept=include_intercept, random_state=42)
            elif regularization_type.lower() == 'elasticnet':
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=include_intercept, random_state=42)
            else:
                return f"❌ Invalid regularization_type: '{regularization_type}'. Must be None, 'lasso', 'ridge', or 'elasticnet'"
                
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
                
                # Create unique key to avoid overwriting previous results
                model_key = f"linear_regression_{target_column}_{int(time.time())}"
                
                # Store results for evaluation and chart tools to access
                st.session_state.ml_model_results[model_key] = {
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
                
                # Also store as 'linear_regression' for backwards compatibility
                st.session_state.ml_model_results['linear_regression'] = st.session_state.ml_model_results[model_key]
                
                # BUGFIX: Also store in trained_models for download functionality
                if not hasattr(st.session_state, 'trained_models'):
                    st.session_state.trained_models = {}
                st.session_state.trained_models[model_key] = st.session_state.ml_model_results[model_key]
            
            # ============================================
            # Quality Assessment (using mixin)
            # ============================================
            quality_assessment = self._assess_model_quality(
                metrics={'train_r2': train_r2, 'test_r2': test_r2, 'train_rmse': train_rmse, 'test_rmse': test_rmse, 'n_samples': len(X_train)},
                model_type='regression',
                dataset_context=dataset_context
            )
            
            # ============================================
            # Generate Smart Suggestions (using mixin)
            # ============================================
            suggestions = self._generate_ml_suggestions(
                quality_assessment, dataset_context, target_column, feature_columns, model_type='regression'
            )
            
            # ============================================
            # Update Workflow Metadata (using mixin)
            # ============================================
            self._update_ml_workflow_metadata(
                model_type='regression',
                model_quality=quality_assessment['quality_level'],
                features_used=feature_columns,
                model_key=model_key,
                target_column=target_column,
                primary_metric=test_r2
            )
            
            # Format results as string for display
            result_lines = [f"📊 **Linear Regression Analysis: {target_column}**\n"]
            result_lines.append(f"🔑 Model Key: `{model_key}`\n")
            
            # Show auto-exclusion warning if features were filtered out
            if auto_exclusion_warning:
                result_lines.append(f"\n{auto_exclusion_warning}\n")
            
            # Model summary
            model_type_str = "Linear Regression"
            if regularization_type:
                if regularization_type.lower() == 'lasso':
                    model_type_str = f"LASSO Regression (L1, α={alpha})"
                elif regularization_type.lower() == 'ridge':
                    model_type_str = f"Ridge Regression (L2, α={alpha})"
                elif regularization_type.lower() == 'elasticnet':
                    model_type_str = f"ElasticNet Regression (L1+L2, α={alpha}, l1_ratio={l1_ratio})"
            
            result_lines.extend([
                f"**Model Type:** {model_type_str}",
                f"**Target Variable:** {target_column}",
                f"**Features Used:** {', '.join(feature_columns)} ({len(feature_columns)} feature{'s' if len(feature_columns) != 1 else ''})",
                f"**Observations:** {len(X):,} (after removing missing values)" if missing_count > 0 else f"**Observations:** {len(X):,}",
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
            
            # ============================================
            # NEW: Auto-included features note
            # ============================================
            if dataset_context.get('auto_included_encoded'):
                result_lines.extend([
                    f"✅ **Auto-included encoded features:** {', '.join(dataset_context['auto_included_encoded'])}",
                    f"   (Detected from previous encoding step)",
                    f""
                ])
            
            # ============================================
            # Regularization-specific notes
            # ============================================
            if regularization_type and regularization_type.lower() == 'lasso':
                # Count features with zero coefficients (feature selection)
                zero_coefs = (np.abs(model.coef_) < 1e-10).sum()
                if zero_coefs > 0:
                    result_lines.extend([
                        f"🎯 **LASSO Feature Selection:** {zero_coefs}/{len(feature_columns)} features eliminated (coefficient = 0)",
                        f"   Active features: {len(feature_columns) - zero_coefs}",
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
            
            # ============================================
            # NEW: Quality Warnings (conditional)
            # ============================================
            if quality_assessment['warnings']:
                result_lines.extend([
                    f"",
                    f"⚠️ **Model Quality Assessment ({quality_assessment['quality_label']}):**"
                ])
                for warning in quality_assessment['warnings']:
                    result_lines.append(f"  • {warning}")
            
            # ============================================
            # NEW: Smart Workflow Suggestions
            # ============================================
            if suggestions:
                result_lines.extend([
                    f"",
                    f"� **Suggested Next Steps:**"
                ])
                for i, suggestion in enumerate(suggestions, 1):
                    result_lines.append(f"  {i}. {suggestion}")
            
            # Data notes (if applicable)
            if missing_count > 0:
                result_lines.append(f"\n📝 **Data Notes:** Removed {missing_count} rows with missing values")
                
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in linear regression analysis: {str(e)}"

    def _arun(self, target_column: str, feature_columns: Optional[List[str]] = None,
              test_size: float = 0.2, include_intercept: bool = True,
              standardize_features: bool = False):
        raise NotImplementedError("Async not supported")


class LogisticRegressionInput(BaseModel):
    target_column: str = Field(description="Column to predict (binary or multiclass classification)")
    feature_columns: Optional[List[str]] = Field(default=None, description="Columns to use as predictors (if None, use all numeric columns except target)")
    test_size: float = Field(default=0.2, description="Proportion of data for testing (0.0-0.5)")
    standardize_features: bool = Field(default=False, description="Whether to standardize features before fitting")
    class_weight: Optional[str] = Field(default=None, description="Handle class imbalance ('balanced' or None)")
    multi_class: str = Field(default="auto", description="Multiclass strategy: 'auto', 'ovr' (one-vs-rest), or 'multinomial' (softmax)")
    penalty: Optional[str] = Field(default='l2', description="Regularization penalty: 'l1' (LASSO), 'l2' (Ridge), 'elasticnet' (L1+L2), or None")
    C: float = Field(default=1.0, description="Inverse of regularization strength (smaller = stronger regularization). Must be positive")
    l1_ratio: float = Field(default=0.5, description="Mix of L1 and L2 for elasticnet (0=ridge, 1=lasso). Only used if penalty='elasticnet'")


class RunLogisticRegressionTool(BaseTool, SmartMLToolMixin):
    """
    Comprehensive logistic regression analysis tool for binary and multiclass classification.
    
    Supports:
    - Binary classification with probability predictions and odds ratio interpretation
    - Multiclass classification using one-vs-rest or multinomial (softmax) strategies
    - PM-friendly coefficient interpretation  
    - Model diagnostics and assumption checking
    - Professional visualizations
    - Smart workflow guidance via SmartMLToolMixin
    """
    
    name: str = "run_logistic_regression"
    description: str = """Fit logistic regression models for binary or multiclass classification with probability predictions.
    
- Target must have at least 2 classes (check with df[target].value_counts())
- For data with >10% missing: Run apply_imputation first or specify clean features explicitly
- For categorical features: Run mean_target_encoding first (auto-includes encoded columns if available)
- Auto-selection: Includes only numeric features with <20% missing data
- For imbalanced classes: Use class_weight='balanced'

📊 Best for: Predicting categories, yes/no outcomes, customer segments, etc."""
    args_schema: Type[BaseModel] = LogisticRegressionInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        
    def _get_current_df(self) -> pd.DataFrame:
        """Get the most current dataframe, checking session state for updates"""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state.df
        return self._df
        
    def _run(self, target_column: str,
             feature_columns: Optional[List[str]] = None,
             test_size: float = 0.2,
             standardize_features: bool = False,
             class_weight: Optional[str] = None,
             multi_class: str = "auto",
             penalty: Optional[str] = 'l2',
             C: float = 1.0,
             l1_ratio: float = 0.5) -> str:
        """
        Execute logistic regression analysis.
        
        Args:
            target_column: Column to predict (binary or multiclass)
            feature_columns: Columns to use as predictors
            test_size: Proportion of data for testing
            standardize_features: Whether to standardize features
            class_weight: Handle class imbalance ('balanced' or None)
            multi_class: Multiclass strategy ('auto', 'ovr', or 'multinomial')
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', or None)
            C: Inverse of regularization strength (smaller = stronger)
            l1_ratio: L1/L2 mix for elasticnet (0=ridge, 1=lasso)
            
        Returns:
            String containing formatted results for display
        """
        try:
            # Get the most current dataframe
            df = self._get_current_df()
            
            # ============================================
            # Dataset Inspection & Feature Analysis (using mixin)
            # ============================================
            dataset_context = self._analyze_ml_features(df, target_column, feature_columns)
            
            # Check target variable and determine classification type
            if target_column not in df.columns:
                return f"❌ Target column '{target_column}' not found in dataset"
            
            unique_values = df[target_column].dropna().unique()
            n_classes = len(unique_values)
            
            if n_classes < 2:
                dtype_info = df[target_column].dtype
                sample_values = df[target_column].head(10).tolist()
                return f"❌ Target column '{target_column}' must have at least 2 classes. Found only: {unique_values} (dtype: {dtype_info})\n\n💡 **Diagnostic Info:**\n  • First 10 values: {sample_values}\n  • Total rows: {len(df)}\n  • Non-null values: {df[target_column].notna().sum()}\n\nℹ️ This usually happens when:\n  1. You filtered the data and removed one class\n  2. Type mismatch in filtering (e.g., filtering integers with strings)\n  3. The column was modified incorrectly"
            
            # Determine if binary or multiclass
            is_binary = n_classes == 2
            is_multiclass = n_classes > 2
            
            # Auto-include encoded features if available
            if feature_columns is None and dataset_context.get('auto_included_encoded'):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != target_column]
                feature_columns = numeric_cols + dataset_context['auto_included_encoded']
            
            # Prepare and validate data using shared helper
            X, y_original, feature_columns, error, missing_count, auto_exclusion_warning = self._prepare_regression_data(
                df, target_column, feature_columns, allow_non_numeric_target=True
            )
            if error:
                return error
            
            # Store missing count in context for suggestions
            dataset_context['missing_removed'] = missing_count
            
            # Use LabelEncoder for all cases - handles strings, booleans, floats robustly
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_original)
            
            # Store the mapping for interpretation
            class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            
            # Missing data removal may have eliminated entire classes
            n_classes_after_cleaning = len(label_encoder.classes_)
            if n_classes_after_cleaning < 2:
                removed_info = f"\n\n💡 **What happened:**\n  • Original data had {n_classes} classes: {list(unique_values)}\n  • After removing {missing_count} rows with missing values, only {n_classes_after_cleaning} class remains: {list(label_encoder.classes_)}\n  • Missing data removal eliminated entire class(es)!"
                
                # Provide actionable guidance
                high_missing_features = [col for col in feature_columns if df[col].isnull().sum() / len(df) > 0.3]
                if high_missing_features:
                    return f"❌ After removing rows with missing data, only {n_classes_after_cleaning} class remains (need at least 2 for classification).{removed_info}\n\n🔧 **Solutions:**\n  1. **Remove high-missing features**: These features have >30% missing data: {high_missing_features}\n     → Retry without these features\n  2. **Impute missing data first**: Use apply_imputation tool before modeling\n  3. **Use simpler features**: Try model with fewer features that have less missing data"
                else:
                    return f"❌ After removing rows with missing data, only {n_classes_after_cleaning} class remains (need at least 2 for classification).{removed_info}\n\n🔧 **Solutions:**\n  1. **Impute missing data first**: Use apply_imputation tool before modeling\n  2. **Collect more data**: The dataset is too small after cleaning"
            
            # Check data sufficiency (need at least 10 samples per class for reliable modeling)
            min_samples_per_class = 10
            min_required_samples = max(30, n_classes * min_samples_per_class)
            if len(X) < min_required_samples:
                return f"❌ Insufficient data for reliable logistic regression. Need at least {min_required_samples} complete rows ({n_classes} classes × {min_samples_per_class} samples/class), got {len(X)}"
                
            # Check class balance
            class_counts = pd.Series(y).value_counts()
            minority_class_pct = class_counts.min() / len(y) * 100
            
            # Track severe imbalance warning for output
            imbalance_warning = None
            if minority_class_pct < 5:
                # Severely imbalanced - warn user even if using class_weight
                if class_weight == 'balanced':
                    imbalance_warning = f"⚠️ **Severe Class Imbalance Detected**: {class_counts.to_dict()}\n\nMinority class represents only {minority_class_pct:.1f}% of data. You're using class_weight='balanced', which helps, but consider:\n\n1. **Resampling**: Use SMOTE or RandomOverSampler for minority classes\n2. **Data Collection**: Gather more samples for minority classes\n3. **Class Combination**: Merge similar rare classes if domain-appropriate\n4. **Ensemble Methods**: Try RandomForest or XGBoost which handle imbalance better\n\n⚠️ Proceed with caution - model performance on minority classes may be poor.\n"
                else:
                    return f"❌ Severely imbalanced classes: {class_counts.to_dict()}. Minority class is only {minority_class_pct:.1f}% of data. Set class_weight='balanced' to continue."
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
                
            # Determine multiclass strategy
            if multi_class == "auto":
                # Use multinomial for multiclass, leave default for binary
                effective_multi_class = "multinomial" if is_multiclass else "auto"
            else:
                effective_multi_class = multi_class
            
            # Determine solver based on penalty
            # L1 and elasticnet require 'saga' or 'liblinear' (saga supports multinomial)
            # L2 works with 'lbfgs', 'saga', 'liblinear'
            # None penalty requires 'saga' or other compatible solvers
            if penalty == 'l1':
                solver = 'saga'  # saga supports both binary and multiclass
            elif penalty == 'elasticnet':
                solver = 'saga'  # only saga supports elasticnet
            elif penalty is None or penalty.lower() == 'none':
                solver = 'saga'  # saga supports no penalty
                penalty = None
            else:  # l2 or default
                solver = 'lbfgs'  # fast and works well for L2
                
            # Fit model with regularization parameters
            if penalty == 'elasticnet':
                model = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    l1_ratio=l1_ratio,
                    class_weight=class_weight,
                    random_state=42,
                    max_iter=2000,  # saga may need more iterations
                    multi_class=effective_multi_class,
                    solver=solver
                )
            else:
                model = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    class_weight=class_weight,
                    random_state=42,
                    max_iter=2000 if solver == 'saga' else 1000,
                    multi_class=effective_multi_class,
                    solver=solver
                )
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            y_train_proba = model.predict_proba(X_train_scaled)
            y_test_proba = model.predict_proba(X_test_scaled)
            
            # For binary classification, extract probability of positive class
            if is_binary:
                y_train_proba_positive = y_train_proba[:, 1]
                y_test_proba_positive = y_test_proba[:, 1]
            else:
                # For multiclass, we'll use the max probability
                y_train_proba_positive = np.max(y_train_proba, axis=1)
                y_test_proba_positive = np.max(y_test_proba, axis=1)
            
            
            # Coefficient analysis
            if is_binary:
                # For binary classification, use odds ratios
                coefficients = pd.DataFrame({
                    'feature': feature_columns,
                    'coefficient': model.coef_[0],
                    'odds_ratio': np.exp(model.coef_[0]),
                    'abs_coefficient': np.abs(model.coef_[0])
                }).sort_values('abs_coefficient', ascending=False)
            else:
                # For multiclass, show coefficients for each class
                # We'll focus on the average absolute coefficient across classes for ranking
                avg_abs_coef = np.mean(np.abs(model.coef_), axis=0)
                coefficients = pd.DataFrame({
                    'feature': feature_columns,
                    'abs_coefficient': avg_abs_coef,
                    'coefficients_by_class': [model.coef_[:, i] for i in range(len(feature_columns))]
                }).sort_values('abs_coefficient', ascending=False)
            
            # Store comprehensive results in session state for evaluation and chart tools
            if hasattr(st, 'session_state'):
                if 'ml_model_results' not in st.session_state:
                    st.session_state.ml_model_results = {}
                
                # Create unique key to avoid overwriting previous results
                model_key = f"logistic_regression_{target_column}_{int(time.time())}"
                
                # Store results for evaluation and chart tools to access
                st.session_state.ml_model_results[model_key] = {
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
                    'unique_values': unique_values,
                    'is_binary': is_binary,
                    'is_multiclass': is_multiclass,
                    'n_classes': n_classes,
                    'multi_class_strategy': effective_multi_class,
                    'class_mapping': class_mapping  # Always store mapping now
                }
                
                # Also store as 'logistic_regression' for backwards compatibility
                st.session_state.ml_model_results['logistic_regression'] = st.session_state.ml_model_results[model_key]
                
                # BUGFIX: Also store in trained_models for download functionality
                if not hasattr(st.session_state, 'trained_models'):
                    st.session_state.trained_models = {}
                st.session_state.trained_models[model_key] = st.session_state.ml_model_results[model_key]
            
            # ============================================
            # Quality Assessment (using mixin)
            # ============================================
            # Prepare metrics for quality assessment
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if is_binary:
                train_auc = roc_auc_score(y_train, y_train_proba_positive)
                test_auc = roc_auc_score(y_test, y_test_proba_positive)
                metrics = {
                    'train_auc': train_auc,
                    'test_auc': test_auc,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'n_samples': len(X_train)
                }
            else:
                # For multiclass, use accuracy as primary metric
                metrics = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_auc': train_acc,  # Use accuracy as proxy for multiclass
                    'test_auc': test_acc,
                    'n_samples': len(X_train)
                }
            
            quality_assessment = self._assess_model_quality(
                metrics=metrics,
                model_type='classification',
                dataset_context=dataset_context
            )
            
            # ============================================
            # Generate Smart Suggestions (using mixin)
            # ============================================
            suggestions = self._generate_ml_suggestions(
                quality_assessment, dataset_context, target_column,
                feature_columns, model_type='classification'
            )
            
            # ============================================
            # Update Workflow Metadata (using mixin)
            # ============================================
            self._update_ml_workflow_metadata(
                model_type='classification',
                model_quality=quality_assessment['quality_level'],
                features_used=feature_columns,
                model_key=model_key,
                target_column=target_column,
                primary_metric=test_auc if is_binary else test_acc
            )
            
            # Format results as string for display
            classification_type = "Binary" if is_binary else f"Multiclass ({n_classes} classes)"
            result_lines = []
            
            # Add imbalance warning at the top if present
            if imbalance_warning:
                result_lines.append(imbalance_warning)
            
            result_lines.append(f"📊 **{classification_type} Logistic Regression Analysis: {target_column}**\n")
            result_lines.append(f"🔑 Model Key: `{model_key}`\n")
            
            # Show auto-exclusion warning if features were filtered out
            if auto_exclusion_warning:
                result_lines.append(f"\n{auto_exclusion_warning}\n")
            
            # Add prominent next steps note
            result_lines.append(f"📋 **Next Steps:**")
            result_lines.append(f"  1️⃣ Run `evaluate_classification_model(model_key='{model_key}')` for detailed metrics")
            result_lines.append(f"  2️⃣ Create visualizations: `create_roc_curve`, `create_precision_recall_curve`, `create_feature_importance_chart`\n")
            
            # Model summary with regularization info
            model_type_str = f"{classification_type} Logistic Regression"
            if penalty and penalty.lower() != 'none':
                if penalty == 'l1':
                    model_type_str += f" (L1/LASSO, C={C})"
                elif penalty == 'l2':
                    model_type_str += f" (L2/Ridge, C={C})"
                elif penalty == 'elasticnet':
                    model_type_str += f" (ElasticNet, C={C}, l1_ratio={l1_ratio})"
            
            if is_binary:
                classes_info = f"Classes: {unique_values[0]}, {unique_values[1]}"
            else:
                classes_info = f"Classes: {', '.join(map(str, unique_values))}"
                
            result_lines.extend([
                f"**Model Type:** {model_type_str}",
                f"**Solver:** {solver}",
                f"**Target Variable:** {target_column} ({classes_info})",
                f"**Features Used:** {', '.join(feature_columns)}",
                f"**Observations:** {len(X):,} (after removing missing values)",
                f"**Train/Test Split:** {int((1-test_size)*100)}/{int(test_size*100)}%" if test_size > 0 else "No split",
                f"**Standardized Features:** {'Yes' if standardize_features else 'No'}",
                f"**Class Balance:** {'Balanced' if class_weight == 'balanced' else 'Natural'}",
                f"**Multiclass Strategy:** {effective_multi_class}" if is_multiclass else "",
                f"",
                f"📈 **Model Performance:**",
                f"  • Training Accuracy = {accuracy_score(y_train, y_train_pred):.3f} ({accuracy_score(y_train, y_train_pred)*100:.1f}%)",
                f"  • Test Accuracy = {accuracy_score(y_test, y_test_pred):.3f} ({accuracy_score(y_test, y_test_pred)*100:.1f}%)",
            ])
            
            # Add AUC only for binary classification (multiclass AUC is more complex)
            if is_binary:
                result_lines.extend([
                    f"  • Training AUC = {roc_auc_score(y_train, y_train_proba_positive):.3f}",
                    f"  • Test AUC = {roc_auc_score(y_test, y_test_proba_positive):.3f}",
                ])
            
            result_lines.append("")
            
            # ============================================
            # Regularization-specific notes (L1 feature selection)
            # ============================================
            if penalty == 'l1':
                # Count features with zero coefficients (feature selection)
                if is_binary:
                    zero_coefs = (np.abs(model.coef_[0]) < 1e-10).sum()
                else:
                    # For multiclass, count features that are zero across all classes
                    zero_coefs = (np.abs(model.coef_).max(axis=0) < 1e-10).sum()
                
                if zero_coefs > 0:
                    result_lines.extend([
                        f"🎯 **L1 (LASSO) Feature Selection:** {zero_coefs}/{len(feature_columns)} features eliminated (coefficient ≈ 0)",
                        f"   Active features: {len(feature_columns) - zero_coefs}",
                        f""
                    ])
            
            # Feature importance
            if is_binary:
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
            else:
                result_lines.extend([
                    f"🎯 **Feature Importance (Top 5 by Average Impact):**"
                ])
                
                top_features = coefficients.head(5)
                for _, row in top_features.iterrows():
                    result_lines.append(f"  • **{row['feature']}**: Average coefficient magnitude: {row['abs_coefficient']:.3f}")
                    # Show coefficients for each class
                    class_coefs = row['coefficients_by_class']
                    for i, coef in enumerate(class_coefs):
                        class_name = list(unique_values)[i] if is_multiclass else f"Class {i}"
                        result_lines.append(f"    → {class_name}: {coef:.3f}")
            
            result_lines.extend([
                f"",
                f"📊 **Business Interpretation:**",
                f"The model shows the following feature impacts (use these specific numbers for decisions):",
                f""
            ])
            
            # Show top 3 features with their interpretation
            top_3_features = coefficients.head(3)
            for _, row in top_3_features.iterrows():
                if is_binary:
                    odds_ratio = row['odds_ratio']
                    if odds_ratio > 1:
                        effect = f"increases odds by {((odds_ratio - 1) * 100):.1f}%"
                    else:
                        effect = f"decreases odds by {((1 - odds_ratio) * 100):.1f}%"
                    
                    result_lines.append(f"• **{row['feature']}**: Odds Ratio = {odds_ratio:.3f}, Coefficient = {row['coefficient']:.4f}")
                    result_lines.append(f"  → 1 unit increase {effect}")
                else:
                    result_lines.append(f"• **{row['feature']}**: Average coefficient magnitude = {row['abs_coefficient']:.3f}")
                    result_lines.append(f"  → Higher values indicate stronger influence on classification decisions")
            
            # Provide model performance context for reliability assessment
            if is_binary:
                test_auc = roc_auc_score(y_test, y_test_proba_positive)
                model_metric = f"AUC = {test_auc:.3f} (discrimination ability)"
                reliability_threshold = test_auc
                metric_name = "discrimination"
            else:
                test_accuracy = accuracy_score(y_test, y_test_pred)
                model_metric = f"Accuracy = {test_accuracy:.3f} ({test_accuracy*100:.1f}%)"
                reliability_threshold = test_accuracy
                metric_name = "accuracy"
            
            result_lines.extend([
                f"",
                f"**Model Reliability:** {model_metric}",
                f"**Use for decisions:** {'✅ Model shows excellent ' + metric_name if reliability_threshold > 0.8 else '⚠️ Model has moderate ' + metric_name + ' - validate results' if reliability_threshold > 0.6 else '❌ Model has poor ' + metric_name + ' - collect better data before making decisions'}",
                f"",
            ])
            
            # Add model equation (simplified for multiclass)
            if is_binary:
                result_lines.append(f"🎯 **Classification Equation:** Probability = 1 / (1 + e^(-({model.intercept_[0]:.4f}")
                for _, row in coefficients.iterrows():
                    sign = "+" if row['coefficient'] >= 0 else ""
                    result_lines[-1] += f" {sign} {row['coefficient']:.4f} × {row['feature']}"
                result_lines[-1] += ")))"
            else:
                result_lines.append(f"🎯 **Classification:** Uses {effective_multi_class} strategy with {n_classes} classes")
                result_lines.append(f"  → Softmax function converts linear combinations to class probabilities")
            
            core_results = "\n".join(result_lines)
            
            # ============================================
            # Auto-included features note
            # ============================================
            auto_included_note = None
            if dataset_context.get('auto_included_encoded'):
                auto_included_note = f"✅ **Auto-included encoded features:** {', '.join(dataset_context['auto_included_encoded'])}\n   (Detected from previous encoding step)"
            
            # ============================================
            # Format final output with smart guidance (using mixin)
            # ============================================
            return self._format_ml_output(
                core_results=core_results,
                quality_assessment=quality_assessment,
                suggestions=suggestions,
                auto_included_note=auto_included_note
            )
            
        except Exception as e:
            return f"❌ Error in logistic regression analysis: {str(e)}"


class FindOptimalARIMAInput(BaseModel):
    time_column: str = Field(description="Column containing time/date values")
    value_column: str = Field(description="Column containing the time series values")
    max_p: int = Field(default=3, description="Maximum autoregressive order to test (default: 3)")
    max_d: int = Field(default=2, description="Maximum differencing order to test (default: 2)")
    max_q: int = Field(default=3, description="Maximum moving average order to test (default: 3)")
    start_date: str = Field(default="", description="Start date for time slice (YYYY-MM-DD)")
    end_date: str = Field(default="", description="End date for time slice (YYYY-MM-DD)")


class FindOptimalARIMAParametersTool(BaseTool):
    """
    Find optimal ARIMA parameters using simple grid search with statsmodels.
    Tests all combinations of (p,d,q) and selects the one with lowest AIC.
    """
    
    name: str = "find_optimal_arima_parameters"
    description: str = "Find optimal ARIMA(p,d,q) parameters using grid search with AIC model selection. No external dependencies - pure statsmodels."
    args_schema: Type[BaseModel] = FindOptimalARIMAInput
    
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _get_current_df(self) -> pd.DataFrame:
        """Get the most current dataframe, checking session state for updates"""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state.df
        return self._df
    
    def _simple_parameter_search(self, time_series, max_p=3, max_d=2, max_q=3):
        """
        Simple grid search for ARIMA parameters.
        No external dependencies - just statsmodels.
        """
        
        best_aic = float('inf')
        best_order = None
        best_bic = None
        tested_count = 0
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        tested_count += 1
                        model = ARIMA(time_series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_bic = fitted.bic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order, best_aic, best_bic, tested_count
    
    def _run(
        self, 
        time_column: str, 
        value_column: str,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        start_date: str = "",
        end_date: str = ""
    ) -> str:
        try:
            # Get and validate data
            df = self._get_current_df()
            
            if time_column not in df.columns:
                return f"❌ Time column '{time_column}' not found. Available: {list(df.columns)}"
            
            if value_column not in df.columns:
                return f"❌ Value column '{value_column}' not found. Available: {list(df.columns)}"
            
            # Prepare data
            df_work = df[[time_column, value_column]].copy().dropna()
            
            if len(df_work) < 20:
                return f"❌ Insufficient data. Need at least 20 observations for ARIMA parameter search, got {len(df_work)}"
            
            # Convert and sort
            try:
                df_work[time_column] = pd.to_datetime(df_work[time_column])
            except:
                return f"❌ Could not convert '{time_column}' to datetime"
            
            df_work = df_work.sort_values(time_column).reset_index(drop=True)
            
            # Apply time slicing if provided
            if start_date or end_date:
                original_length = len(df_work)
                if start_date:
                    df_work = df_work[df_work[time_column] >= pd.to_datetime(start_date)]
                if end_date:
                    df_work = df_work[df_work[time_column] <= pd.to_datetime(end_date)]
                
                if len(df_work) == 0:
                    return f"❌ No data in date range: {start_date} to {end_date}"
                
                slice_info = f" (sliced from {original_length} to {len(df_work)} observations)"
            else:
                slice_info = ""
            
            # Convert to numeric
            try:
                df_work[value_column] = pd.to_numeric(df_work[value_column])
            except:
                return f"❌ Could not convert '{value_column}' to numeric"
            
            time_series = df_work[value_column].values
            
            # Run grid search
            result_lines = [
                f"🔍 **Finding Optimal ARIMA Parameters for {value_column}**",
                f"",
                f"📊 Dataset: {len(time_series)} observations{slice_info}",
                f"🔎 Search space: p≤{max_p}, d≤{max_d}, q≤{max_q}",
                f"⏳ Testing combinations... (may take 30-60 seconds)",
                f""
            ]
            
            best_order, best_aic, best_bic, tested_count = self._simple_parameter_search(
                time_series, max_p, max_d, max_q
            )
            
            if best_order is None:
                return "❌ Could not find any valid ARIMA model. Try different parameters or check your data."
            
            p, d, q = best_order
            
            result_lines.extend([
                f"✅ **Optimal Parameters Found** (tested {tested_count} combinations):",
                f"  • ARIMA Order: ({p}, {d}, {q})",
                f"  • AIC: {best_aic:.2f}",
                f"  • BIC: {best_bic:.2f}",
                f"",
                f"📊 **What This Means:**",
                f"  • p={p}: Uses {p} past {'value' if p == 1 else 'values'} for prediction",
                f"  • d={d}: Time series {'needs' if d > 0 else 'does not need'} differencing",
                f"  • q={q}: Uses {q} past forecast {'error' if q == 1 else 'errors'}",
                f"",
                f"💡 **Next Step:**",
                f"  Run ARIMA analysis with these parameters:",
                f"  `run_arima_analysis(time_column='{time_column}', value_column='{value_column}', p={p}, d={d}, q={q})`"
            ])
            
            # Store in session state for convenience
            if hasattr(st, 'session_state'):
                if 'optimal_arima_params' not in st.session_state:
                    st.session_state.optimal_arima_params = {}
                
                st.session_state.optimal_arima_params[value_column] = {
                    'order': best_order,
                    'aic': best_aic,
                    'bic': best_bic
                }
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error finding optimal parameters: {str(e)}"
    
    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")


class ARIMAInput(BaseModel):
    time_column: str = Field(description="Column containing time/date values")
    value_column: str = Field(description="Column containing the time series values to model")
    p: int = Field(default=1, description="Autoregressive (AR) order - number of lag observations")
    d: int = Field(default=1, description="Differencing order - degree of differencing")
    q: int = Field(default=1, description="Moving average (MA) order - size of moving average window")
    forecast_periods: int = Field(default=12, description="Number of periods to forecast into the future (used if forecast_number and forecast_unit not specified)")
    forecast_number: Optional[int] = Field(default=None, description="Number of time units to forecast (e.g., 30 for '30 days', 6 for '6 months')")
    forecast_unit: Optional[str] = Field(default=None, description="Time unit for forecast: 'days', 'weeks', 'months', 'quarters', or 'years'")
    start_date: str = Field(default="", description="Start date for time slice (YYYY-MM-DD format, empty for full dataset)")
    end_date: str = Field(default="", description="End date for time slice (YYYY-MM-DD format, empty for full dataset)")


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
    description: str = "Fit ARIMA time series model for forecasting and trend analysis. Supports structured time period inputs (forecast_number + forecast_unit like 30 days, 6 months, 2 years) with automatic step calculation based on pandas-detected data frequency."
    args_schema: Type[BaseModel] = ARIMAInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _get_current_df(self) -> pd.DataFrame:
        """Get the most current dataframe, checking session state for updates"""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state.df
        return self._df

    def _infer_time_frequency(self, time_index: pd.DatetimeIndex) -> pd.Timedelta:
        """
        Infer the frequency of a time series as a Timedelta.
        Returns the MEDIAN time difference between consecutive observations.
        
        Why median? Robust to:
        - Missing data points
        - Occasional gaps (weekends, holidays)
        - Irregular but mostly-consistent spacing
        
        Args:
            time_index: DatetimeIndex of the time series
            
        Returns:
            pd.Timedelta representing the typical time between observations
        """
        if len(time_index) < 2:
            return pd.Timedelta(days=1)  # Default fallback
        
        # Calculate all time differences
        time_diffs = time_index.to_series().diff().dropna()
        
        if len(time_diffs) == 0:
            return pd.Timedelta(days=1)
        
        # Use median - more robust than mean or mode
        return time_diffs.median()

    def _convert_forecast_period_to_steps(
        self, 
        time_index: pd.DatetimeIndex,
        forecast_number: int, 
        forecast_unit: str
    ) -> int:
        """
        Convert a human-readable forecast period (e.g., "30 days", "6 months") 
        into the number of data points to forecast.
        
        Args:
            time_index: DatetimeIndex of the time series
            forecast_number: Number of time units (e.g., 30 for "30 days")
            forecast_unit: Unit of time ('days', 'weeks', 'months', 'quarters', 'years')
            
        Returns:
            Number of forecast steps to generate
        """
        # Get the actual data frequency
        data_freq = self._infer_time_frequency(time_index)
        
        # Convert user's request to timedelta
        unit_mapping = {
            'days': pd.Timedelta(days=forecast_number),
            'weeks': pd.Timedelta(weeks=forecast_number),
            'months': pd.Timedelta(days=forecast_number * 30),  # Approximate
            'quarters': pd.Timedelta(days=forecast_number * 91),  # Approximate
            'years': pd.Timedelta(days=forecast_number * 365)  # Approximate
        }
        
        requested_period = unit_mapping.get(forecast_unit.lower())
        if requested_period is None:
            # Invalid unit, just return the number as-is
            return forecast_number
        
        # Calculate steps: how many data_freq periods fit in requested_period?
        steps = int(round(requested_period / data_freq))
        
        return max(1, steps)  # At least 1 step

    def _generate_forecast_dates(
        self, 
        last_date: pd.Timestamp, 
        data_freq: pd.Timedelta, 
        steps: int
    ) -> pd.DatetimeIndex:
        """
        Generate forecast dates using simple timedelta arithmetic.
        
        Args:
            last_date: Last date in the historical data
            data_freq: Time frequency as a Timedelta
            steps: Number of forecast steps
            
        Returns:
            DatetimeIndex of forecast dates
        """
        return pd.date_range(
            start=last_date + data_freq,
            periods=steps,
            freq=data_freq
        )

    def _parse_forecast_period(self, df_work: pd.DataFrame, time_column: str, forecast_number: int, forecast_unit: str) -> int:
        """
        Convert forecast request (e.g., 30 days, 6 months) into appropriate number of forecast steps.
        
        This is a simple wrapper around _convert_forecast_period_to_steps for backward compatibility.
        
        Args:
            df_work: DataFrame with time series data
            time_column: Name of the time column
            forecast_number: Number of time units (e.g., 30 for "30 days")
            forecast_unit: Unit of time ('days', 'weeks', 'months', 'quarters', 'years')
            
        Returns:
            Number of forecast steps to generate
        """
        time_index = pd.DatetimeIndex(df_work[time_column])
        return self._convert_forecast_period_to_steps(time_index, forecast_number, forecast_unit)

    def _run(self, time_column: str, value_column: str, p: int = 1, d: int = 1, q: int = 1, 
             forecast_periods: int = 12, forecast_number: Optional[int] = None, 
             forecast_unit: Optional[str] = None, start_date: str = "", end_date: str = "") -> str:
        try:
            # Get the most current dataframe
            df = self._get_current_df()
            
            # Validate inputs
            if time_column not in df.columns:
                return f"❌ Time column '{time_column}' not found. Available columns: {list(df.columns)}"
            
            if value_column not in df.columns:
                return f"❌ Value column '{value_column}' not found. Available columns: {list(df.columns)}"
                        
            # Prepare data
            df_work = df[[time_column, value_column]].copy()
            df_work = df_work.dropna()
            
            if len(df_work) < 30:
                return f"❌ Insufficient data for ARIMA analysis. Need at least 30 observations for reliable modeling, got {len(df_work)}"
            
            # Convert time column to datetime and sort (IMPORTANT: sorting ensures frequency calculation is correct)
            try:
                df_work[time_column] = pd.to_datetime(df_work[time_column])
            except:
                return f"❌ Could not convert '{time_column}' to datetime format"
            
            df_work = df_work.sort_values(time_column).reset_index(drop=True)
            
            # Apply time slicing if dates provided
            original_length = len(df_work)
            if start_date or end_date:
                try:
                    if start_date:
                        start_dt = pd.to_datetime(start_date)
                        df_work = df_work[df_work[time_column] >= start_dt]
                    if end_date:
                        end_dt = pd.to_datetime(end_date)
                        df_work = df_work[df_work[time_column] <= end_dt]
                    
                    if len(df_work) == 0:
                        return f"❌ No data found in specified date range: {start_date} to {end_date}"
                    
                    df_work = df_work.reset_index(drop=True)
                    slice_info = f" (sliced from {original_length} to {len(df_work)} observations)"
                except Exception as e:
                    return f"❌ Error processing date range: {str(e)}"
            else:
                slice_info = ""
            
            # Ensure numeric values
            try:
                df_work[value_column] = pd.to_numeric(df_work[value_column])
            except:
                return f"❌ Could not convert '{value_column}' to numeric values"
            
            time_series = df_work[value_column].values
            
            # Parse forecast period if number and unit provided (overrides forecast_periods)
            if forecast_number is not None and forecast_unit is not None:
                forecast_periods = self._parse_forecast_period(df_work, time_column, forecast_number, forecast_unit)
            
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
            
            # Diagnostic checks for overfitting
            # 1. Check if model is just following noise (R² too high for time series)
            actual_subset = time_series[len(time_series) - len(fitted_values):]
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((actual_subset - np.mean(actual_subset))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 2. Check residual patterns (should be white noise)
            residual_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
            
            # 3. Check if fitted values are too close to actual (overfitting indicator)
            mean_abs_error = np.mean(np.abs(residuals))
            data_std = np.std(actual_subset)
            overfitting_ratio = mean_abs_error / data_std if data_std > 0 else 0
            
            # 4. Ljung-Box test for residual autocorrelation (if available)
            try:
                
                ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
                ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]
            except:
                ljung_box_pvalue = None
            
            # Create forecast dates using simple, reliable method
            last_date = df_work[time_column].iloc[-1]
            time_index = pd.DatetimeIndex(df_work[time_column])
            
            # Infer frequency as Timedelta (median time difference)
            inferred_freq = self._infer_time_frequency(time_index)
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(last_date, inferred_freq, forecast_periods)
            
            # Store results in session state
            if hasattr(st, 'session_state'):
                if 'arima_results' not in st.session_state:
                    st.session_state.arima_results = {}
                
                # Create unique key to avoid overwriting previous results
                arima_key = f"arima_{value_column}_{int(time.time())}"
                
                # Create time series with proper datetime index
                time_series_with_index = pd.Series(time_series, index=df_work[time_column])
                
                st.session_state.arima_results[arima_key] = {
                    'model': fitted_model,
                    'time_series': time_series_with_index,  # Now has datetime index
                    'fitted_values': fitted_values,
                    'residuals': residuals,
                    'forecast': forecast_result.values if hasattr(forecast_result, 'values') else forecast_result,
                    'forecast_conf_int': forecast_conf_int.values if hasattr(forecast_conf_int, 'values') else forecast_conf_int,
                    'time_column': time_column,
                    'value_column': value_column,
                    'original_dates': df_work[time_column].values,
                    'forecast_dates': forecast_dates.values,  # Ensure it's stored as array
                    'inferred_freq': inferred_freq,  # Store the inferred frequency for plot tool
                    'order': (p, d, q),
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'is_stationary': is_stationary,
                    'adf_pvalue': adf_test[1],
                    'slice_info': slice_info
                }
                
                # Also store with simple key for backwards compatibility
                st.session_state.arima_results[f"arima_{value_column}"] = st.session_state.arima_results[arima_key]
            
            # Calculate performance metrics
            mse = np.mean(residuals**2)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(mse)
            
            # Overfitting diagnostics
            overfitting_warnings = []
            if r_squared > 0.95:
                overfitting_warnings.append("⚠️ R² too high - possible overfitting")
            if overfitting_ratio < 0.1:
                overfitting_warnings.append("⚠️ Fitted values too close to actual - likely overfitting")
            if abs(residual_autocorr) > 0.1:
                overfitting_warnings.append("⚠️ Residuals show autocorrelation - model may be inadequate")
            if ljung_box_pvalue is not None and ljung_box_pvalue < 0.05:
                overfitting_warnings.append("⚠️ Ljung-Box test failed - residuals not white noise")
            
            # Generate summary
            result_lines = [
                f"🔮 **ARIMA({p},{d},{q}) Time Series Analysis: {value_column}**",
                f"🔑 Model Key: `{arima_key}`",
                f"",
                f"📊 **Model Performance:**",
                f"  • AIC (Akaike Information Criterion): {fitted_model.aic:.2f}",
                f"  • BIC (Bayesian Information Criterion): {fitted_model.bic:.2f}",
                f"  • Root Mean Squared Error: {rmse:.4f}",
                f"  • Mean Absolute Error: {mae:.4f}",
                f"  • R-squared: {r_squared:.4f}",
                f"",
                f"📈 **Time Series Properties:**",
                f"  • Data points: {len(time_series)}{slice_info}",
                f"  • Date range: {df_work[time_column].min().strftime('%Y-%m-%d')} to {df_work[time_column].max().strftime('%Y-%m-%d')}",
                f"  • Stationarity test (ADF): {'✅ Stationary' if is_stationary else '⚠️ Non-stationary'} (p-value: {adf_test[1]:.4f})",
                f"  • Differencing applied: {d} {'time' if d == 1 else 'times'}",
                f"",
                f"🔍 **Model Diagnostics:**",
                f"  • Residual autocorrelation: {residual_autocorr:.4f}",
                f"  • Overfitting ratio (MAE/StdDev): {overfitting_ratio:.4f}",
            ]
            
            # Add Ljung-Box p-value with proper formatting
            if ljung_box_pvalue is not None:
                result_lines.append(f"  • Ljung-Box p-value: {ljung_box_pvalue:.4f}")
            else:
                result_lines.append(f"  • Ljung-Box p-value: N/A")
            
            # Calculate data frequency for display
            freq_days = inferred_freq.days + inferred_freq.seconds / 86400  # Convert to decimal days
            
            # Build forecast summary with conversion explanation
            forecast_summary = [
                f"",
                f"🔮 **Forecast Summary:**",
                f"  • Forecast periods: {forecast_periods} data points"
            ]
            
            # Add conversion explanation if forecast was specified in time units
            if forecast_number is not None and forecast_unit is not None:
                calendar_days = (forecast_dates[-1] - forecast_dates[0]).days + 1
                forecast_summary.append(f"  • Conversion: {forecast_number} {forecast_unit} → {forecast_periods} data points (based on ~{freq_days:.1f}-day frequency)")
                forecast_summary.append(f"  • Forecast range: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')} ({calendar_days} calendar days)")
            else:
                forecast_summary.append(f"  • Forecast range: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
            
            forecast_summary.extend([
                f"  • Data frequency: ~{freq_days:.1f} days between observations",
                f"  • Average forecast value: {np.mean(forecast_result):.2f}",
                f"",
                f"📊 **Model Interpretation:**"
            ])
            
            result_lines.extend(forecast_summary)
            
            # Add overfitting warnings if detected
            if overfitting_warnings:
                result_lines.append("")
                result_lines.append("⚠️ **Potential Issues Detected:**")
                for warning in overfitting_warnings:
                    result_lines.append(f"  • {warning}")
                result_lines.append("  • Consider: Lower order parameters, train/test split, or alternative models")
                result_lines.append("")
            
            # Add model interpretation based on performance
            result_lines.append("📊 **Model Interpretation:**")
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
                f"",
                f"📊 **Next Steps for Visualization:**",
                f"  • To plot model fit: `create_arima_plot(title='ARIMA Model Fit')`",
                f"  • To plot forecast: `create_arima_forecast_plot(forecast_steps={forecast_periods}, title='Forecast')`",
                f"  • Note: Use forecast_steps={forecast_periods} to show all {forecast_periods} computed data points"
            ])
            
            if forecast_number is not None and forecast_unit is not None:
                result_lines.append(f"  • This covers the requested {forecast_number} {forecast_unit} forecast period")
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error in ARIMA analysis: {str(e)}"

    def _arun(self, time_column: str, value_column: str, p: int = 1, d: int = 1, q: int = 1, forecast_periods: int = 12):
        raise NotImplementedError("Async not supported")
