# stats_compass/tools/ml_evaluation_tools.py
"""
ML Model Evaluation Tools for DS Auto Insights.
Provides comprehensive model evaluation capabilities including metrics, 
assumption checking, and validation for machine learning models.
"""

from typing import Type, Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from scipy import stats
from sklearn.preprocessing import StandardScaler


class EvaluateRegressionModelInput(BaseModel):
    model_key: str = Field(default="linear_regression", description="Key of the ML model results to evaluate")
    include_assumptions: bool = Field(default=True, description="Whether to check regression assumptions")
    confidence_level: float = Field(default=0.95, description="Confidence level for statistical tests")


class EvaluateRegressionModelTool(BaseTool):
    name: str = "evaluate_regression_model"
    description: str = "Comprehensive evaluation of regression models including metrics, assumptions, and interpretation."
    args_schema: Type[BaseModel] = EvaluateRegressionModelInput

    def _run(self, model_key: str = "linear_regression", include_assumptions: bool = True, 
             confidence_level: float = 0.95) -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a regression analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            
            # Extract data
            y_train = results['y_train']
            y_train_pred = results['y_train_pred']
            y_test = results['y_test']
            y_test_pred = results['y_test_pred']
            target_column = results['target_column']
            feature_columns = results['feature_columns']
            model = results.get('model')
            
            # Calculate comprehensive metrics
            evaluation = self._calculate_regression_metrics(
                y_train, y_train_pred, y_test, y_test_pred
            )
            
            # Check assumptions if requested
            if include_assumptions and model is not None:
                X_train = results['X_train']
                residuals = results.get('residuals', y_train - y_train_pred)
                
                assumptions = self._check_regression_assumptions(
                    X_train, y_train, y_train_pred, residuals, confidence_level
                )
                evaluation['assumptions'] = assumptions
            
            # Generate interpretation
            interpretation = self._create_regression_interpretation(
                evaluation, target_column, feature_columns, confidence_level
            )
            
            # Update session state with evaluation results
            results['evaluation'] = evaluation
            st.session_state.ml_model_results[model_key] = results
            
            return interpretation
            
        except Exception as e:
            return f"❌ Error evaluating regression model: {str(e)}"

    def _calculate_regression_metrics(self, y_train: pd.Series, y_train_pred: np.ndarray,
                                    y_test: pd.Series, y_test_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive regression metrics."""
        
        metrics = {}
        
        # Training metrics
        metrics['train'] = {
            'r2': r2_score(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'n_observations': len(y_train)
        }
        
        # Test metrics
        metrics['test'] = {
            'r2': r2_score(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'n_observations': len(y_test)
        }
        
        # Model performance assessment
        metrics['performance'] = {
            'overfitting_risk': abs(metrics['train']['r2'] - metrics['test']['r2']),
            'quality': self._assess_model_quality(metrics['train']['r2']),
            'generalization': self._assess_regression_generalization(metrics['train']['r2'], metrics['test']['r2'])
        }
        
        # Cross-validation equivalent (using train-test difference as proxy)
        metrics['validation'] = {
            'r2_difference': metrics['train']['r2'] - metrics['test']['r2'],
            'rmse_ratio': metrics['test']['rmse'] / metrics['train']['rmse'],
            'stable': abs(metrics['train']['r2'] - metrics['test']['r2']) < 0.1
        }
        
        return metrics
    
    def _check_regression_assumptions(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    y_pred: np.ndarray, residuals: np.ndarray,
                                    confidence_level: float) -> Dict[str, Any]:
        """Check regression model assumptions comprehensively."""
        
        assumptions = {}
        alpha = 1 - confidence_level
        
        try:
            # 1. Linearity (correlation between actual and predicted)
            linearity_corr = np.corrcoef(y_train, y_pred)[0, 1]
            assumptions["linearity"] = {
                "correlation": round(linearity_corr, 4),
                "status": "Good" if linearity_corr > 0.8 else 
                         "Moderate" if linearity_corr > 0.6 else "Poor",
                "interpretation": f"Linear relationship strength: {linearity_corr:.3f}"
            }
            
            # 2. Normality of residuals (Shapiro-Wilk test)
            if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limitations
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                assumptions["normality"] = {
                    "shapiro_statistic": round(shapiro_stat, 4),
                    "p_value": round(shapiro_p, 4),
                    "status": "Good" if shapiro_p > alpha else "Violated",
                    "interpretation": f"Residuals {'appear' if shapiro_p > alpha else 'do not appear'} normally distributed (p={shapiro_p:.4f})"
                }
            else:
                # Use alternative test for large samples
                assumptions["normality"] = {
                    "status": "Check required",
                    "interpretation": "Sample too large for Shapiro-Wilk test. Visual inspection recommended."
                }
            
            # 3. Homoscedasticity (Breusch-Pagan test approximation)
            # Simple version: correlation between absolute residuals and predicted values
            abs_residuals = np.abs(residuals)
            homo_corr = np.corrcoef(y_pred, abs_residuals)[0, 1]
            
            assumptions["homoscedasticity"] = {
                "correlation": round(homo_corr, 4),
                "status": "Good" if abs(homo_corr) < 0.3 else 
                         "Moderate" if abs(homo_corr) < 0.5 else "Violated",
                "interpretation": f"Constant variance {'maintained' if abs(homo_corr) < 0.3 else 'questionable'} (correlation: {homo_corr:.3f})"
            }
            
            # 4. Independence (Durbin-Watson approximation)
            # Simple autocorrelation check on residuals
            if len(residuals) > 1:
                residuals_lag1 = residuals[1:]
                residuals_current = residuals[:-1]
                autocorr = np.corrcoef(residuals_current, residuals_lag1)[0, 1]
                
                assumptions["independence"] = {
                    "autocorrelation": round(autocorr, 4),
                    "status": "Good" if abs(autocorr) < 0.3 else 
                             "Moderate" if abs(autocorr) < 0.5 else "Violated",
                    "interpretation": f"Residual independence {'satisfied' if abs(autocorr) < 0.3 else 'questionable'} (autocorr: {autocorr:.3f})"
                }
            
            # 5. Multicollinearity (VIF approximation using correlations)
            if len(X_train.columns) > 1:
                corr_matrix = X_train.corr()
                max_corr = 0
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > max_corr:
                            max_corr = corr_val
                        if corr_val > 0.8:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                assumptions["multicollinearity"] = {
                    "max_correlation": round(max_corr, 4),
                    "high_correlations": len(high_corr_pairs),
                    "status": "Good" if max_corr < 0.8 else 
                             "Moderate" if max_corr < 0.9 else "High",
                    "interpretation": f"Multicollinearity {'low' if max_corr < 0.8 else 'concerning'} (max correlation: {max_corr:.3f})"
                }
            
        except Exception as e:
            assumptions["error"] = f"Could not check all assumptions: {str(e)}"
            
        return assumptions
    
    def _assess_model_quality(self, r2: float) -> str:
        """Assess model quality based on R-squared."""
        if r2 > 0.9:
            return "Excellent"
        elif r2 > 0.8:
            return "Very Good"
        elif r2 > 0.7:
            return "Good"
        elif r2 > 0.5:
            return "Moderate"
        elif r2 > 0.3:
            return "Weak"
        else:
            return "Poor"
    
    def _assess_regression_generalization(self, train_r2: float, test_r2: float) -> str:
        """Assess regression model generalization capability."""
        diff = abs(train_r2 - test_r2)
        
        if diff < 0.05:
            return "Excellent generalization"
        elif diff < 0.1:
            return "Good generalization"
        elif diff < 0.15:
            return "Moderate generalization"
        else:
            return "Poor generalization (overfitting risk)"
    
    def _create_regression_interpretation(self, evaluation: Dict[str, Any], 
                                        target_column: str, feature_columns: List[str],
                                        confidence_level: float) -> str:
        """Create comprehensive business interpretation."""
        
        train_metrics = evaluation['train']
        test_metrics = evaluation['test']
        performance = evaluation['performance']
        
        interpretation = [
            f"🔍 **Comprehensive Regression Model Evaluation: {target_column}**\n",
            f"**Model Performance Analysis:**",
            f"  • **Training Performance:** R² = {train_metrics['r2']:.4f} ({train_metrics['r2']*100:.1f}% variance explained)",
            f"  • **Test Performance:** R² = {test_metrics['r2']:.4f} ({test_metrics['r2']*100:.1f}% variance explained)",
            f"  • **Model Quality:** {performance['quality']}",
            f"  • **Generalization:** {performance['generalization']}",
            f"",
            f"📊 **Detailed Metrics:**",
            f"  • **Training RMSE:** {train_metrics['rmse']:.4f}",
            f"  • **Test RMSE:** {test_metrics['rmse']:.4f}",
            f"  • **Training MAE:** {train_metrics['mae']:.4f}",
            f"  • **Test MAE:** {test_metrics['mae']:.4f}",
            f"  • **Overfitting Risk:** {performance['overfitting_risk']:.4f}",
            f""
        ]
        
        # Add warnings
        if performance['overfitting_risk'] > 0.1:
            interpretation.append("⚠️ **Warning:** Significant difference between training and test performance suggests overfitting")
        
        if test_metrics['r2'] < 0.5:
            interpretation.append("⚠️ **Warning:** Low test R² suggests limited predictive power")
        
        # Assumptions check
        if 'assumptions' in evaluation:
            assumptions = evaluation['assumptions']
            interpretation.extend([
                f"",
                f"✅ **Model Assumptions Check (α = {1-confidence_level:.2f}):**"
            ])
            
            for assumption, details in assumptions.items():
                if assumption != 'error' and isinstance(details, dict):
                    status = details.get('status', 'Unknown')
                    interp = details.get('interpretation', '')
                    interpretation.append(f"  • **{assumption.title()}:** {status} - {interp}")
        
        interpretation.extend([
            f"",
            f"💡 **Business Recommendations:**",
            f"  • Focus on features with significant coefficients for decision-making",
            f"  • Consider model assumptions when interpreting results",
            f"  • Use visualization tools to validate model performance",
            f"  • Monitor model performance on new data for drift detection"
        ])
        
        return "\n".join(interpretation)

    def _arun(self, model_key: str = "linear_regression", include_assumptions: bool = True, 
              confidence_level: float = 0.95):
        raise NotImplementedError("Async not supported")


class EvaluateClassificationModelInput(BaseModel):
    model_key: str = Field(default="logistic_regression", description="Key of the ML model results to evaluate")
    average: str = Field(default="weighted", description="Averaging strategy for multi-class metrics")
    include_probabilities: bool = Field(default=True, description="Whether to evaluate probability predictions")


class EvaluateClassificationModelTool(BaseTool):
    name: str = "evaluate_classification_model"
    description: str = "Comprehensive evaluation of classification models including metrics, confusion matrix, and interpretation."
    args_schema: Type[BaseModel] = EvaluateClassificationModelInput

    def _run(self, model_key: str = "logistic_regression", average: str = "weighted", 
             include_probabilities: bool = True) -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a classification analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            
            # Extract data
            y_train = results['y_train']
            y_train_pred = results['y_train_pred']
            y_test = results['y_test']
            y_test_pred = results['y_test_pred']
            target_column = results['target_column']
            feature_columns = results['feature_columns']
            
            # Get probabilities if available
            y_train_proba = results.get('y_train_proba')
            y_test_proba = results.get('y_test_proba')
            
            # Calculate comprehensive metrics
            evaluation = self._calculate_classification_metrics(
                y_train, y_train_pred, y_test, y_test_pred,
                y_train_proba, y_test_proba, average, include_probabilities
            )
            
            # Generate interpretation
            interpretation = self._create_classification_interpretation(
                evaluation, target_column, feature_columns, average
            )
            
            # Update session state with evaluation results
            results['evaluation'] = evaluation
            st.session_state.ml_model_results[model_key] = results
            
            return interpretation
            
        except Exception as e:
            return f"❌ Error evaluating classification model: {str(e)}"

    def _calculate_classification_metrics(self, y_train: pd.Series, y_train_pred: np.ndarray,
                                        y_test: pd.Series, y_test_pred: np.ndarray,
                                        y_train_proba: Optional[np.ndarray] = None,
                                        y_test_proba: Optional[np.ndarray] = None,
                                        average: str = "weighted",
                                        include_probabilities: bool = True) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        
        metrics = {}
        
        # Training metrics
        metrics['train'] = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average=average, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average=average, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, average=average, zero_division=0),
            'n_observations': len(y_train)
        }
        
        # Test metrics
        metrics['test'] = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, average=average, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, average=average, zero_division=0),
            'n_observations': len(y_test)
        }
        
        # Confusion matrices
        metrics['confusion_matrix'] = {
            'train': confusion_matrix(y_train, y_train_pred),
            'test': confusion_matrix(y_test, y_test_pred)
        }
        
        # ROC AUC if probabilities available and binary classification
        if include_probabilities and y_test_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:  # Binary classification
                    if y_test_proba.ndim == 2 and y_test_proba.shape[1] == 2:
                        # Use probability of positive class
                        metrics['train']['roc_auc'] = roc_auc_score(y_train, y_train_proba[:len(y_train), 1])
                        metrics['test']['roc_auc'] = roc_auc_score(y_test, y_test_proba[:len(y_test), 1])
                    else:
                        # Single column probability
                        metrics['train']['roc_auc'] = roc_auc_score(y_train, y_train_proba[:len(y_train)])
                        metrics['test']['roc_auc'] = roc_auc_score(y_test, y_test_proba[:len(y_test)])
                else:
                    # Multi-class ROC AUC
                    metrics['train']['roc_auc'] = roc_auc_score(y_train, y_train_proba[:len(y_train)], 
                                                              multi_class='ovr', average=average)
                    metrics['test']['roc_auc'] = roc_auc_score(y_test, y_test_proba[:len(y_test)], 
                                                             multi_class='ovr', average=average)
            except Exception:
                # ROC AUC calculation failed, skip it
                pass
        
        # Model performance assessment
        metrics['performance'] = {
            'overfitting_risk': abs(metrics['train']['accuracy'] - metrics['test']['accuracy']),
            'quality': self._assess_classification_quality(metrics['test']['accuracy']),
            'generalization': self._assess_classification_generalization(
                metrics['train']['accuracy'], metrics['test']['accuracy']
            )
        }
        
        return metrics
    
    def _assess_classification_quality(self, accuracy: float) -> str:
        """Assess classification model quality based on accuracy."""
        if accuracy > 0.95:
            return "Excellent"
        elif accuracy > 0.9:
            return "Very Good"
        elif accuracy > 0.8:
            return "Good"
        elif accuracy > 0.7:
            return "Moderate"
        elif accuracy > 0.6:
            return "Weak"
        else:
            return "Poor"
    
    def _assess_classification_generalization(self, train_acc: float, test_acc: float) -> str:
        """Assess classification model generalization capability."""
        diff = abs(train_acc - test_acc)
        
        if diff < 0.02:
            return "Excellent generalization"
        elif diff < 0.05:
            return "Good generalization"
        elif diff < 0.1:
            return "Moderate generalization"
        else:
            return "Poor generalization (overfitting risk)"
    
    def _create_classification_interpretation(self, evaluation: Dict[str, Any], 
                                            target_column: str, feature_columns: List[str],
                                            average: str) -> str:
        """Create comprehensive business interpretation for classification."""
        
        train_metrics = evaluation['train']
        test_metrics = evaluation['test']
        performance = evaluation['performance']
        
        interpretation = [
            f"🎯 **Comprehensive Classification Model Evaluation: {target_column}**\n",
            f"**Model Performance Analysis:**",
            f"  • **Training Accuracy:** {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.1f}%)",
            f"  • **Test Accuracy:** {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.1f}%)",
            f"  • **Model Quality:** {performance['quality']}",
            f"  • **Generalization:** {performance['generalization']}",
            f"",
            f"📊 **Detailed Metrics ({average} average):**",
            f"  • **Precision:** Train={train_metrics['precision']:.4f}, Test={test_metrics['precision']:.4f}",
            f"  • **Recall:** Train={train_metrics['recall']:.4f}, Test={test_metrics['recall']:.4f}",
            f"  • **F1-Score:** Train={train_metrics['f1']:.4f}, Test={test_metrics['f1']:.4f}",
        ]
        
        # Add ROC AUC if available
        if 'roc_auc' in test_metrics:
            interpretation.append(f"  • **ROC AUC:** Train={train_metrics['roc_auc']:.4f}, Test={test_metrics['roc_auc']:.4f}")
        
        interpretation.extend([
            f"  • **Overfitting Risk:** {performance['overfitting_risk']:.4f}",
            f""
        ])
        
        # Add warnings
        if performance['overfitting_risk'] > 0.05:
            interpretation.append("⚠️ **Warning:** Significant difference between training and test performance suggests overfitting")
        
        if test_metrics['accuracy'] < 0.7:
            interpretation.append("⚠️ **Warning:** Low test accuracy suggests limited predictive power")
        
        interpretation.extend([
            f"",
            f"💡 **Business Recommendations:**",
            f"  • Use confusion matrix to understand prediction patterns",
            f"  • Focus on precision vs recall trade-offs based on business needs",
            f"  • Consider feature importance for model interpretability",
            f"  • Monitor model performance on new data for drift detection"
        ])
        
        return "\n".join(interpretation)

    def _arun(self, model_key: str = "logistic_regression", average: str = "weighted", 
              include_probabilities: bool = True):
        raise NotImplementedError("Async not supported")
