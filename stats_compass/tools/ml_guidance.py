"""
Shared Intelligence Layer for ML Tools

This module provides reusable functionality that ML tools can inherit.
Centralizes:
- Dataset feature analysis
- Model quality assessment (regression & classification)
- Workflow suggestion generation
- Output formatting

Usage:
    class RunLinearRegressionTool(BaseTool, SmartMLToolMixin):
        def _run(self, target, features=None, ...):
            # 1. Analyze dataset
            context = self._analyze_ml_features(df, target, features)
            
            # 2. Train model (tool-specific)
            model, metrics = self._train_model(...)
            
            # 3. Assess quality
            quality = self._assess_model_quality(metrics, 'regression', context)
            
            # 4. Generate suggestions
            suggestions = self._generate_ml_suggestions(quality, context, target, features)
            
            # 5. Format output
            return self._format_ml_output(results, quality, suggestions)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st

from stats_compass.utils.workflow_state import (
    get_workflow_state, update_workflow_state
)


# ============================================
# Quality Level Constants
# ============================================

QUALITY_LEVELS = {
    'poor': {
        'label': '❌ Poor',
        'description': 'Not suitable for decision-making',
        'regression_r2_threshold': 0.3,
        'classification_auc_threshold': 0.6,
    },
    'moderate': {
        'label': '⚠️ Moderate',
        'description': 'Suitable for exploratory analysis, use caution for decisions',
        'regression_r2_threshold': 0.5,
        'classification_auc_threshold': 0.7,
    },
    'good': {
        'label': '✅ Good',
        'description': 'Suitable for decision-making with monitoring',
        'regression_r2_threshold': 0.7,
        'classification_auc_threshold': 0.8,
    },
    'excellent': {
        'label': '🎯 Excellent',
        'description': 'Ready for production use',
        'regression_r2_threshold': float('inf'),
        'classification_auc_threshold': float('inf'),
    },
}


def assess_regression_quality(r2: float) -> str:
    """
    Assess regression model quality based on R².
    
    Args:
        r2: R-squared value (0 to 1)
        
    Returns:
        Quality level: 'poor', 'moderate', 'good', or 'excellent'
        
    Example:
        >>> quality = assess_regression_quality(0.73)
        >>> print(quality)  # 'good'
    """
    if r2 < 0.3:
        return 'poor'
    elif r2 < 0.5:
        return 'moderate'
    elif r2 < 0.7:
        return 'good'
    else:
        return 'excellent'


def assess_classification_quality(auc: float) -> str:
    """
    Assess classification model quality based on AUC.
    
    Args:
        auc: Area Under ROC Curve (0 to 1)
        
    Returns:
        Quality level: 'poor', 'moderate', 'good', or 'excellent'
        
    Example:
        >>> quality = assess_classification_quality(0.82)
        >>> print(quality)  # 'excellent'
    """
    if auc < 0.6:
        return 'poor'
    elif auc < 0.7:
        return 'moderate'
    elif auc < 0.8:
        return 'good'
    else:
        return 'excellent'


# ============================================
# Priority Level Constants for Suggestions
# ============================================

PRIORITY_CRITICAL = '🔴 CRITICAL'
PRIORITY_RECOMMENDED = '🟡 RECOMMENDED'
PRIORITY_OPTIONAL = '🟢 OPTIONAL'


def format_suggestion(priority: str, action: str, reason: str = '') -> str:
    """
    Format a suggestion with priority indicator.
    
    Args:
        priority: PRIORITY_CRITICAL, PRIORITY_RECOMMENDED, or PRIORITY_OPTIONAL
        action: The action to take (should include tool name and parameters)
        reason: Optional explanation of why this is suggested
        
    Returns:
        Formatted suggestion string
        
    Example:
        >>> suggestion = format_suggestion(
        >>>     PRIORITY_CRITICAL,
        >>>     "Apply mean_target_encoding to ['Genre', 'Language']",
        >>>     "will significantly improve R²"
        >>> )
        >>> print(suggestion)
        # "🔴 CRITICAL: Apply mean_target_encoding to ['Genre', 'Language'] (will significantly improve R²)"
    """
    if reason:
        return f"{priority}: {action} ({reason})"
    else:
        return f"{priority}: {action}"


class SmartMLToolMixin:
    """
    Mixin class providing smart workflow guidance for ML tools.
    
    Shares:
    - Data preparation logic
    - Feature analysis logic
    - Quality assessment logic
    - Suggestion generation logic
    - Output formatting logic
    
    """
    
    # ============================================
    # DATA PREPARATION (Shared across tools)
    # ============================================
    
    def _prepare_regression_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]],
        allow_non_numeric_target: bool = False
    ) -> tuple:
        """
        Shared helper for preparing and validating regression data.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            feature_columns: List of feature columns (None = auto-select numeric)
            allow_non_numeric_target: Whether to allow non-numeric targets (for classification)
            
        Returns:
            Tuple of (X, y, feature_columns, error_message)
            If error occurs, returns (None, None, None, error_string)
        """
        # Validate target column exists
        if target_column not in df.columns:
            return None, None, None, f"❌ Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}"
        
        # Validate target is numeric (for regression only)
        if not allow_non_numeric_target and not pd.api.types.is_numeric_dtype(df[target_column]):
            return None, None, None, f"❌ Target column '{target_column}' must be numeric for regression"
        
        # Auto-select features if not provided
        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]
        
        if not feature_columns:
            return None, None, None, "❌ No numeric feature columns available for regression"
        
        # Check for missing feature columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return None, None, None, f"❌ Feature columns not found: {missing_cols}"
        
        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing values
        missing_mask = X.isnull().any(axis=1) | y.isnull()
        if missing_mask.sum() > 0:
            X = X[~missing_mask]
            y = y[~missing_mask]
        
        return X, y, feature_columns, None
    
    # ============================================
    # FEATURE ANALYSIS (Universal for all ML tools)
    # ============================================
    
    def _analyze_ml_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        requested_features: Optional[List[str]],
        auto_include_encoded: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze dataset to identify feature engineering opportunities.
        
        Works for regression, classification, and time series models.
        
        Args:
            df: Input dataframe
            target_column: Target variable
            requested_features: Features user requested (None = auto-select)
            auto_include_encoded: Whether to auto-include previously encoded columns
            
        Returns:
            Dictionary with:
            - unused_categorical: List[str] - Categorical columns not in features
            - unused_numeric: List[str] - Numeric columns not in features
            - unused_datetime: List[str] - Datetime columns not in features
            - total_columns: int - Total columns available
            - feature_count: int - Number of features that will be used
            - has_unused_features: bool - Whether there are unused columns
            - auto_included_encoded: List[str] - Encoded features auto-included
            - missing_removed: int - Rows removed due to missing values (set by caller)
        """
        context = {
            'unused_categorical': [],
            'unused_numeric': [],
            'unused_datetime': [],
            'total_columns': len(df.columns) - 1,  # Exclude target
            'auto_included_encoded': [],
            'missing_removed': 0,  # Will be set by caller after data prep
        }
        
        all_columns = set(df.columns) - {target_column}
        
        # Determine which features will be used
        if requested_features:
            used_features = set(requested_features)
        else:
            # Check workflow state for encoded features
            workflow_state = get_workflow_state()
            encoded_cols = workflow_state.get('available_encoded_columns', [])
            available_encoded = [col for col in encoded_cols if col in df.columns]
            
            # Auto-include encoded columns if available and enabled
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != target_column]
            
            if auto_include_encoded and available_encoded:
                used_features = set(numeric_cols + available_encoded)
                context['auto_included_encoded'] = available_encoded
            else:
                used_features = set(numeric_cols)
        
        unused_columns = all_columns - used_features
        
        # Categorize unused columns by type
        # Note: Skip categorical columns if their encoded version was used
        for col in unused_columns:
            dtype = df[col].dtype
            if dtype == 'object' or dtype.name == 'category':
                # Check if encoded version of this categorical column was used
                encoded_version = f"{col}_encoded"
                if encoded_version not in used_features:
                    # Only suggest encoding if encoded version wasn't already used
                    context['unused_categorical'].append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                context['unused_numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                context['unused_datetime'].append(col)
        
        context['has_unused_features'] = len(unused_columns) > 0
        context['feature_count'] = len(used_features)
        
        return context
    
    # ============================================
    # QUALITY ASSESSMENT (Metric-agnostic)
    # ============================================
    
    def _assess_model_quality(
        self,
        metrics: Dict[str, float],
        model_type: str,
        dataset_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess model quality and generate contextual warnings.
        
        Polymorphic - works for regression (R²) or classification (AUC).
        
        Args:
            metrics: Dictionary with model metrics
                - For regression: {'train_r2', 'test_r2', 'train_rmse', 'test_rmse'}
                - For classification: {'train_auc', 'test_auc', 'train_acc', 'test_acc'}
            model_type: 'regression' or 'classification'
            dataset_context: Output from _analyze_ml_features()
            
        Returns:
            Dictionary with:
            - quality_level: str - 'poor', 'moderate', 'good', 'excellent'
            - quality_label: str - Formatted label with emoji
            - warnings: List[str] - Warning messages
            - metrics: Dict - Copy of input metrics
        """
        assessment = {
            'warnings': [],
            'metrics': metrics.copy()
        }
        
        # Determine quality level based on model type
        if model_type == 'regression':
            test_r2 = metrics.get('test_r2', 0)
            train_r2 = metrics.get('train_r2', 0)
            assessment['quality_level'] = assess_regression_quality(test_r2)
            primary_metric = test_r2
            metric_name = 'R²'
            
            # Quality-based warnings
            if test_r2 < 0.1:
                assessment['warnings'].append(
                    f"Very low {metric_name} ({test_r2:.3f}) - model explains <10% of variance, essentially no predictive power"
                )
            elif test_r2 < 0.3:
                assessment['warnings'].append(
                    f"Low {metric_name} ({test_r2:.3f}) - model has weak predictive power"
                )
            
            # Check for overfitting
            if train_r2 - test_r2 > 0.15:
                assessment['warnings'].append(
                    f"Possible overfitting: train {metric_name}={train_r2:.3f}, test {metric_name}={test_r2:.3f} (gap={train_r2-test_r2:.3f})"
                )
        
        elif model_type == 'classification':
            test_auc = metrics.get('test_auc', 0.5)
            train_auc = metrics.get('train_auc', 0.5)
            assessment['quality_level'] = assess_classification_quality(test_auc)
            primary_metric = test_auc
            metric_name = 'AUC'
            
            # Quality-based warnings
            if test_auc < 0.6:
                assessment['warnings'].append(
                    f"Very low {metric_name} ({test_auc:.3f}) - model barely better than random guessing (0.5)"
                )
            elif test_auc < 0.7:
                assessment['warnings'].append(
                    f"Low {metric_name} ({test_auc:.3f}) - model has weak discriminative power"
                )
            
            # Check for overfitting
            if train_auc - test_auc > 0.10:
                assessment['warnings'].append(
                    f"Possible overfitting: train {metric_name}={train_auc:.3f}, test {metric_name}={test_auc:.3f} (gap={train_auc-test_auc:.3f})"
                )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'regression' or 'classification'")
        
        # Add quality label
        quality_labels = {
            'poor': '❌ Poor',
            'moderate': '⚠️ Moderate',
            'good': '✅ Good',
            'excellent': '🎯 Excellent'
        }
        assessment['quality_label'] = quality_labels[assessment['quality_level']]
        
        # Feature count warnings
        n_features = dataset_context.get('feature_count', 0)
        if n_features == 1:
            assessment['warnings'].append(
                f"Only 1 feature used - model may be missing important predictors"
            )
        elif n_features < 3 and dataset_context['total_columns'] > 5:
            assessment['warnings'].append(
                f"Only {n_features} features used out of {dataset_context['total_columns']} available columns"
            )
        
        # Unused feature warnings
        if dataset_context.get('unused_categorical'):
            n_unused = len(dataset_context['unused_categorical'])
            categorical_list = ', '.join(dataset_context['unused_categorical'][:5])
            if n_unused > 5:
                categorical_list += f', ... ({n_unused-5} more)'
            
            assessment['warnings'].append(
                f"{n_unused} unused categorical column{'s' if n_unused > 1 else ''}: {categorical_list}"
            )
        
        if dataset_context.get('unused_numeric') and len(dataset_context['unused_numeric']) > 0:
            n_unused_numeric = len(dataset_context['unused_numeric'])
            if n_unused_numeric <= 3:
                assessment['warnings'].append(
                    f"{n_unused_numeric} unused numeric column{'s' if n_unused_numeric > 1 else ''}: {', '.join(dataset_context['unused_numeric'])}"
                )
            else:
                assessment['warnings'].append(
                    f"{n_unused_numeric} unused numeric columns: {', '.join(dataset_context['unused_numeric'][:3])}, ..."
                )
        
        return assessment
    
    # ============================================
    # SUGGESTION GENERATION (Universal workflow patterns)
    # ============================================
    
    def _generate_ml_suggestions(
        self,
        quality_assessment: Dict[str, Any],
        dataset_context: Dict[str, Any],
        target_column: str,
        current_features: List[str],
        model_type: str = 'regression'
    ) -> List[str]:
        """
        Generate prioritized workflow suggestions based on quality and dataset.
        
        Works for regression and classification models.
        
        Args:
            quality_assessment: Output from _assess_model_quality()
            dataset_context: Output from _analyze_ml_features()
            target_column: Target variable name
            current_features: List of features currently used
            model_type: 'regression' or 'classification'
            
        Returns:
            List of formatted suggestions with priority indicators (max 5)
        """
        suggestions = []
        quality = quality_assessment['quality_level']
        
        # ============================================
        # Priority 1 (CRITICAL): Categorical encoding for poor/moderate quality
        # ============================================
        if quality in ['poor', 'moderate'] and dataset_context.get('unused_categorical'):
            categorical_cols = dataset_context['unused_categorical']
            suggestion = format_suggestion(
                PRIORITY_CRITICAL,
                f"Apply mean_target_encoding to {categorical_cols} with target_column='{target_column}', then rerun {model_type}",
                "categorical features often contain crucial predictive information and will significantly improve model performance"
            )
            suggestions.append(suggestion)
        
        # ============================================
        # Priority 2 (CRITICAL): Class imbalance for classification
        # ============================================
        if model_type == 'classification' and quality in ['poor', 'moderate']:
            # Check if class_weight parameter is available in workflow state
            workflow_state = get_workflow_state()
            if not workflow_state.get('class_balance_addressed'):
                suggestion = format_suggestion(
                    PRIORITY_CRITICAL,
                    "Check class balance with df[target].value_counts(), then rerun with class_weight='balanced' if imbalanced",
                    "class imbalance can severely hurt model performance on minority class"
                )
                suggestions.append(suggestion)
        
        # ============================================
        # Priority 3 (RECOMMENDED): Missing data recovery
        # ============================================
        if quality in ['poor', 'moderate'] and dataset_context.get('missing_removed', 0) > 50:
            suggestion = format_suggestion(
                PRIORITY_RECOMMENDED,
                f"Run analyze_missing_data to check if imputation could recover {dataset_context['missing_removed']} removed rows",
                "more data may improve model performance"
            )
            suggestions.append(suggestion)
        
        # ============================================
        # Priority 4 (RECOMMENDED): Feature scaling
        # ============================================
        if len(current_features) > 1 and quality in ['poor', 'moderate']:
            workflow_state = get_workflow_state()
            if not workflow_state.get('features_standardized'):
                suggestion = format_suggestion(
                    PRIORITY_RECOMMENDED,
                    "Rerun with standardize_features=True to normalize feature scales",
                    "standardization improves coefficient interpretation and model stability"
                )
                suggestions.append(suggestion)
        
        # ============================================
        # Priority 5 (RECOMMENDED): Outlier handling
        # ============================================
        workflow_state = get_workflow_state()
        if quality in ['poor', 'moderate'] and workflow_state.get('outliers_detected'):
            outlier_cols = workflow_state.get('outlier_columns', [])
            if outlier_cols:
                suggestion = format_suggestion(
                    PRIORITY_RECOMMENDED,
                    f"Consider handling outliers in {outlier_cols[:3]} which may be affecting model fit",
                    "outliers can disproportionately influence model coefficients"
                )
                suggestions.append(suggestion)
        
        # ============================================
        # Priority 6 (RECOMMENDED): Model evaluation for good quality
        # ============================================
        if quality in ['moderate', 'good', 'excellent']:
            eval_tool = 'evaluate_regression_model' if model_type == 'regression' else 'evaluate_classification_model'
            suggestion = format_suggestion(
                PRIORITY_RECOMMENDED,
                f"Use {eval_tool} for comprehensive diagnostics and assumption validation",
                "validates model assumptions before making decisions"
            )
            suggestions.append(suggestion)
        
        # ============================================
        # Priority 7 (OPTIONAL): Visualization
        # ============================================
        if model_type == 'regression':
            viz_tools = "create_regression_plot, create_residual_plot, and create_coefficient_chart"
        else:
            viz_tools = "create_roc_curve, create_confusion_matrix, and create_feature_importance_chart"
        
        suggestion = format_suggestion(
            PRIORITY_OPTIONAL,
            f"Create visualizations with {viz_tools}",
            "visual inspection reveals patterns metrics may miss"
        )
        suggestions.append(suggestion)
        
        return suggestions[:5]  # Cap at 5 to control token usage
    
    # ============================================
    # OUTPUT FORMATTING (Consistent format across tools)
    # ============================================
    
    def _format_ml_output(
        self,
        core_results: str,
        quality_assessment: Dict[str, Any],
        suggestions: List[str],
        auto_included_note: Optional[str] = None
    ) -> str:
        """
        Format ML tool output with consistent structure.
        
        Args:
            core_results: Main results string (model performance, coefficients, etc.)
            quality_assessment: Output from _assess_model_quality()
            suggestions: Output from _generate_ml_suggestions()
            auto_included_note: Optional note about auto-included features
            
        Returns:
            Formatted output string with:
            - Core results
            - Auto-included features note (if applicable)
            - Quality warnings (if issues exist)
            - Suggested next steps (if suggestions exist)
        """
        output_lines = [core_results]
        
        # Add auto-included features note
        if auto_included_note:
            output_lines.extend(['', auto_included_note])
        
        # Add quality warnings (conditional)
        if quality_assessment.get('warnings'):
            output_lines.extend([
                '',
                f"⚠️ **Model Quality Assessment ({quality_assessment['quality_label']}):**"
            ])
            for warning in quality_assessment['warnings']:
                output_lines.append(f"  • {warning}")
        
        # Add suggestions (conditional)
        if suggestions:
            output_lines.extend([
                '',
                '💡 **Suggested Next Steps:**'
            ])
            for i, suggestion in enumerate(suggestions, 1):
                output_lines.append(f"  {i}. {suggestion}")
        
        return '\n'.join(output_lines)
    
    # ============================================
    # WORKFLOW METADATA (State tracking)
    # ============================================
    
    def _update_ml_workflow_metadata(
        self,
        model_type: str,
        model_quality: str,
        features_used: List[str],
        model_key: str,
        target_column: str,
        primary_metric: float
    ):
        """
        Update workflow metadata for tool coordination.
        
        Args:
            model_type: 'regression' or 'classification'
            model_quality: 'poor', 'moderate', 'good', 'excellent'
            features_used: List of feature column names
            model_key: Unique model identifier
            target_column: Target variable name
            primary_metric: Main quality metric (R² or AUC)
        """
        update_workflow_state({
            'last_model_type': model_type,
            'last_model_quality': model_quality,
            'last_model_features': features_used,
            'last_model_key': model_key,
            'last_model_target': target_column,
        })
        
        # Add to models trained list
        workflow_state = get_workflow_state()
        models_trained = workflow_state.get('models_trained', [])
        models_trained.append(model_key)
        update_workflow_state({'models_trained': models_trained})
