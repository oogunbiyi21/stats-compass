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

# Import centralized ML quality constants
from stats_compass.constants import (
    QUALITY_LEVELS,
    QUALITY_POOR,
    QUALITY_MODERATE,
    QUALITY_GOOD,
    QUALITY_EXCELLENT,
    R2_POOR_THRESHOLD,
    R2_MODERATE_THRESHOLD,
    R2_GOOD_THRESHOLD,
    AUC_POOR_THRESHOLD,
    AUC_MODERATE_THRESHOLD,
    AUC_GOOD_THRESHOLD,
    OVERFITTING_GAP_THRESHOLD_REGRESSION,
    OVERFITTING_GAP_THRESHOLD_CLASSIFICATION,
    MIN_SAMPLES_PER_FEATURE_SEVERE,
    MIN_SAMPLES_PER_FEATURE_MODERATE,
    SAMPLE_SIZE_PENALTY_SEVERE,
    SAMPLE_SIZE_PENALTY_MODERATE,
    MISSING_DATA_RECOVERY_THRESHOLD,
    PRIORITY_CRITICAL,
    PRIORITY_RECOMMENDED,
    PRIORITY_OPTIONAL,
    PRIORITY_CRITICAL_INT,
    PRIORITY_RECOMMENDED_INT,
    PRIORITY_OPTIONAL_INT,
)


def assess_regression_quality(r2: float, n_samples: Optional[int] = None, n_features: Optional[int] = None) -> str:
    """
    Assess regression model quality based on R² with sample size adjustment.
    
    Args:
        r2: R-squared value (0 to 1)
        n_samples: Number of training samples (optional, for overfit detection)
        n_features: Number of features used (optional, for overfit detection)
        
    Returns:
        Quality level: 'poor', 'moderate', 'good', or 'excellent'
        
    Example:
        >>> quality = assess_regression_quality(0.73)
        >>> print(quality)  # 'good'
        
        >>> quality = assess_regression_quality(0.85, n_samples=20, n_features=10)
        >>> print(quality)  # 'moderate' (penalized for overfit risk)
    """
    # Apply sample size penalty if both n_samples and n_features provided
    r2_adjusted = r2
    if n_samples is not None and n_features is not None and n_features > 0:
        samples_per_feature = n_samples / n_features
        if samples_per_feature < MIN_SAMPLES_PER_FEATURE_SEVERE:
            # Severe overfit risk - heavy penalty
            r2_adjusted = r2 - SAMPLE_SIZE_PENALTY_SEVERE
        elif samples_per_feature < MIN_SAMPLES_PER_FEATURE_MODERATE:
            # Moderate overfit risk - light penalty
            r2_adjusted = r2 - SAMPLE_SIZE_PENALTY_MODERATE
    
    # Apply quality thresholds to adjusted R²
    if r2_adjusted < R2_POOR_THRESHOLD:
        return QUALITY_POOR
    elif r2_adjusted < R2_MODERATE_THRESHOLD:
        return QUALITY_MODERATE
    elif r2_adjusted < R2_GOOD_THRESHOLD:
        return QUALITY_GOOD
    else:
        return QUALITY_EXCELLENT


def assess_classification_quality(auc: float, n_samples: Optional[int] = None, n_features: Optional[int] = None) -> str:
    """
    Assess classification model quality based on AUC with sample size adjustment.
    
    Args:
        auc: Area Under ROC Curve (0 to 1)
        n_samples: Number of training samples (optional, for overfit detection)
        n_features: Number of features used (optional, for overfit detection)
        
    Returns:
        Quality level: 'poor', 'moderate', 'good', or 'excellent'
        
    Example:
        >>> quality = assess_classification_quality(0.82)
        >>> print(quality)  # 'excellent'
        
        >>> quality = assess_classification_quality(0.88, n_samples=25, n_features=12)
        >>> print(quality)  # 'good' (penalized for overfit risk)
    """
    # Apply sample size penalty if both n_samples and n_features provided
    auc_adjusted = auc
    if n_samples is not None and n_features is not None and n_features > 0:
        samples_per_feature = n_samples / n_features
        if samples_per_feature < MIN_SAMPLES_PER_FEATURE_SEVERE:
            # Severe overfit risk - heavy penalty
            auc_adjusted = auc - SAMPLE_SIZE_PENALTY_SEVERE
        elif samples_per_feature < MIN_SAMPLES_PER_FEATURE_MODERATE:
            # Moderate overfit risk - light penalty
            auc_adjusted = auc - SAMPLE_SIZE_PENALTY_MODERATE
    
    # Apply quality thresholds to adjusted AUC
    if auc_adjusted < AUC_POOR_THRESHOLD:
        return QUALITY_POOR
    elif auc_adjusted < AUC_MODERATE_THRESHOLD:
        return QUALITY_MODERATE
    elif auc_adjusted < AUC_GOOD_THRESHOLD:
        return QUALITY_GOOD
    else:
        return QUALITY_EXCELLENT


# ============================================
# Helper Functions for Suggestions
# ============================================


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
            feature_columns: List of feature columns (None = auto-select clean numeric features)
            allow_non_numeric_target: Whether to allow non-numeric targets (for classification)
            
        Returns:
            Tuple of (X, y, feature_columns, error_message, n_missing, auto_exclusion_warning)
            - X: Feature matrix (None if error)
            - y: Target vector (None if error)
            - feature_columns: List of selected features (None if error)
            - error_message: Error string if validation failed (None if success)
            - n_missing: Number of rows removed due to missing data
            - auto_exclusion_warning: Warning about auto-excluded features (None if none excluded)
        """
        auto_exclusion_warning = None  # Track if we auto-excluded features
        
        # Validate target column exists
        if target_column not in df.columns:
            return None, None, None, f"❌ Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}", 0, None
        
        # Validate target is numeric (for regression only)
        if not allow_non_numeric_target and not pd.api.types.is_numeric_dtype(df[target_column]):
            return None, None, None, f"❌ Target column '{target_column}' must be numeric for regression", 0, None
        
        # Auto-select features if not provided
        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != target_column]
            
            # Only include features with <20% missing data for reliable modeling
            MISSING_THRESHOLD = 0.20
            clean_features = []
            excluded_features = []
            
            for col in numeric_cols:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct < MISSING_THRESHOLD:
                    clean_features.append(col)
                else:
                    excluded_features.append((col, missing_pct))
            
            feature_columns = clean_features
            
            # Provide helpful feedback if we excluded features
            if excluded_features and not feature_columns:
                excluded_info = ", ".join([f"{col} ({pct:.1%} missing)" for col, pct in excluded_features])
                return None, None, None, f"❌ All numeric features have >{MISSING_THRESHOLD:.0%} missing data: {excluded_info}\n\n💡 Run apply_imputation first or specify features explicitly with feature_columns=['col1', 'col2']", 0, None
            elif excluded_features:
                # Store excluded info for informational message (will be shown in results)
                excluded_list = [f"{col} ({pct:.1%} missing)" for col, pct in excluded_features]
                auto_exclusion_warning = f"ℹ️ Auto-excluded {len(excluded_features)} high-missing feature(s): {', '.join(excluded_list[:3])}" + (f" and {len(excluded_list)-3} more" if len(excluded_list) > 3 else "")
        
        if not feature_columns:
            return None, None, None, "❌ No numeric feature columns available for regression", 0, None
        
        # Check for missing feature columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return None, None, None, f"❌ Feature columns not found: {missing_cols}", 0, None
        
        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing values and track count
        # Remove rows with missing values and track count
        missing_mask = X.isnull().any(axis=1) | y.isnull()
        n_missing = int(missing_mask.sum())
        if n_missing > 0:
            X = X[~missing_mask]
            y = y[~missing_mask]
        
        return X, y, feature_columns, None, n_missing, auto_exclusion_warning
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
        # Respect explicit feature lists from users - only auto-include when None
        if requested_features is not None:
            # User explicitly specified features - use ONLY what they asked for
            used_features = set(requested_features)
            context['auto_included_encoded'] = []
        else:
            # User wants auto-selection - now we can consider encoded columns
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
                context['auto_included_encoded'] = []
        
        unused_columns = all_columns - used_features
        
        # Get encoded column mapping from workflow state
        workflow_state = get_workflow_state()
        encoded_mapping = workflow_state.get('encoded_column_mapping', {})
        
        # Categorize unused columns by type
        # Note: Skip categorical columns if their encoded version was used
        for col in unused_columns:
            dtype = df[col].dtype
            if dtype == 'object' or dtype.name == 'category':
                # Check if this column has an encoded version being used
                if col in encoded_mapping:
                    encoded_col_name = encoded_mapping[col]
                    if encoded_col_name in used_features:
                        # Encoded version is being used - don't suggest encoding again
                        continue
                
                # Column not encoded OR encoded version not used - suggest encoding
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
        
        # Extract sample size and feature count for overfit detection
        n_samples = metrics.get('n_samples', None)
        n_features = dataset_context.get('feature_count', None)
        
        # Determine quality level based on model type
        if model_type == 'regression':
            test_r2 = metrics.get('test_r2', 0)
            train_r2 = metrics.get('train_r2', 0)
            assessment['quality_level'] = assess_regression_quality(test_r2, n_samples, n_features)
            primary_metric = test_r2
            metric_name = 'R²'
            
            # Add sample size penalty transparency warning
            if n_samples is not None and n_features is not None and n_features > 0:
                samples_per_feature = n_samples / n_features
                if samples_per_feature < MIN_SAMPLES_PER_FEATURE_SEVERE:
                    assessment['warnings'].append(
                        f"⚠️ Small sample size: {n_samples} samples / {n_features} features = {samples_per_feature:.1f} samples/feature. "
                        f"Quality rating adjusted down by {SAMPLE_SIZE_PENALTY_SEVERE} to account for high overfitting risk."
                    )
                elif samples_per_feature < MIN_SAMPLES_PER_FEATURE_MODERATE:
                    assessment['warnings'].append(
                        f"⚠️ Limited sample size: {n_samples} samples / {n_features} features = {samples_per_feature:.1f} samples/feature. "
                        f"Quality rating adjusted down by {SAMPLE_SIZE_PENALTY_MODERATE} to account for potential overfitting."
                    )
            
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
            assessment['quality_level'] = assess_classification_quality(test_auc, n_samples, n_features)
            primary_metric = test_auc
            metric_name = 'AUC'
            
            # Add sample size penalty transparency warning
            if n_samples is not None and n_features is not None and n_features > 0:
                samples_per_feature = n_samples / n_features
                if samples_per_feature < MIN_SAMPLES_PER_FEATURE_SEVERE:
                    assessment['warnings'].append(
                        f"⚠️ Small sample size: {n_samples} samples / {n_features} features = {samples_per_feature:.1f} samples/feature. "
                        f"Quality rating adjusted down by {SAMPLE_SIZE_PENALTY_SEVERE} to account for high overfitting risk."
                    )
                elif samples_per_feature < MIN_SAMPLES_PER_FEATURE_MODERATE:
                    assessment['warnings'].append(
                        f"⚠️ Limited sample size: {n_samples} samples / {n_features} features = {samples_per_feature:.1f} samples/feature. "
                        f"Quality rating adjusted down by {SAMPLE_SIZE_PENALTY_MODERATE} to account for potential overfitting."
                    )
            
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
        
        # Add quality label using QUALITY_LEVELS from constants
        assessment['quality_label'] = QUALITY_LEVELS[assessment['quality_level']]['label']
        
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
        Generate prioritized workflow suggestions using priority queue.
        
        Suggestions are automatically sorted by priority (CRITICAL → RECOMMENDED → OPTIONAL)
        before returning top 5. This ensures most important suggestions always appear first
        regardless of code order.
        
        Args:
            quality_assessment: Output from _assess_model_quality()
            dataset_context: Output from _analyze_ml_features()
            target_column: Target variable name
            current_features: List of features currently used
            model_type: 'regression' or 'classification'
            
        Returns:
            List of formatted suggestions with priority indicators (max 5)
        """
        # Priority queue: list of (priority_int, suggestion_str)
        # Lower priority_int = higher priority (0=CRITICAL, 1=RECOMMENDED, 2=OPTIONAL)
        queue = []
        
        quality = quality_assessment['quality_level']
        
        # Delegate to focused suggestion functions
        self._suggest_categorical_encoding(queue, quality, dataset_context, target_column, model_type)
        self._suggest_class_balance(queue, quality, model_type)
        self._suggest_missing_data_recovery(queue, quality, dataset_context)
        self._suggest_feature_scaling(queue, quality, current_features)
        self._suggest_outlier_handling(queue, quality)
        self._suggest_model_evaluation(queue, quality, model_type)
        self._suggest_visualization(queue, model_type)
        
        # Sort by priority (0=CRITICAL first, then 1=RECOMMENDED, then 2=OPTIONAL)
        queue.sort(key=lambda x: x[0])
        
        # Return top 5 formatted suggestions
        return [suggestion for _, suggestion in queue[:5]]
    
    def _suggest_categorical_encoding(
        self,
        queue: List[tuple],
        quality: str,
        dataset_context: Dict[str, Any],
        target_column: str,
        model_type: str
    ):
        """Add categorical encoding suggestion if applicable."""
        if quality not in [QUALITY_POOR, QUALITY_MODERATE]:
            return
        
        unused_cats = dataset_context.get('unused_categorical', [])
        if not unused_cats:
            return
        
        priority = PRIORITY_CRITICAL_INT
        suggestion = format_suggestion(
            PRIORITY_CRITICAL,
            f"Apply mean_target_encoding to {unused_cats} with target_column='{target_column}', then rerun {model_type}",
            "categorical features often contain crucial predictive information and will significantly improve model performance"
        )
        queue.append((priority, suggestion))
    
    def _suggest_class_balance(
        self,
        queue: List[tuple],
        quality: str,
        model_type: str
    ):
        """Add class balance suggestion if applicable."""
        if model_type != 'classification' or quality not in [QUALITY_POOR, QUALITY_MODERATE]:
            return
        
        workflow_state = get_workflow_state()
        if workflow_state.get('class_balance_addressed'):
            return  # Already addressed
        
        priority = PRIORITY_CRITICAL_INT
        suggestion = format_suggestion(
            PRIORITY_CRITICAL,
            "Check class balance with df[target].value_counts(), then rerun with class_weight='balanced' if imbalanced",
            "class imbalance can severely hurt model performance on minority class"
        )
        queue.append((priority, suggestion))
    
    def _suggest_missing_data_recovery(
        self,
        queue: List[tuple],
        quality: str,
        dataset_context: Dict[str, Any]
    ):
        """Add missing data recovery suggestion if applicable."""
        if quality not in [QUALITY_POOR, QUALITY_MODERATE]:
            return
        
        n_missing = dataset_context.get('missing_removed', 0)
        if n_missing <= MISSING_DATA_RECOVERY_THRESHOLD:
            return  # Not enough missing data to worry about
        
        priority = PRIORITY_RECOMMENDED_INT
        suggestion = format_suggestion(
            PRIORITY_RECOMMENDED,
            f"Run analyze_missing_data to check if imputation could recover {n_missing} removed rows",
            "more data may improve model performance"
        )
        queue.append((priority, suggestion))
    
    def _suggest_feature_scaling(
        self,
        queue: List[tuple],
        quality: str,
        current_features: List[str]
    ):
        """Add feature scaling suggestion if applicable."""
        if len(current_features) <= 1 or quality not in [QUALITY_POOR, QUALITY_MODERATE]:
            return
        
        workflow_state = get_workflow_state()
        if workflow_state.get('features_standardized'):
            return  # Already standardized
        
        priority = PRIORITY_RECOMMENDED_INT
        suggestion = format_suggestion(
            PRIORITY_RECOMMENDED,
            "Rerun with standardize_features=True to normalize feature scales",
            "standardization improves coefficient interpretation and model stability"
        )
        queue.append((priority, suggestion))
    
    def _suggest_outlier_handling(
        self,
        queue: List[tuple],
        quality: str
    ):
        """Add outlier handling suggestion if applicable."""
        if quality not in [QUALITY_POOR, QUALITY_MODERATE]:
            return
        
        workflow_state = get_workflow_state()
        if not workflow_state.get('outliers_detected'):
            return  # No outliers detected
        
        outlier_cols = workflow_state.get('outlier_columns', [])
        if not outlier_cols:
            return
        
        priority = PRIORITY_RECOMMENDED_INT
        suggestion = format_suggestion(
            PRIORITY_RECOMMENDED,
            f"Consider handling outliers in {outlier_cols[:3]} which may be affecting model fit",
            "outliers can disproportionately influence model coefficients"
        )
        queue.append((priority, suggestion))
    
    def _suggest_model_evaluation(
        self,
        queue: List[tuple],
        quality: str,
        model_type: str
    ):
        """Add model evaluation suggestion if applicable."""
        if quality not in [QUALITY_MODERATE, QUALITY_GOOD, QUALITY_EXCELLENT]:
            return  # Only suggest for decent models
        
        eval_tool = 'evaluate_regression_model' if model_type == 'regression' else 'evaluate_classification_model'
        
        priority = PRIORITY_RECOMMENDED_INT
        suggestion = format_suggestion(
            PRIORITY_RECOMMENDED,
            f"Use {eval_tool} for comprehensive diagnostics and assumption validation",
            "validates model assumptions before making decisions"
        )
        queue.append((priority, suggestion))
    
    def _suggest_visualization(
        self,
        queue: List[tuple],
        model_type: str
    ):
        """Add visualization suggestion (always applicable)."""
        if model_type == 'regression':
            viz_tools = "create_regression_plot, create_residual_plot, and create_coefficient_chart"
        else:
            viz_tools = "create_roc_curve, create_precision_recall_curve, and create_feature_importance_chart"
        
        priority = PRIORITY_OPTIONAL_INT
        suggestion = format_suggestion(
            PRIORITY_OPTIONAL,
            f"Create visualizations with {viz_tools}",
            "visual inspection reveals patterns metrics may miss"
        )
        queue.append((priority, suggestion))
    
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
