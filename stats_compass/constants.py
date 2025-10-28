"""
Configuration constants for Stats Compass tools.

Centralized constants for:
- Query execution limits
- Display settings
- Correlation thresholds
- ML quality assessment thresholds
- Sample size requirements

This module provides a single source of truth for all configuration values
used across the Stats Compass application.
"""

# ============================================
# Query Execution Limits
# ============================================

# Maximum number of lines in a single query
MAX_QUERY_LINES = 3

# Maximum size of stored variables in MB
MAX_VARIABLE_SIZE_MB = 50

# Memory threshold for clearing user variables (MB)
MAX_MEMORY_USAGE_MB = 100

# Threshold for memory usage warnings (MB)
MEMORY_WARNING_THRESHOLD_MB = 100


# ============================================
# Display Settings
# ============================================

# Maximum columns to display in output
MAX_DISPLAY_COLUMNS = 50

# Maximum width for column content
MAX_DISPLAY_COLWIDTH = 50

# Maximum rows for dataset preview
MAX_PREVIEW_ROWS = 20

# Maximum columns for dataset preview
MAX_PREVIEW_COLUMNS = 100


# ============================================
# Analysis Thresholds
# ============================================

# Threshold for flagging strong correlations
STRONG_CORRELATION_THRESHOLD = 0.7


# ============================================
# ML Quality Assessment Thresholds
# ============================================

# Regression R² thresholds
R2_POOR_THRESHOLD = 0.3
R2_MODERATE_THRESHOLD = 0.5
R2_GOOD_THRESHOLD = 0.7

# Classification AUC thresholds
AUC_POOR_THRESHOLD = 0.6
AUC_MODERATE_THRESHOLD = 0.7
AUC_GOOD_THRESHOLD = 0.8

# Overfitting detection thresholds (train-test gap)
OVERFITTING_GAP_THRESHOLD_REGRESSION = 0.15
OVERFITTING_GAP_THRESHOLD_CLASSIFICATION = 0.10

# Overfitting risk thresholds for warnings
OVERFITTING_WARNING_THRESHOLD_REGRESSION = 0.1
OVERFITTING_WARNING_THRESHOLD_CLASSIFICATION = 0.05


# ============================================
# ML Sample Size Requirements
# ============================================

# Sample size thresholds for overfitting detection
MIN_SAMPLES_PER_FEATURE_SEVERE = 10  # Heavy penalty threshold
MIN_SAMPLES_PER_FEATURE_MODERATE = 30  # Light penalty threshold

# Quality penalties for small sample sizes
SAMPLE_SIZE_PENALTY_SEVERE = 0.15  # Applied when samples/feature < 10
SAMPLE_SIZE_PENALTY_MODERATE = 0.05  # Applied when samples/feature < 30

# Minimum samples per feature for ML models (general guidance)
MIN_SAMPLES_PER_FEATURE = 10

# Recommended samples per feature for robust models
RECOMMENDED_SAMPLES_PER_FEATURE = 30

# Minimum features threshold for warnings
MIN_FEATURES_WARNING_THRESHOLD = 3

# Minimum columns needed before feature count warnings
MIN_COLUMNS_FOR_FEATURE_WARNING = 5

# Missing data recovery threshold (rows)
MISSING_DATA_RECOVERY_THRESHOLD = 50


# ============================================
# Suggestion Priorities
# ============================================

# Priority labels (for display)
PRIORITY_CRITICAL = '🔴 CRITICAL'
PRIORITY_RECOMMENDED = '🟡 RECOMMENDED'
PRIORITY_OPTIONAL = '🟢 OPTIONAL'

# Priority integers (for sorting in priority queue)
PRIORITY_CRITICAL_INT = 0
PRIORITY_RECOMMENDED_INT = 1
PRIORITY_OPTIONAL_INT = 2


# ============================================
# Quality Level Names
# ============================================

QUALITY_POOR = 'poor'
QUALITY_MODERATE = 'moderate'
QUALITY_GOOD = 'good'
QUALITY_EXCELLENT = 'excellent'


# ============================================
# Quality Levels (for ML assessment)
# ============================================

QUALITY_LEVELS = {
    QUALITY_POOR: {
        'label': '❌ Poor',
        'description': 'Not suitable for decision-making',
        'regression_r2_threshold': R2_POOR_THRESHOLD,
        'classification_auc_threshold': AUC_POOR_THRESHOLD,
    },
    QUALITY_MODERATE: {
        'label': '⚠️ Moderate',
        'description': 'Suitable for exploratory analysis, use caution for decisions',
        'regression_r2_threshold': R2_MODERATE_THRESHOLD,
        'classification_auc_threshold': AUC_MODERATE_THRESHOLD,
    },
    QUALITY_GOOD: {
        'label': '✅ Good',
        'description': 'Suitable for decision-making with monitoring',
        'regression_r2_threshold': R2_GOOD_THRESHOLD,
        'classification_auc_threshold': AUC_GOOD_THRESHOLD,
    },
    QUALITY_EXCELLENT: {
        'label': '🎯 Excellent',
        'description': 'Ready for production use',
        'regression_r2_threshold': float('inf'),
        'classification_auc_threshold': float('inf'),
    },
}
