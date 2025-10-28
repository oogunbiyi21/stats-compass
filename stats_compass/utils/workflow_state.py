"""
Workflow State Schema for Tool Coordination

This module defines the metadata schema that tools use to coordinate multi-step workflows.
Tools read/write to st.session_state.workflow_metadata using these standardized fields.

Usage:
    from utils.workflow_state import get_workflow_state, update_workflow_state, WORKFLOW_STATE_SCHEMA
    
    # Reading state
    state = get_workflow_state()
    if state.get('categorical_encoded'):
        encoded_cols = state.get('available_encoded_columns', [])
    
    # Writing state
    update_workflow_state({
        'categorical_encoded': True,
        'available_encoded_columns': ['Genre_encoded', 'Language_encoded']
    })
"""

from typing import Dict, Any, List, Optional
import streamlit as st


# ============================================
# Workflow State Schema Definition
# ============================================

WORKFLOW_STATE_SCHEMA = {
    # ============================================
    # Feature Engineering
    # ============================================
    'categorical_encoded': {
        'type': bool,
        'description': 'Whether categorical columns have been encoded',
        'default': False,
        'set_by': ['MeanTargetEncodingTool', 'BinRareCategoriesTool'],
        'read_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool']
    },
    'available_encoded_columns': {
        'type': List[str],
        'description': 'List of encoded column names available for ML',
        'default': [],
        'set_by': ['MeanTargetEncodingTool'],
        'read_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool']
    },
    'encoded_column_mapping': {
        'type': Dict[str, str],
        'description': 'Maps original categorical column names to their encoded versions. Example: {"city": "city_encoded", "genre": "genre_mean_enc"}',
        'default': {},
        'set_by': ['MeanTargetEncodingTool'],
        'read_by': ['SmartMLToolMixin._analyze_ml_features']
    },
    'encoding_target': {
        'type': str,
        'description': 'Target column used for encoding (for consistency checks)',
        'default': None,
        'set_by': ['MeanTargetEncodingTool'],
        'read_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool']
    },
    'rare_categories_binned': {
        'type': bool,
        'description': 'Whether rare categories have been grouped into "Other"',
        'default': False,
        'set_by': ['BinRareCategoriesTool'],
        'read_by': ['MeanTargetEncodingTool']
    },
    
    # ============================================
    # Data Quality
    # ============================================
    'missing_data_analyzed': {
        'type': bool,
        'description': 'Whether missing data patterns have been analyzed',
        'default': False,
        'set_by': ['AnalyzeMissingDataTool'],
        'read_by': ['ApplyImputationTool', 'RunLinearRegressionTool']
    },
    'missing_data_handled': {
        'type': bool,
        'description': 'Whether missing values have been imputed or removed',
        'default': False,
        'set_by': ['ApplyImputationTool', 'ApplyBasicCleaningTool'],
        'read_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool']
    },
    'outliers_detected': {
        'type': bool,
        'description': 'Whether outlier detection has been performed',
        'default': False,
        'set_by': ['DetectOutliersTool'],
        'read_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool']
    },
    'outlier_columns': {
        'type': List[str],
        'description': 'Columns with detected outliers',
        'default': [],
        'set_by': ['DetectOutliersTool'],
        'read_by': ['RunLinearRegressionTool']
    },
    
    # ============================================
    # Modeling
    # ============================================
    'last_model_key': {
        'type': str,
        'description': 'Key of the most recently trained model (for evaluation/visualization)',
        'default': None,
        'set_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool', 'RunTimeSeriesAnalysisTool'],
        'read_by': ['EvaluateRegressionTool', 'CreateRegressionPlotTool']
    },
    'last_model_type': {
        'type': str,
        'description': 'Type of last model: "regression", "classification", "timeseries"',
        'default': None,
        'set_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool', 'RunTimeSeriesAnalysisTool'],
        'read_by': ['EvaluateRegressionTool', 'CreateRegressionPlotTool']
    },
    'last_model_quality': {
        'type': str,
        'description': 'Quality assessment: "poor", "moderate", "good", "excellent"',
        'default': None,
        'set_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool'],
        'read_by': ['EvaluateRegressionTool']
    },
    'last_model_target': {
        'type': str,
        'description': 'Target column of the last model',
        'default': None,
        'set_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool'],
        'read_by': ['EvaluateRegressionTool', 'CreateRegressionPlotTool']
    },
    'last_model_features': {
        'type': List[str],
        'description': 'Features used in the last model',
        'default': [],
        'set_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool'],
        'read_by': ['EvaluateRegressionTool']
    },
    'models_trained': {
        'type': List[str],
        'description': 'List of all model keys trained in this session',
        'default': [],
        'set_by': ['RunLinearRegressionTool', 'RunLogisticRegressionTool', 'RunTimeSeriesAnalysisTool'],
        'read_by': []  # For tracking/debugging
    },
}


# ============================================
# Helper Functions
# ============================================

def get_workflow_state() -> Dict[str, Any]:
    """
    Get the current workflow state with defaults.
    
    Returns:
        Dictionary with current workflow metadata. Missing fields use schema defaults.
        
    Example:
        >>> state = get_workflow_state()
        >>> if state.get('categorical_encoded'):
        >>>     cols = state.get('available_encoded_columns', [])
    """
    if not hasattr(st, 'session_state'):
        return {}
    
    if 'workflow_metadata' not in st.session_state:
        st.session_state.workflow_metadata = {}
    
    # Apply defaults for missing fields
    state = st.session_state.workflow_metadata.copy()
    for field, spec in WORKFLOW_STATE_SCHEMA.items():
        if field not in state:
            state[field] = spec['default']
    
    return state


def update_workflow_state(updates: Dict[str, Any]) -> None:
    """
    Update workflow state with new values.
    
    Args:
        updates: Dictionary of field_name: value pairs to update
        
    Example:
        >>> update_workflow_state({
        >>>     'categorical_encoded': True,
        >>>     'available_encoded_columns': ['Genre_encoded', 'Language_encoded']
        >>> })
    """
    if not hasattr(st, 'session_state'):
        return
    
    if 'workflow_metadata' not in st.session_state:
        st.session_state.workflow_metadata = {}
    
    st.session_state.workflow_metadata.update(updates)


def reset_workflow_state() -> None:
    """
    Reset workflow state to defaults (useful when loading new dataset).
    
    Example:
        >>> # When user uploads new file
        >>> reset_workflow_state()
    """
    if hasattr(st, 'session_state'):
        st.session_state.workflow_metadata = {}


def get_state_field(field: str, default: Any = None) -> Any:
    """
    Get a single field from workflow state.
    
    Args:
        field: Field name from WORKFLOW_STATE_SCHEMA
        default: Default value if field not found (overrides schema default)
        
    Returns:
        Field value or default
        
    Example:
        >>> encoded_cols = get_state_field('available_encoded_columns', [])
    """
    state = get_workflow_state()
    
    if default is not None:
        return state.get(field, default)
    else:
        # Use schema default if available
        schema_default = WORKFLOW_STATE_SCHEMA.get(field, {}).get('default')
        return state.get(field, schema_default)


def set_state_field(field: str, value: Any) -> None:
    """
    Set a single field in workflow state.
    
    Args:
        field: Field name from WORKFLOW_STATE_SCHEMA
        value: Value to set
        
    Example:
        >>> set_state_field('categorical_encoded', True)
    """
    update_workflow_state({field: value})
