"""
Unit tests for smart ML tool outputs (Phase 1 implementation).

Tests verify that tools:
1. Detect unused categorical features
2. Generate quality-based suggestions
3. Coordinate via workflow metadata
4. Format suggestions with priority indicators
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import streamlit as st

from stats_compass.tools.ml_regression_tools import RunLinearRegressionTool
from stats_compass.utils.workflow_state import (
    get_workflow_state, update_workflow_state, reset_workflow_state
)
from stats_compass.tools.ml_guidance import (
    PRIORITY_CRITICAL, PRIORITY_RECOMMENDED, PRIORITY_OPTIONAL
)


class TestDatasetFeatureAnalysis:
    """Test dataset inspection and feature opportunity detection."""
    
    def setup_method(self):
        """Reset workflow state before each test."""
        reset_workflow_state()
    
    def test_detect_unused_categorical_features(self):
        """Test that tool identifies unused categorical columns."""
        df = pd.DataFrame({
            'Runtime': [90, 120, 95, 110, 105],
            'IMDB Score': [7.5, 8.0, 6.8, 7.2, 7.8],
            'Genre': ['Action', 'Drama', 'Comedy', 'Action', 'Drama'],
            'Language': ['English', 'English', 'Spanish', 'English', 'French'],
            'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
        })
        
        tool = RunLinearRegressionTool(df)
        context = tool._analyze_dataset_features(df, 'IMDB Score', ['Runtime'])
        
        # Should detect 3 unused categorical columns
        assert 'Genre' in context['unused_categorical']
        assert 'Language' in context['unused_categorical']
        assert 'Title' in context['unused_categorical']
        assert context['feature_count'] == 1  # Only Runtime
        assert context['has_unused_features'] is True
    
    def test_detect_unused_numeric_features(self):
        """Test that tool identifies unused numeric columns."""
        df = pd.DataFrame({
            'Runtime': [90, 120, 95, 110, 105],
            'Budget': [1000000, 2000000, 1500000, 1800000, 1200000],
            'IMDB Score': [7.5, 8.0, 6.8, 7.2, 7.8],
            'Revenue': [5000000, 8000000, 4000000, 6000000, 5500000]
        })
        
        tool = RunLinearRegressionTool(df)
        context = tool._analyze_dataset_features(df, 'IMDB Score', ['Runtime'])
        
        # Should detect 2 unused numeric columns
        assert 'Budget' in context['unused_numeric']
        assert 'Revenue' in context['unused_numeric']
        assert context['feature_count'] == 1
    
    def test_auto_include_encoded_features(self):
        """Test that encoded features are auto-included when available."""
        df = pd.DataFrame({
            'Runtime': [90, 120, 95, 110, 105],
            'IMDB Score': [7.5, 8.0, 6.8, 7.2, 7.8],
            'Genre_encoded': [0.5, 0.8, 0.6, 0.5, 0.8],
            'Language_encoded': [0.7, 0.7, 0.6, 0.7, 0.65]
        })
        
        # Simulate previous encoding step
        update_workflow_state({
            'categorical_encoded': True,
            'available_encoded_columns': ['Genre_encoded', 'Language_encoded']
        })
        
        tool = RunLinearRegressionTool(df)
        context = tool._analyze_dataset_features(df, 'IMDB Score', None)
        
        # Should auto-include encoded columns
        assert context['auto_included_encoded'] == ['Genre_encoded', 'Language_encoded']
        assert context['feature_count'] == 3  # Runtime + 2 encoded


class TestModelQualityAssessment:
    """Test model quality assessment and warning generation."""
    
    def test_poor_quality_generates_warnings(self):
        """Test that poor R² triggers appropriate warnings."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        dataset_context = {
            'unused_categorical': ['Genre', 'Language'],
            'unused_numeric': [],
            'total_columns': 5,
            'feature_count': 1
        }
        
        assessment = tool._assess_model_quality(
            train_r2=0.05,
            test_r2=0.02,
            n_features=1,
            dataset_context=dataset_context
        )
        
        assert assessment['quality_level'] == 'poor'
        assert assessment['quality_label'] == '❌ Poor'
        assert len(assessment['warnings']) >= 3
        
        # Check specific warnings
        warnings_text = ' '.join(assessment['warnings'])
        assert 'low R²' in warnings_text.lower() or 'very low r²' in warnings_text.lower()
        assert '1 feature' in warnings_text.lower() or 'only 1 feature' in warnings_text.lower()
        assert 'unused categorical' in warnings_text.lower()
    
    def test_overfitting_warning(self):
        """Test that overfitting is detected."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        assessment = tool._assess_model_quality(
            train_r2=0.85,
            test_r2=0.65,
            n_features=5,
            dataset_context={'unused_categorical': [], 'total_columns': 7, 'feature_count': 5}
        )
        
        warnings_text = ' '.join(assessment['warnings'])
        assert 'overfitting' in warnings_text.lower()
    
    def test_quality_levels(self):
        """Test quality level classification."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        context = {'unused_categorical': [], 'total_columns': 5, 'feature_count': 3}
        
        # Test each quality level
        poor = tool._assess_model_quality(0.2, 0.15, 3, context)
        assert poor['quality_level'] == 'poor'
        
        moderate = tool._assess_model_quality(0.45, 0.42, 3, context)
        assert moderate['quality_level'] == 'moderate'
        
        good = tool._assess_model_quality(0.65, 0.62, 3, context)
        assert good['quality_level'] == 'good'
        
        excellent = tool._assess_model_quality(0.85, 0.82, 3, context)
        assert excellent['quality_level'] == 'excellent'


class TestWorkflowSuggestions:
    """Test smart workflow suggestion generation."""
    
    def test_critical_encoding_suggestion_for_poor_quality(self):
        """Test that poor quality with unused categoricals triggers CRITICAL encoding suggestion."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        quality_assessment = {
            'quality_level': 'poor',
            'warnings': ['Low R²']
        }
        
        dataset_context = {
            'unused_categorical': ['Genre', 'Language', 'Title'],
            'unused_numeric': [],
            'missing_removed': 5
        }
        
        suggestions = tool._generate_workflow_suggestions(
            quality_assessment,
            dataset_context,
            target_column='IMDB Score',
            current_features=['Runtime']
        )
        
        # First suggestion should be CRITICAL encoding
        assert len(suggestions) > 0
        assert PRIORITY_CRITICAL in suggestions[0]
        assert 'mean_target_encoding' in suggestions[0]
        assert 'Genre' in suggestions[0] or "['Genre', 'Language', 'Title']" in suggestions[0]
    
    def test_evaluation_suggestion_for_good_quality(self):
        """Test that good quality triggers evaluation suggestion."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        quality_assessment = {
            'quality_level': 'good',
            'warnings': []
        }
        
        dataset_context = {
            'unused_categorical': [],
            'unused_numeric': [],
            'missing_removed': 0
        }
        
        suggestions = tool._generate_workflow_suggestions(
            quality_assessment,
            dataset_context,
            target_column='IMDB Score',
            current_features=['Runtime', 'Budget', 'Genre_encoded']
        )
        
        # Should suggest evaluation
        suggestions_text = ' '.join(suggestions)
        assert 'evaluate_regression_model' in suggestions_text
        assert PRIORITY_RECOMMENDED in suggestions_text
    
    def test_suggestion_count_limit(self):
        """Test that suggestions are capped at 5."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        quality_assessment = {'quality_level': 'poor', 'warnings': []}
        dataset_context = {
            'unused_categorical': ['Genre', 'Language'],
            'unused_numeric': ['Budget'],
            'missing_removed': 100
        }
        
        suggestions = tool._generate_workflow_suggestions(
            quality_assessment,
            dataset_context,
            'IMDB Score',
            ['Runtime']
        )
        
        # Should be capped at 5
        assert len(suggestions) <= 5
    
    def test_priority_indicators_present(self):
        """Test that suggestions include priority indicators."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        quality_assessment = {'quality_level': 'moderate', 'warnings': []}
        dataset_context = {
            'unused_categorical': ['Genre'],
            'unused_numeric': [],
            'missing_removed': 10
        }
        
        suggestions = tool._generate_workflow_suggestions(
            quality_assessment,
            dataset_context,
            'IMDB Score',
            ['Runtime']
        )
        
        # Should have priority indicators
        all_suggestions = ' '.join(suggestions)
        assert '🔴' in all_suggestions or '🟡' in all_suggestions or '🟢' in all_suggestions


class TestWorkflowMetadataCoordination:
    """Test workflow state coordination between tools."""
    
    def setup_method(self):
        """Reset workflow state before each test."""
        reset_workflow_state()
    
    @patch('streamlit.session_state', {})
    def test_metadata_updated_after_regression(self):
        """Test that regression updates workflow metadata."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        tool._update_workflow_metadata(
            model_quality='good',
            features_used=['Runtime', 'Budget'],
            model_key='linear_regression_IMDB Score_123',
            target_column='IMDB Score',
            test_r2=0.65
        )
        
        state = get_workflow_state()
        assert state['last_model_type'] == 'regression'
        assert state['last_model_quality'] == 'good'
        assert state['last_model_features'] == ['Runtime', 'Budget']
        assert state['last_model_key'] == 'linear_regression_IMDB Score_123'
        assert state['last_model_target'] == 'IMDB Score'
    
    @patch('streamlit.session_state', {})
    def test_models_trained_list_updated(self):
        """Test that models_trained list is maintained."""
        tool = RunLinearRegressionTool(pd.DataFrame())
        
        # Train first model
        tool._update_workflow_metadata('poor', ['Runtime'], 'model_1', 'IMDB Score', 0.1)
        
        # Train second model
        tool._update_workflow_metadata('good', ['Runtime', 'Genre_encoded'], 'model_2', 'IMDB Score', 0.65)
        
        state = get_workflow_state()
        assert len(state['models_trained']) == 2
        assert 'model_1' in state['models_trained']
        assert 'model_2' in state['models_trained']


class TestEndToEndIntegration:
    """Integration tests for complete workflow."""
    
    @patch('streamlit.session_state', {})
    def test_poor_model_triggers_encoding_workflow(self):
        """Test that poor model with categoricals suggests encoding."""
        df = pd.DataFrame({
            'Runtime': [90, 120, 95, 110, 105, 100, 115, 92, 108, 97],
            'IMDB Score': [7.5, 8.0, 6.8, 7.2, 7.8, 7.1, 7.9, 6.9, 7.4, 7.0],
            'Genre': ['Action', 'Drama', 'Comedy', 'Action', 'Drama', 'Comedy', 'Action', 'Drama', 'Comedy', 'Action'],
            'Language': ['English', 'English', 'Spanish', 'English', 'French', 'English', 'Spanish', 'French', 'English', 'Spanish']
        })
        
        tool = RunLinearRegressionTool(df)
        output = tool._run(target_column='IMDB Score', feature_columns=['Runtime'])
        
        # Output should contain:
        # 1. Warning about low R²
        # 2. Warning about unused categorical columns
        # 3. CRITICAL suggestion to encode
        assert '⚠️' in output or '❌' in output  # Quality warning
        assert 'unused categorical' in output.lower() or 'genre' in output.lower()
        assert 'mean_target_encoding' in output.lower()
        assert PRIORITY_CRITICAL in output or '🔴' in output
    
    @patch('streamlit.session_state', {'workflow_metadata': {}})
    def test_auto_include_encoded_features_workflow(self):
        """Test that regression auto-includes encoded features from previous step."""
        df = pd.DataFrame({
            'Runtime': [90, 120, 95, 110, 105, 100, 115, 92, 108, 97],
            'IMDB Score': [7.5, 8.0, 6.8, 7.2, 7.8, 7.1, 7.9, 6.9, 7.4, 7.0],
            'Genre_encoded': [0.5, 0.8, 0.6, 0.5, 0.8, 0.6, 0.5, 0.8, 0.6, 0.5],
            'Language_encoded': [0.7, 0.7, 0.6, 0.7, 0.65, 0.7, 0.6, 0.65, 0.7, 0.6]
        })
        
        # Simulate previous encoding
        update_workflow_state({
            'categorical_encoded': True,
            'available_encoded_columns': ['Genre_encoded', 'Language_encoded']
        })
        
        tool = RunLinearRegressionTool(df)
        output = tool._run(target_column='IMDB Score')  # No features specified
        
        # Should auto-include encoded features
        assert 'Genre_encoded' in output
        assert 'Language_encoded' in output
        assert 'auto-included' in output.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
