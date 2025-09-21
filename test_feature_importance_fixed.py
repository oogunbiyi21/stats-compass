#!/usr/bin/env python3
"""
Test script for the CreateFeatureImportanceChartTool - Fixed version that works without Streamlit context
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock Streamlit for testing
class MockSessionState:
    def __init__(self):
        self.data = {}
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __contains__(self, key):
        return key in self.data
    
    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        raise AttributeError(f"'MockSessionState' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        if key == 'data':
            super().__setattr__(key, value)
        else:
            if not hasattr(self, 'data'):
                super().__setattr__('data', {})
            self.data[key] = value
    
    def get(self, key, default=None):
        return self.data.get(key, default)

class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()
        
    def hasattr(self, obj, name):
        return hasattr(obj, name)

# Create mock streamlit module
import sys
sys.modules['streamlit'] = MockStreamlit()
import streamlit as st

# Now import our tool
from stats_compass.tools.chart_tools import CreateFeatureImportanceChartTool

def test_feature_importance_chart():
    print("🧪 Testing CreateFeatureImportanceChartTool...")
    
    # Create sample dataset for logistic regression
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    feature_1 = np.random.normal(0, 1, n_samples)
    feature_2 = np.random.normal(0, 1, n_samples) 
    feature_3 = np.random.normal(0, 1, n_samples)
    feature_4 = np.random.normal(0, 1, n_samples)
    
    # Create target with known relationships
    # feature_1 and feature_2 should be important, feature_3 and feature_4 less so
    linear_combination = 2 * feature_1 + 1.5 * feature_2 + 0.5 * feature_3 + 0.1 * feature_4
    probabilities = 1 / (1 + np.exp(-linear_combination))  # Sigmoid
    target = np.random.binomial(1, probabilities, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_1': feature_1,
        'feature_2': feature_2, 
        'feature_3': feature_3,
        'feature_4': feature_4,
        'target': target
    })
    
    # Fit logistic regression model
    X = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Create coefficient data similar to what our ML tools produce
    coefficients = pd.DataFrame({
        'feature': ['feature_1', 'feature_2', 'feature_3', 'feature_4'],
        'coefficient': model.coef_[0],
        'odds_ratio': np.exp(model.coef_[0]),
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    # Mock session state with ML results (like our logistic regression tool would create)
    st.session_state.ml_model_results = {
        'logistic_regression': {
            'model_type': 'logistic_regression',
            'model': model,
            'scaler': None,
            'target_column': 'target',
            'feature_columns': ['feature_1', 'feature_2', 'feature_3', 'feature_4'],
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'coefficients': coefficients,
            'standardized': False
        }
    }
    
    st.session_state.current_response_charts = []
    
    # Create the tool and test it
    tool = CreateFeatureImportanceChartTool()
    
    # Test with odds ratios (default for logistic regression)
    print("\n📊 Testing odds ratios visualization...")
    result = tool._run(model_key="logistic_regression", top_n=4, show_odds_ratios=True, title="Test Feature Importance")
    
    print("Result:")
    print(result)
    
    # Check that chart was stored
    charts = st.session_state.current_response_charts
    if charts:
        chart = charts[-1]
        print(f"\n✅ Chart created: {chart['type']}")
        print(f"   Title: {chart['title']}")
        print(f"   Model type: {chart['model_type']}")
        print(f"   Interpretation type: {chart['interpretation_type']}")
        print(f"   Features shown: {len(chart['data'])}")
        
        # Show the data
        print("\n📊 Chart data:")
        print(chart['data'][['feature', 'odds_ratio', 'coefficient', 'abs_coefficient']].to_string())
    else:
        print("❌ No charts were created")
    
    # Test with coefficients instead of odds ratios
    print("\n📊 Testing coefficients visualization...")
    result2 = tool._run(model_key="logistic_regression", top_n=4, show_odds_ratios=False, title="Test Coefficients")
    
    print("Result:")
    print(result2)
    
    # Test with linear regression data
    print("\n📊 Testing with linear regression...")
    
    # Create linear regression mock data
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    
    y_continuous = linear_combination + np.random.normal(0, 0.5, n_samples)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_continuous[:len(X_train)])
    
    # Calculate confidence intervals (simplified)
    y_pred = lr_model.predict(X_train)
    residuals = y_continuous[:len(X_train)] - y_pred
    mse = np.mean(residuals**2)
    var_coef = mse * np.linalg.inv(X_train.T @ X_train).diagonal()
    se_coef = np.sqrt(var_coef)
    t_val = stats.t.ppf(0.975, len(X_train) - len(X_train.columns) - 1)
    
    lr_coefficients = pd.DataFrame({
        'feature': ['feature_1', 'feature_2', 'feature_3', 'feature_4'],
        'coefficient': lr_model.coef_,
        'abs_coefficient': np.abs(lr_model.coef_),
        'std_error': se_coef,
        'conf_int_lower': lr_model.coef_ - t_val * se_coef,
        'conf_int_upper': lr_model.coef_ + t_val * se_coef,
        'significant': (lr_model.coef_ - t_val * se_coef > 0) | (lr_model.coef_ + t_val * se_coef < 0)
    }).sort_values('abs_coefficient', ascending=False)
    
    st.session_state.ml_model_results['linear_regression'] = {
        'model_type': 'linear_regression',
        'model': lr_model,
        'target_column': 'continuous_target',
        'coefficients': lr_coefficients,
        'standardized': False
    }
    
    result3 = tool._run(model_key="linear_regression", top_n=4, title="Test Linear Regression")
    print("Linear Regression Result:")
    print(result3)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_feature_importance_chart()