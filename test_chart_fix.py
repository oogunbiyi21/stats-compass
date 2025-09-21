#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to check if chart issue is fixed
"""
import pandas as pd
import numpy as np
import sys
import os

# Mock streamlit session state for testing
class MockSessionState:
    def __init__(self):
        self.ml_model_results = {}
    
    def __contains__(self, key):
        return hasattr(self, key)

class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()
    
    def plotly_chart(self, fig, use_container_width=True):
        print("[CHART] Chart would be displayed here (mocked for testing)")

# Replace streamlit with mock
mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st

# Now import tools after mocking
from stats_compass.tools.ml_regression_tools import RunLinearRegressionTool
from stats_compass.tools.chart_tools import CreateRegressionPlotTool

# Ensure the imported st module uses our mock
import stats_compass.tools.ml_regression_tools as ml_module
ml_module.st = mock_st

def main():
    print("Testing linear regression with chart fix...")
    
    # Create test data with clear relationships
    np.random.seed(42)
    n = 100
    data = {
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'x3': np.random.normal(0, 1, n),
    }
    # Create target with strong relationship (R² should be ~0.8-0.9)
    data['y'] = 2 * data['x1'] + 1.5 * data['x2'] + 0.5 * data['x3'] + np.random.normal(0, 0.3, n)
    
    df = pd.DataFrame(data)
    
    print(f"[OK] Created test data: {len(df)} rows, 4 columns")
    print(f"   Target correlation with x1: {df['y'].corr(df['x1']):.3f}")
    print(f"   Target correlation with x2: {df['y'].corr(df['x2']):.3f}")
    print(f"   Target correlation with x3: {df['y'].corr(df['x3']):.3f}")
    
    # Initialize tools
    lr_tool = RunLinearRegressionTool(df)
    chart_tool = CreateRegressionPlotTool()
    
    # Run regression
    print("\n[RUNNING] Linear regression...")
    result = lr_tool._run(
        target_column='y',
        feature_columns=['x1', 'x2', 'x3'],
        test_size=0.2,
        standardize_features=False
    )
    
    # Extract R² values from output
    lines = result.split('\n')
    train_r2_line = [line for line in lines if 'Training R²' in line]
    test_r2_line = [line for line in lines if 'Test R²' in line]
    
    print("\n[PERFORMANCE] Regression Performance:")
    if train_r2_line:
        print(f"   {train_r2_line[0].strip()}")
    if test_r2_line:
        print(f"   {test_r2_line[0].strip()}")
    
    # Check session state
    print("\n[CHECK] Session State Check:")
    if hasattr(mock_st, 'session_state') and hasattr(mock_st.session_state, 'ml_model_results'):
        print(f"   [OK] Session state exists")
        print(f"   Available models: {list(mock_st.session_state.ml_model_results.keys())}")
        
        if 'linear_regression' in mock_st.session_state.ml_model_results:
            lr_data = mock_st.session_state.ml_model_results['linear_regression']
            print(f"   [OK] Linear regression data found")
            
            required_keys = ['train_r2', 'test_r2', 'y_train', 'y_test', 'y_train_pred', 'y_test_pred']
            missing_keys = [key for key in required_keys if key not in lr_data]
            
            if not missing_keys:
                print(f"   [OK] All required keys present for charts")
                print(f"   Train R²: {lr_data['train_r2']:.3f}")
                print(f"   Test R²: {lr_data['test_r2']:.3f}")
                
                # Test chart creation
                print("\n[CHART] Testing Chart Creation:")
                try:
                    chart_result = chart_tool._run(
                        model_key='linear_regression',
                        title='Test Regression Plot'
                    )
                    print(f"   [OK] Chart tool executed successfully")
                    print(f"   Chart output (first 100 chars): {chart_result[:100]}...")
                except Exception as e:
                    print(f"   [ERROR] Chart tool failed: {str(e)}")
            else:
                print(f"   [ERROR] Missing required keys: {missing_keys}")
        else:
            print(f"   [ERROR] No linear_regression results in session state")
    else:
        print(f"   [ERROR] No session state available")
    
    print("\n[COMPLETE] Chart fix verification complete!")

if __name__ == "__main__":
    main()