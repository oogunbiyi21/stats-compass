#!/usr/bin/env python3
"""
Quick test of linear regression implementation
"""

import pandas as pd
import numpy as np
from ds_auto_insights.tools.ml_regression_tools import RunLinearRegressionTool

def create_test_data():
    """Create a simple test dataset for regression."""
    np.random.seed(42)
    n = 100
    
    # Create features
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(2, 0.5, n)
    x3 = np.random.normal(-1, 2, n)
    
    # Create target with known relationship
    noise = np.random.normal(0, 0.1, n)
    y = 2.5 + 1.5*x1 - 0.8*x2 + 0.3*x3 + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_1': x1,
        'feature_2': x2,
        'feature_3': x3,
        'target': y
    })
    
    return df

def test_linear_regression():
    """Test the linear regression tool."""
    print("🧪 Testing Linear Regression Tool...")
    
    # Create test data
    df = create_test_data()
    print(f"✅ Created test dataset: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Initialize tool
    tool = RunLinearRegressionTool(df=df)
    print("✅ Initialized RunLinearRegressionTool")
    
    # Test the tool
    result = tool._run(
        target_column='target',
        feature_columns=['feature_1', 'feature_2', 'feature_3'],
        test_size=0.2,
        standardize_features=False
    )
    
    print("✅ Linear regression completed!")
    print("📊 Results:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        test_linear_regression()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()