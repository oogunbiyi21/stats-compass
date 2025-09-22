"""
Test script to verify the fixes for chart export and regression issues
"""

# Test data creation
import pandas as pd
import numpy as np

# Create sample data for testing
np.random.seed(42)
n_samples = 100

# Generate sample data for linear regression
X_linear = np.random.randn(n_samples, 3)
y_linear = 2 * X_linear[:, 0] + 1.5 * X_linear[:, 1] - 0.5 * X_linear[:, 2] + np.random.randn(n_samples) * 0.5

# Generate sample data for logistic regression  
X_logistic = np.random.randn(n_samples, 3)
linear_combo = 1.5 * X_logistic[:, 0] + 0.8 * X_logistic[:, 1] - 0.3 * X_logistic[:, 2]
y_logistic = (linear_combo + np.random.randn(n_samples) * 0.3 > 0).astype(int)

# Create DataFrame
test_df = pd.DataFrame({
    'feature_1': X_linear[:, 0],
    'feature_2': X_linear[:, 1], 
    'feature_3': X_linear[:, 2],
    'target_linear': y_linear,
    'feature_1_log': X_logistic[:, 0],
    'feature_2_log': X_logistic[:, 1],
    'feature_3_log': X_logistic[:, 2],
    'target_binary': y_logistic
})

print("✅ Test data created successfully!")
print(f"Dataset shape: {test_df.shape}")
print(f"Linear target mean: {test_df['target_linear'].mean():.3f}")
print(f"Binary target distribution: {test_df['target_binary'].value_counts().to_dict()}")

# Save test data
test_df.to_csv('/Users/tunjiogunbiyi/Dev/stats-compass/test_regression_data.csv', index=False)
print("✅ Test data saved to test_regression_data.csv")

print("\n🧪 To test the fixes:")
print("1. Upload test_regression_data.csv to Stats Compass")
print("2. Run linear regression: target_linear ~ feature_1 + feature_2 + feature_3")
print("3. Run logistic regression: target_binary ~ feature_1_log + feature_2_log + feature_3_log") 
print("4. Test chart creation and export functionality")
print("5. Verify charts display automatically for both regression types")