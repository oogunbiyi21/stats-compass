# stats_compass/tools/ml_util_tools.py
"""
Machine Learning utility tools for data preprocessing and feature engineering.
Provides simple, effective preprocessing techniques for ML workflows.
"""

from typing import Type, List
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool
from sklearn.preprocessing import TargetEncoder


class MeanTargetEncodingInput(BaseModel):
    categorical_columns: List[str] = Field(description="List of categorical column names to encode")
    target_column: str = Field(description="Target variable column name for computing means")
    cv: int = Field(default=5, description="Number of cross-validation folds to prevent target leakage")
    smooth: str = Field(default="auto", description="Smoothing strategy: 'auto' or float value")
    target_type: str = Field(default="auto", description="Target type: 'auto', 'continuous', or 'binary'")


class MeanTargetEncodingTool(BaseTool):
    """
    Simple mean target encoding tool for categorical variables.
    
    Converts categorical variables to numeric by replacing each category with the mean
    of the target variable for that category. Includes smoothing to prevent overfitting.
    """
    
    name: str = "mean_target_encoding"
    description: str = "Apply mean target encoding to categorical variables using target variable means with smoothing"
    args_schema: Type[BaseModel] = MeanTargetEncodingInput

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, categorical_columns: List[str], target_column: str, cv: int = 5, 
             smooth: str = "auto", target_type: str = "auto") -> str:
        try:
            # Validate inputs
            if target_column not in self._df.columns:
                return f"❌ Target column '{target_column}' not found. Available columns: {list(self._df.columns)}"
            
            missing_cols = [col for col in categorical_columns if col not in self._df.columns]
            if missing_cols:
                return f"❌ Categorical columns not found: {missing_cols}. Available columns: {list(self._df.columns)}"
            
            # Create a copy of the dataframe
            df_encoded = self._df.copy()
            
            # Handle target column - convert categorical to numeric if needed
            target_conversion_msg = ""
            if not pd.api.types.is_numeric_dtype(df_encoded[target_column]):
                # Check if it's a categorical that can be converted
                unique_values = df_encoded[target_column].dropna().unique()
        
                if len(unique_values) == 2:
                    # Binary categorical - map to 0/1
                    sorted_values = sorted(unique_values.astype(str))
                    value_map = {sorted_values[0]: 0, sorted_values[1]: 1}
                    df_encoded[target_column] = df_encoded[target_column].map(value_map)
                    target_conversion_msg = f"🔄 Converted binary categorical target '{target_column}' to numeric: {dict(zip(value_map.keys(), value_map.values()))}\n"
                elif len(unique_values) == 1:
                    # Single unique value - edge case
                    value_map = {unique_values[0]: 0}
                    df_encoded[target_column] = df_encoded[target_column].map(value_map)
                    target_conversion_msg = f"⚠️ Target '{target_column}' has only one unique value '{unique_values[0]}' - mapped to 0\n"
                # For multi-class targets, sklearn's TargetEncoder will handle it automatically
            
            # Handle continuous targets with too many unique values by binning
            binning_msg = ""
            unique_target_count = df_encoded[target_column].nunique()
            if pd.api.types.is_numeric_dtype(df_encoded[target_column]) and unique_target_count > 50:
                # For continuous targets with many unique values, bin them to prevent column explosion
                target_values = df_encoded[target_column].dropna()
                
                # Use quantile-based binning to create more balanced bins
                n_bins = min(20, max(5, unique_target_count // 10))  # Reasonable number of bins
                
                try:
                    df_encoded[f'{target_column}_original'] = df_encoded[target_column].copy()
                    df_encoded[target_column] = pd.qcut(
                        df_encoded[target_column], 
                        q=n_bins, 
                        labels=False, 
                        duplicates='drop'
                    )
                    binning_msg = f"📊 Binned continuous target '{target_column}' from {unique_target_count} unique values to {n_bins} quantile-based bins\n"
                except Exception as e:
                    # If quantile binning fails, use equal-width binning
                    df_encoded[f'{target_column}_original'] = df_encoded[target_column].copy()
                    df_encoded[target_column] = pd.cut(
                        df_encoded[target_column], 
                        bins=n_bins, 
                        labels=False
                    )
                    binning_msg = f"📊 Binned continuous target '{target_column}' from {unique_target_count} unique values to {n_bins} equal-width bins\n"
            
            # Remove target column from categorical columns if it was mistakenly included
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
            
            # Validate categorical columns exist and are actually categorical
            valid_categorical_columns = []
            for col in categorical_columns:
                if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                    valid_categorical_columns.append(col)
                else:
                    return f"❌ Column '{col}' appears to be numeric, not categorical. Unique values: {df_encoded[col].nunique()}"
            
            if not valid_categorical_columns:
                return "❌ No valid categorical columns found for encoding."
            
            # Prepare smooth parameter for sklearn
            smooth_param = smooth
            if smooth == "auto":
                smooth_param = "auto"
            else:
                try:
                    smooth_param = float(smooth)
                except (ValueError, TypeError):
                    smooth_param = "auto"
            
            # Create and configure the TargetEncoder
            encoder = TargetEncoder(
                categories='auto',
                target_type=target_type,
                smooth=smooth_param,
                cv=cv,
                shuffle=True,
                random_state=42
            )
            
            # Fit and transform the categorical columns
            X_categorical = df_encoded[valid_categorical_columns]
            y_target = df_encoded[target_column]
            
            # Handle missing values by filling with a placeholder
            X_categorical_filled = X_categorical.fillna('_MISSING_')
            
            # Apply target encoding
            encoded_features = encoder.fit_transform(X_categorical_filled, y_target)
            
            # Create encoded column names - sklearn TargetEncoder always produces one column per input feature
            encoded_column_names = [f'{col}_encoded' for col in valid_categorical_columns]
            
            # Add encoded columns to dataframe
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_column_names, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            
            # Update the tool's dataframe with encoded columns
            self._df = df_encoded
            
            # Also update the main session state dataframe that all tools share
            if hasattr(st, 'session_state'):
                if 'df' in st.session_state:
                    st.session_state.df = df_encoded
                
                # Store detailed encoding information for reference
                if 'encoded_dataframes' not in st.session_state:
                    st.session_state.encoded_dataframes = {}
                
                encoding_key = f"target_encoded_{len(st.session_state.encoded_dataframes)}"
                st.session_state.encoded_dataframes[encoding_key] = {
                    'dataframe': df_encoded,
                    'encoding_type': 'sklearn_target_encoding',
                    'original_columns': valid_categorical_columns,
                    'encoded_columns': encoded_column_names,
                    'target_column': target_column,
                    'encoder': encoder,
                    'parameters': {
                        'cv': cv,
                        'smooth': smooth_param,
                        'target_type': target_type
                    }
                }
            
            # Generate summary report
            summary_lines = [
                f"🎯 **Mean Target Encoding Applied (sklearn TargetEncoder)**",
                f"",
            ]
            
            # Add target conversion message if applicable
            if target_conversion_msg:
                summary_lines.append(target_conversion_msg)
            
            # Add binning message if applicable
            if binning_msg:
                summary_lines.append(binning_msg)
            
            # Get unique target info
            unique_targets = len(np.unique(y_target))
            is_multiclass = unique_targets > 2
            
            summary_lines.extend([
                f"📊 **Encoding Summary:**",
                f"  • Target variable: {target_column} ({'multi-class' if is_multiclass else 'binary/continuous'})",
                f"  • Unique target values: {unique_targets}",
                f"  • Cross-validation folds: {cv}",
                f"  • Smoothing: {smooth_param}",
                f"  • Target type: {target_type}",
                f""
            ])
            
            for i, col in enumerate(valid_categorical_columns):
                unique_cats = X_categorical[col].nunique()
                summary_lines.extend([
                    f"📋 **{col} → {encoded_column_names[i]}:**",
                    f"  • Original categories: {unique_cats}",
                    f""
                ])
            
            summary_lines.extend([
                f"💡 **Usage Notes:**",
                f"  • Uses sklearn's TargetEncoder with cross-validation to prevent overfitting",
                f"  • Handles multi-class targets automatically" if is_multiclass else "  • Handles binary/continuous targets directly",
                f"  • Original columns preserved for reference",
                f"  • New encoded columns: {encoded_column_names}",
                f"  • Missing values handled automatically",
                f"  • Use encoded columns for ML models"
            ])
            
            return '\n'.join(summary_lines)
            
        except Exception as e:
            return f"❌ Error in mean target encoding: {str(e)}"

    def _arun(self, categorical_columns: List[str], target_column: str, cv: int = 5, 
              smooth: str = "auto", target_type: str = "auto"):
        raise NotImplementedError("Async not supported")
