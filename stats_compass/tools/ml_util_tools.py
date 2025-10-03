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

    def _get_current_df(self) -> pd.DataFrame:
        """Get the most current dataframe, checking session state for updates"""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state.df
        return self._df

    def _run(self, categorical_columns: List[str], target_column: str, cv: int = 5, 
             smooth: str = "auto", target_type: str = "auto") -> str:
        try:
            # Get the most current dataframe (includes any updates from imputation/cleaning)
            current_df = self._get_current_df()
            
            # Validate inputs
            if target_column not in current_df.columns:
                return f"❌ Target column '{target_column}' not found. Available columns: {list(current_df.columns)}"
            
            missing_cols = [col for col in categorical_columns if col not in current_df.columns]
            if missing_cols:
                return f"❌ Categorical columns not found: {missing_cols}. Available columns: {list(current_df.columns)}"
            
            # Create a copy of the dataframe early so we can reference it throughout
            df_encoded = current_df.copy()
            
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
            

            # Target type handling
            if pd.api.types.is_numeric_dtype(df_encoded[target_column]) and df_encoded[target_column].nunique() > 10:
                effective_target_type = "continuous"
            elif df_encoded[target_column].nunique() == 2:
                effective_target_type = "binary"
            else:
                effective_target_type = "multiclass" 


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
                target_type=effective_target_type,
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
            
            # Handle dynamic column naming based on actual output shape
            n_features = len(valid_categorical_columns)
            n_output_cols = encoded_features.shape[1]
            
            if n_output_cols == n_features:
                # Simple case: one column per feature (binary/continuous targets)
                encoded_column_names = [f'{col}_encoded' for col in valid_categorical_columns]
            else:
                # Multiclass case: multiple columns per feature
                n_classes = n_output_cols // n_features
                encoded_column_names = []
                for i, col in enumerate(valid_categorical_columns):
                    if n_classes > 1:
                        # Multiple columns per feature - name them with class suffixes
                        for class_idx in range(n_classes):
                            encoded_column_names.append(f'{col}_encoded_class_{class_idx}')
                    else:
                        encoded_column_names.append(f'{col}_encoded')
            
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
                        'target_type': target_type,
                        'effective_target_type': effective_target_type,
                        'n_output_columns': n_output_cols,
                        'n_input_features': n_features
                    }
                }
            
            # Generate summary report
            summary_lines = [
                f"🎯 **Mean Target Encoding Applied (sklearn TargetEncoder)**",
                f"",
            ]
            
            # Get unique target info
            y_target = df_encoded[target_column]
            unique_targets = len(np.unique(y_target))
            is_multiclass = unique_targets > 2
            n_features = len(valid_categorical_columns)
            n_output_cols = len(encoded_column_names)
            
            summary_lines.extend([
                f"📊 **Encoding Summary:**",
                f"  • Target variable: {target_column}",
                f"  • Unique target values: {unique_targets}",
                f"  • Target type used: {effective_target_type}",
                f"  • Original target type requested: {target_type}",
                f"  • Cross-validation folds: {cv}",
                f"  • Smoothing: {smooth_param}",
                f"  • Input features: {n_features}",
                f"  • Output columns: {n_output_cols}",
                f""
            ])
            
            # Show feature mapping details
            cols_per_feature = n_output_cols // n_features if n_features > 0 else 1
            for i, col in enumerate(valid_categorical_columns):
                unique_cats = df_encoded[col].nunique()
                if cols_per_feature > 1:
                    # Multiple columns per feature
                    encoded_cols_for_feature = [name for name in encoded_column_names if name.startswith(f'{col}_encoded')]
                    summary_lines.extend([
                        f"📋 **{col} → {len(encoded_cols_for_feature)} columns:**",
                        f"  • Original categories: {unique_cats}",
                        f"  • Encoded columns: {', '.join(encoded_cols_for_feature)}",
                        f""
                    ])
                else:
                    # Single column per feature
                    summary_lines.extend([
                        f"� **{col} → {encoded_column_names[i]}:**",
                        f"  • Original categories: {unique_cats}",
                        f""
                    ])
            
            summary_lines.extend([
                f"💡 **Usage Notes:**",
                f"  • Uses sklearn's TargetEncoder with cross-validation to prevent overfitting",
                f"  • Multiclass targets create multiple columns per feature (one per class)" if is_multiclass else "  • Binary/continuous targets create one column per feature",
                f"  • Multiple columns provide richer encoding for complex target relationships" if is_multiclass else "  • Single columns provide efficient encoding for simple targets",
                f"  • Original columns preserved for reference",
                f"  • Total new encoded columns: {n_output_cols}",
                f"  • Missing values handled automatically",
                f"  • Use encoded columns for ML models"
            ])
            
            return '\n'.join(summary_lines)
            
        except Exception as e:
            return f"❌ Error in mean target encoding: {str(e)}"

    def _arun(self, categorical_columns: List[str], target_column: str, cv: int = 5, 
              smooth: str = "auto", target_type: str = "auto"):
        raise NotImplementedError("Async not supported")

class BinRareCategoriesInput(BaseModel):
    categorical_columns: List[str] = Field(description="List of categorical column names to encode")
    target_column: str = Field(description="Target variable column name for computing means")
    threshold: float = Field(default=0.05, description="Percentage threshold below which categories are considered rare")

class BinRareCategoriesTool(BaseTool):
    """
    Simple tool to bin rare categories in categorical variables.
    
    Groups infrequent categories into a single 'Other' category based on a specified threshold.
    Helps reduce noise and improve model performance by limiting the number of unique categories.
    """
    
    name: str = "bin_rare_categories"
    description: str = "Bin rare categories in categorical variables into 'Other' based on frequency threshold"
    args_schema: Type[BaseModel] = BinRareCategoriesInput

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _get_current_df(self) -> pd.DataFrame:
        """Get the most current dataframe, checking session state for updates"""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state.df
        return self._df

    def _run(self, categorical_columns: List[str], target_column: str, threshold: float = 0.05) -> str:
        try:
            # Get the most current dataframe (includes any updates from imputation/cleaning)  
            current_df = self._get_current_df()
                
            # Validate inputs
            if target_column not in current_df.columns:
                return f"❌ Target column '{target_column}' not found. Available columns: {list(current_df.columns)}"
            
            missing_cols = [col for col in categorical_columns if col not in current_df.columns]
            if missing_cols:
                return f"❌ Categorical columns not found: {missing_cols}. Available columns: {list(current_df.columns)}"
            
            # Create a copy of the dataframe early so we can reference it throughout
            df_binned = current_df.copy()
            
            # Remove target column from categorical columns if it was mistakenly included
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
            
            # Validate categorical columns exist and detect data quality issues
            valid_categorical_columns = []
            data_quality_issues = []
            
            for col in categorical_columns:
                if df_binned[col].dtype == 'object' or df_binned[col].dtype.name == 'category':
                    # Check for corrupted data in object columns that should be numeric
                    unique_vals = df_binned[col].dropna().unique()
                    
                    # Check if this looks like a corrupted numeric column
                    numeric_looking = 0
                    corrupt_values = []
                    
                    for val in unique_vals:
                        try:
                            float(str(val))
                            numeric_looking += 1
                        except (ValueError, TypeError):
                            corrupt_values.append(val)
                    
                    # If most values are numeric but some aren't, flag as corrupted
                    if numeric_looking > 0 and corrupt_values:
                        total_vals = len(unique_vals)
                        if numeric_looking / total_vals > 0.8:  # 80% numeric suggests corruption
                            data_quality_issues.append(f"Column '{col}' appears to be a corrupted numeric column with invalid values: {corrupt_values}")
                            continue  # Skip this column
                    
                    valid_categorical_columns.append(col)
                else:
                    return f"❌ Column '{col}' appears to be numeric, not categorical. Unique values: {df_binned[col].nunique()}"
            
            # Report data quality issues
            if data_quality_issues:
                issues_report = "\n".join([f"⚠️ {issue}" for issue in data_quality_issues])
                return f"❌ Data quality issues detected:\n{issues_report}\n\n💡 Please clean the data first using data cleaning tools before binning categorical variables."
            
            if not valid_categorical_columns:
                return "❌ No valid categorical columns found for binning."
            
            # Binning rare categories
            rare_category_feature_list = []
            for col in valid_categorical_columns:
                value_counts = df_binned[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < threshold].index
                if rare_categories.any():
                    rare_category_feature_list.append(col)
                    # Replace rare categories with 'Other'
                    df_binned[col] = df_binned[col].replace(rare_categories, 'Other')
            
            binned_features_string = ', '.join(rare_category_feature_list) if rare_category_feature_list else 'None'

            # Also update the main session state dataframe that all tools share
            if hasattr(st, 'session_state'):
                if 'df' in st.session_state:
                    st.session_state.df = df_binned
                
                # Store detailed encoding information for reference
                if 'binned_dataframes' not in st.session_state:
                    st.session_state.binned_dataframes = {}
                
                binned_key = f"binned_{len(st.session_state.binned_dataframes)}"
                st.session_state.binned_dataframes[binned_key] = {
                    'dataframe': df_binned,
                    'original_columns': valid_categorical_columns,
                    'binned_features': binned_features_string,
                    'target_column': target_column,
                    'threshold': threshold,
                }

            # Generate summary report
            summary_lines = [
                f"🎯 **Rare category binning applied**",
                f"",
            ]
            
            
            summary_lines.extend([
                f"📊 **Binning Summary:**",
                f"  • Target variable: {target_column}",
                f"  • Categorical features: {valid_categorical_columns}",
                f"  • Binning threshold: {threshold} (categories with frequency below this are binned to 'Other')",
                f"  • Binned features: {binned_features_string}",
            ])

            for col in valid_categorical_columns:
                value_counts = df_binned[col].value_counts()
                summary_lines.extend([
                    f"📋 **{col}:**",
                    f"  • Unique categories after binning: {value_counts.nunique()}",
                    f"  • Categories and counts: {value_counts.to_dict()}",
                    f""
                ])
            
            summary_lines.extend([
                f"💡 **Usage Notes:**",
                f"  • Helps reduce noise from infrequent categories",
                f"  • Original columns preserved for reference",
                f"  • Use binned columns for ML models"
            ])
            
            return '\n'.join(summary_lines)
        except Exception as e:
            return f"❌ Error in binning rare categories: {str(e)}"
        
    def _arun(self, categorical_columns: List[str], target_column: str, threshold: float = 0.05):
        raise NotImplementedError("Async not supported")
    