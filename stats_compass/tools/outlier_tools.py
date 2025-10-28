"""
Outlier handling tools for Stats Compass.

Provides safe, validated outlier treatment methods that avoid the complexity
and failure modes of custom pandas queries.
"""

from typing import Type, Optional, List, Literal
from pydantic import BaseModel, Field, PrivateAttr, validator
from langchain.tools.base import BaseTool
import pandas as pd
import numpy as np
import streamlit as st


class HandleOutliersInput(BaseModel):
    """Input schema for outlier handling tool."""
    
    column: str = Field(
        description="Column name to handle outliers in (must be numeric)"
    )
    
    method: Literal['cap', 'remove', 'winsorize', 'log_transform', 'clip_iqr'] = Field(
        default='cap',
        description="""
        Outlier handling method:
        - 'cap': Cap values at specified percentile (default: 99th)
        - 'remove': Remove rows with outliers
        - 'winsorize': Replace outliers with boundary values
        - 'log_transform': Apply log transformation (for right-skewed data)
        - 'clip_iqr': Cap at IQR boundaries (1.5 * IQR beyond Q1/Q3)
        """
    )
    
    percentile: float = Field(
        default=99,
        ge=50,
        le=100,
        description="Percentile for capping (50-100). Default: 99. Only used for 'cap' and 'winsorize' methods."
    )
    
    lower_percentile: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="Lower percentile for two-sided capping (0-50). If None, only caps upper tail."
    )
    
    create_new_column: bool = Field(
        default=False,
        description="If True, creates new column '{column}_cleaned' instead of modifying original"
    )
    
    @validator('percentile')
    def validate_percentile(cls, v):
        if v < 50 or v > 100:
            raise ValueError("Percentile must be between 50 and 100")
        return v


class HandleOutliersTool(BaseTool):
    """
    Handle outliers in numeric columns using validated statistical methods.
    
    This tool provides safe, tested outlier treatment that avoids the complexity
    and failure modes of custom pandas queries. All methods are designed to work
    reliably with the execution environment.
    
    Examples:
        # Cap extreme values at 99th percentile
        handle_outliers(column='price', method='cap', percentile=99)
        
        # Two-sided capping (1st to 99th percentile)
        handle_outliers(column='income', method='cap', percentile=99, lower_percentile=1)
        
        # Remove outliers using IQR method
        handle_outliers(column='age', method='clip_iqr')
        
        # Log transform right-skewed data
        handle_outliers(column='sales', method='log_transform')
        
        # Create cleaned column without modifying original
        handle_outliers(column='price', method='cap', create_new_column=True)
    """
    
    name: str = "handle_outliers"
    description: str = """
    Handle outliers in numeric columns using validated statistical methods.
    
    PREFERRED over run_pandas_query for outlier treatment.
    
    Methods:
    • cap - Cap values at percentile threshold (safest, preserves all rows)
    • clip_iqr - Cap using IQR method (robust, standard practice)
    • winsorize - Replace outliers with boundary values
    • remove - Remove rows with outliers (reduces dataset size)
    • log_transform - Log transformation for right-skewed data
    
    Use this instead of:
    • df['col'].clip()
    • df[df['col'] < threshold]
    • df['col'].apply(lambda...)
    
    This tool handles edge cases and provides clear feedback.
    """
    args_schema: Type[BaseModel] = HandleOutliersInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
    
    def _get_current_df(self) -> pd.DataFrame:
        """Get the most current dataframe from session state if available."""
        if hasattr(st, 'session_state') and 'df' in st.session_state:
            return st.session_state['df']
        return self._df
    
    def _validate_column(self, df: pd.DataFrame, column: str) -> Optional[str]:
        """
        Validate that column exists and is numeric.
        
        Returns:
            Error message if validation fails, None if valid
        """
        if column not in df.columns:
            available_cols = list(df.columns)
            return f"❌ Column '{column}' not found.\n💡 Available columns: {available_cols}"
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            dtype = df[column].dtype
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            return (
                f"❌ Column '{column}' must be numeric (currently: {dtype}).\n"
                f"💡 Numeric columns: {numeric_cols}"
            )
        
        return None
    
    def _cap_at_percentile(
        self, 
        df: pd.DataFrame, 
        column: str, 
        upper_pct: float,
        lower_pct: Optional[float] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Cap values at specified percentiles.
        
        Returns:
            Tuple of (modified_df, stats_dict)
        """
        df_result = df.copy()
        original_values = df_result[column].copy()
        
        # Calculate thresholds
        upper_threshold = df_result[column].quantile(upper_pct / 100)
        
        if lower_pct is not None:
            lower_threshold = df_result[column].quantile(lower_pct / 100)
            # Two-sided capping
            df_result[column] = df_result[column].clip(lower=lower_threshold, upper=upper_threshold)
            n_lower = (original_values < lower_threshold).sum()
            n_upper = (original_values > upper_threshold).sum()
            
            stats = {
                'method': f'cap_{lower_pct}_{upper_pct}',
                'lower_threshold': lower_threshold,
                'upper_threshold': upper_threshold,
                'n_lower_capped': int(n_lower),
                'n_upper_capped': int(n_upper),
                'total_capped': int(n_lower + n_upper),
                'pct_capped': (n_lower + n_upper) / len(df) * 100
            }
        else:
            # One-sided capping (upper only)
            df_result[column] = df_result[column].clip(upper=upper_threshold)
            n_capped = (original_values > upper_threshold).sum()
            
            stats = {
                'method': f'cap_{upper_pct}',
                'upper_threshold': upper_threshold,
                'n_capped': int(n_capped),
                'pct_capped': n_capped / len(df) * 100
            }
        
        return df_result, stats
    
    def _clip_iqr(self, df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
        """
        Cap values using IQR method (1.5 * IQR beyond Q1/Q3).
        
        Returns:
            Tuple of (modified_df, stats_dict)
        """
        df_result = df.copy()
        original_values = df_result[column].copy()
        
        Q1 = df_result[column].quantile(0.25)
        Q3 = df_result[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_result[column] = df_result[column].clip(lower=lower_bound, upper=upper_bound)
        
        n_lower = (original_values < lower_bound).sum()
        n_upper = (original_values > upper_bound).sum()
        
        stats = {
            'method': 'clip_iqr',
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_lower_capped': int(n_lower),
            'n_upper_capped': int(n_upper),
            'total_capped': int(n_lower + n_upper),
            'pct_capped': (n_lower + n_upper) / len(df) * 100
        }
        
        return df_result, stats
    
    def _remove_outliers(self, df: pd.DataFrame, column: str, percentile: float) -> tuple[pd.DataFrame, dict]:
        """
        Remove rows with outlier values.
        
        Returns:
            Tuple of (filtered_df, stats_dict)
        """
        threshold = df[column].quantile(percentile / 100)
        df_result = df[df[column] <= threshold].copy()
        
        n_removed = len(df) - len(df_result)
        
        stats = {
            'method': f'remove_{percentile}',
            'threshold': threshold,
            'n_removed': int(n_removed),
            'pct_removed': n_removed / len(df) * 100,
            'rows_before': len(df),
            'rows_after': len(df_result)
        }
        
        return df_result, stats
    
    def _winsorize(
        self, 
        df: pd.DataFrame, 
        column: str, 
        upper_pct: float,
        lower_pct: Optional[float] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Winsorize: replace outliers with boundary values.
        Similar to cap, but explicitly replaces rather than clips.
        
        Returns:
            Tuple of (modified_df, stats_dict)
        """
        # Winsorize is essentially the same as cap for our purposes
        return self._cap_at_percentile(df, column, upper_pct, lower_pct)
    
    def _log_transform(self, df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
        """
        Apply log transformation (log1p to handle zeros).
        
        Returns:
            Tuple of (modified_df, stats_dict)
        """
        df_result = df.copy()
        original_values = df_result[column].copy()
        
        # Check for negative values
        if (original_values < 0).any():
            return None, {
                'error': f"Cannot log-transform column '{column}' - contains negative values"
            }
        
        # Use log1p (log(1+x)) to handle zeros
        df_result[column] = np.log1p(original_values)
        
        stats = {
            'method': 'log_transform',
            'original_min': float(original_values.min()),
            'original_max': float(original_values.max()),
            'original_mean': float(original_values.mean()),
            'transformed_min': float(df_result[column].min()),
            'transformed_max': float(df_result[column].max()),
            'transformed_mean': float(df_result[column].mean()),
            'n_zeros': int((original_values == 0).sum())
        }
        
        return df_result, stats
    
    def _format_results(self, column: str, stats: dict, create_new_column: bool) -> str:
        """Format the results for display."""
        lines = ["🔧 **Outlier Handling Results**\n"]
        
        method_names = {
            'cap': 'Capping at Percentile',
            'clip_iqr': 'IQR Clipping',
            'remove': 'Outlier Removal',
            'winsorize': 'Winsorization',
            'log_transform': 'Log Transformation'
        }
        
        method_key = stats['method'].split('_')[0]
        method_display = method_names.get(method_key, stats['method'])
        
        lines.append(f"**Method:** {method_display}")
        lines.append(f"**Column:** {column}")
        
        if create_new_column:
            lines.append(f"**New Column Created:** {column}_cleaned")
        else:
            lines.append(f"**Action:** Modified '{column}' in-place")
        
        lines.append("")
        
        # Method-specific statistics
        # Check IQR first (it also has 'total_capped' but different structure)
        if 'IQR' in stats:
            lines.append("📊 **Statistics:**")
            lines.append(f"  • Q1: {stats['Q1']:.2f}")
            lines.append(f"  • Q3: {stats['Q3']:.2f}")
            lines.append(f"  • IQR: {stats['IQR']:.2f}")
            lines.append(f"  • Lower bound: {stats['lower_bound']:.2f}")
            lines.append(f"  • Upper bound: {stats['upper_bound']:.2f}")
            lines.append(f"  • Values capped: {stats['total_capped']} ({stats['pct_capped']:.1f}%)")
        
        elif 'total_capped' in stats:
            lines.append("📊 **Statistics:**")
            if 'lower_threshold' in stats:
                lines.append(f"  • Lower threshold: {stats['lower_threshold']:.2f}")
                lines.append(f"  • Upper threshold: {stats['upper_threshold']:.2f}")
                lines.append(f"  • Values capped below: {stats['n_lower_capped']}")
                lines.append(f"  • Values capped above: {stats['n_upper_capped']}")
            else:
                lines.append(f"  • Threshold: {stats['upper_threshold']:.2f}")
                lines.append(f"  • Values capped: {stats['n_capped']}")
            
            lines.append(f"  • Total affected: {stats.get('total_capped', stats.get('n_capped'))} rows ({stats['pct_capped']:.1f}%)")
        
        elif 'n_removed' in stats:
            lines.append("📊 **Statistics:**")
            lines.append(f"  • Threshold: {stats['threshold']:.2f}")
            lines.append(f"  • Rows removed: {stats['n_removed']} ({stats['pct_removed']:.1f}%)")
            lines.append(f"  • Dataset size: {stats['rows_before']} → {stats['rows_after']}")
        
        elif 'transformed_mean' in stats:
            lines.append("📊 **Statistics:**")
            lines.append(f"  • Original range: [{stats['original_min']:.2f}, {stats['original_max']:.2f}]")
            lines.append(f"  • Transformed range: [{stats['transformed_min']:.2f}, {stats['transformed_max']:.2f}]")
            lines.append(f"  • Original mean: {stats['original_mean']:.2f}")
            lines.append(f"  • Transformed mean: {stats['transformed_mean']:.2f}")
            if stats['n_zeros'] > 0:
                lines.append(f"  • Zero values handled: {stats['n_zeros']}")
        
        lines.append("")
        lines.append("✅ **Outliers handled successfully**")
        
        if not create_new_column:
            lines.append("\n💡 Use this cleaned dataset for your analysis.")
        else:
            lines.append(f"\n💡 Use column '{column}_cleaned' for outlier-free analysis.")
        
        return "\n".join(lines)

    def _run(
        self,
        column: str,
        method: str = 'cap',
        percentile: float = 99,
        lower_percentile: Optional[float] = None,
        create_new_column: bool = False
    ) -> str:
        """
        Handle outliers in a numeric column.
        
        Args:
            column: Column name to process
            method: Outlier handling method
            percentile: Upper percentile for capping (50-100)
            lower_percentile: Lower percentile for two-sided capping (0-50)
            create_new_column: If True, create new column instead of modifying original
            
        Returns:
            Formatted string with results and statistics
        """
        try:
            # Get current dataframe
            df = self._get_current_df()
            
            # Validate column
            validation_error = self._validate_column(df, column)
            if validation_error:
                return validation_error
            
            # Determine target column name
            target_column = f"{column}_cleaned" if create_new_column else column
            
            # Copy data if creating new column
            if create_new_column:
                df[target_column] = df[column].copy()
                working_column = target_column
            else:
                working_column = column
            
            # Apply the selected method
            if method == 'cap':
                df_result, stats = self._cap_at_percentile(
                    df, working_column, percentile, lower_percentile
                )
            
            elif method == 'clip_iqr':
                df_result, stats = self._clip_iqr(df, working_column)
            
            elif method == 'remove':
                df_result, stats = self._remove_outliers(df, working_column, percentile)
            
            elif method == 'winsorize':
                df_result, stats = self._winsorize(
                    df, working_column, percentile, lower_percentile
                )
            
            elif method == 'log_transform':
                df_result, stats = self._log_transform(df, working_column)
                
                # Check for errors
                if df_result is None:
                    return f"❌ {stats['error']}"
            
            else:
                return f"❌ Unknown method: {method}. Use: cap, clip_iqr, remove, winsorize, or log_transform"
            
            # Update session state
            if hasattr(st, 'session_state'):
                st.session_state['df'] = df_result
            
            # Update internal dataframe
            self._df = df_result
            
            # Track in workflow state
            if hasattr(st, 'session_state') and 'workflow_state' in st.session_state:
                workflow_state = st.session_state['workflow_state']
                if not hasattr(workflow_state, 'outliers_handled'):
                    workflow_state.outliers_handled = {}
                workflow_state.outliers_handled[column] = {
                    'method': method,
                    'stats': stats
                }
            
            # Format and return results
            return self._format_results(working_column, stats, create_new_column)
            
        except Exception as e:
            import traceback
            return (
                f"❌ Error handling outliers: {type(e).__name__}: {str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()[-500:]}"
            )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
