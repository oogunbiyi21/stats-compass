# stats_compass/utils/data_cleaning.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


def analyze_missing_data(df: pd.DataFrame) -> Dict:
    """
    Comprehensive missing data analysis - our competitive advantage over ChatGPT.
    Returns detailed insights about missing data patterns.
    """
    missing_info = {}
    
    # Basic missing data stats
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_info['total_rows'] = len(df)
    missing_info['columns_with_missing'] = missing_counts[missing_counts > 0].to_dict()
    missing_info['missing_percentages'] = missing_percentages[missing_percentages > 0].to_dict()
    
    # Advanced pattern analysis
    missing_info['completely_empty_columns'] = missing_counts[missing_counts == len(df)].index.tolist()
    missing_info['mostly_missing'] = missing_percentages[missing_percentages > 80].index.tolist()
    missing_info['partially_missing'] = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 80)].index.tolist()
    
    # Missing data correlation (which columns tend to be missing together)
    if len(missing_info['columns_with_missing']) > 1:
        missing_matrix = df.isnull()
        missing_corr = missing_matrix.corr()
        # Find high correlations in missing patterns
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    high_corr_pairs.append({
                        'col1': missing_corr.columns[i],
                        'col2': missing_corr.columns[j], 
                        'correlation': round(corr_val, 3)
                    })
        missing_info['correlated_missing_patterns'] = high_corr_pairs
    
    # Row-wise missing analysis
    rows_missing_counts = df.isnull().sum(axis=1)
    missing_info['rows_with_no_missing'] = (rows_missing_counts == 0).sum()
    missing_info['rows_completely_empty'] = (rows_missing_counts == len(df.columns)).sum()
    missing_info['avg_missing_per_row'] = round(rows_missing_counts.mean(), 2)
    
    return missing_info


def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict:
    """
    Advanced outlier detection using multiple methods.
    This systematic approach gives us advantage over ChatGPT's basic suggestions.
    """
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {'message': 'No numeric columns found for outlier detection'}
    
    outlier_info['method_used'] = method
    outlier_info['numeric_columns_analyzed'] = list(numeric_cols)
    outlier_info['outliers_by_column'] = {}
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = col_data[z_scores > 3]
            
        elif method == 'modified_zscore':
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad
            outliers = col_data[np.abs(modified_z_scores) > 3.5]
        
        outlier_info['outliers_by_column'][col] = {
            'count': len(outliers),
            'percentage': round((len(outliers) / len(col_data)) * 100, 2),
            'values': outliers.tolist()[:10] if len(outliers) <= 10 else outliers.tolist()[:10] + ['...'],
            'min_outlier': outliers.min() if len(outliers) > 0 else None,
            'max_outlier': outliers.max() if len(outliers) > 0 else None
        }
    
    # Summary statistics
    total_outliers = sum([info['count'] for info in outlier_info['outliers_by_column'].values()])
    outlier_info['total_outliers_found'] = total_outliers
    outlier_info['columns_with_outliers'] = [col for col, info in outlier_info['outliers_by_column'].items() if info['count'] > 0]
    
    return outlier_info


def find_duplicates(df: pd.DataFrame, subset_cols: Optional[list] = None) -> Dict:
    """
    Comprehensive duplicate detection and analysis.
    More thorough than ChatGPT's basic duplicate checking.
    """
    duplicate_info = {}
    
    # Complete row duplicates
    complete_duplicates = df.duplicated()
    duplicate_info['complete_duplicates'] = {
        'count': complete_duplicates.sum(),
        'percentage': round((complete_duplicates.sum() / len(df)) * 100, 2),
        'duplicate_rows': df[complete_duplicates].index.tolist() if complete_duplicates.sum() <= 20 else df[complete_duplicates].index.tolist()[:20] + ['...']
    }
    
    # Subset duplicates if specified
    if subset_cols:
        valid_cols = [col for col in subset_cols if col in df.columns]
        if valid_cols:
            subset_duplicates = df.duplicated(subset=valid_cols)
            duplicate_info['subset_duplicates'] = {
                'columns_checked': valid_cols,
                'count': subset_duplicates.sum(),
                'percentage': round((subset_duplicates.sum() / len(df)) * 100, 2),
                'duplicate_rows': df[subset_duplicates].index.tolist() if subset_duplicates.sum() <= 20 else df[subset_duplicates].index.tolist()[:20] + ['...']
            }
    
    # Column-wise duplicate analysis
    duplicate_info['column_analysis'] = {}
    for col in df.columns:
        col_duplicates = df[col].duplicated()
        unique_vals = df[col].nunique()
        duplicate_info['column_analysis'][col] = {
            'unique_values': unique_vals,
            'duplicate_count': col_duplicates.sum(),
            'uniqueness_ratio': round(unique_vals / len(df), 3)
        }
    
    # Identify potential key columns (high uniqueness)
    high_uniqueness_cols = [col for col, info in duplicate_info['column_analysis'].items() 
                           if info['uniqueness_ratio'] > 0.95]
    duplicate_info['potential_key_columns'] = high_uniqueness_cols
    
    # Identify low uniqueness columns (may need attention)
    low_uniqueness_cols = [col for col, info in duplicate_info['column_analysis'].items() 
                          if info['uniqueness_ratio'] < 0.1]
    duplicate_info['low_uniqueness_columns'] = low_uniqueness_cols
    
    return duplicate_info


def suggest_data_cleaning_actions(df: pd.DataFrame) -> Dict:
    """
    AI-powered data cleaning suggestions based on comprehensive analysis.
    This strategic recommendation system is our key differentiator.
    """
    suggestions = {
        'priority_actions': [],
        'optional_improvements': [],
        'data_quality_score': 0,
        'analysis_summary': {}
    }
    
    # Run all analyses
    missing_analysis = analyze_missing_data(df)
    outlier_analysis = detect_outliers(df)
    duplicate_analysis = find_duplicates(df)
    
    suggestions['analysis_summary'] = {
        'missing_data': missing_analysis,
        'outliers': outlier_analysis,
        'duplicates': duplicate_analysis
    }
    
    # Generate priority actions
    priority_score = 100
    
    # Critical issues (high priority)
    if missing_analysis.get('completely_empty_columns'):
        suggestions['priority_actions'].append({
            'action': 'Remove completely empty columns',
            'reason': f"Found {len(missing_analysis['completely_empty_columns'])} columns with no data",
            'columns': missing_analysis['completely_empty_columns'],
            'impact': 'High - reduces noise and improves analysis efficiency'
        })
        priority_score -= 20
    
    if duplicate_analysis['complete_duplicates']['count'] > 0:
        suggestions['priority_actions'].append({
            'action': 'Remove duplicate rows',
            'reason': f"Found {duplicate_analysis['complete_duplicates']['count']} duplicate rows ({duplicate_analysis['complete_duplicates']['percentage']}%)",
            'impact': 'High - prevents skewed analysis results'
        })
        priority_score -= 15
    
    if missing_analysis.get('mostly_missing'):
        suggestions['priority_actions'].append({
            'action': 'Consider removing high-missing columns',
            'reason': f"Columns with >80% missing data: {missing_analysis['mostly_missing']}",
            'columns': missing_analysis['mostly_missing'],
            'impact': 'Medium - may not provide reliable insights'
        })
        priority_score -= 10
    
    # Optional improvements
    if outlier_analysis.get('total_outliers_found', 0) > 0:
        suggestions['optional_improvements'].append({
            'action': 'Investigate outliers',
            'reason': f"Found {outlier_analysis['total_outliers_found']} outliers across {len(outlier_analysis.get('columns_with_outliers', []))} columns",
            'impact': 'Medium - may indicate data quality issues or interesting patterns'
        })
        priority_score -= 5
    
    if missing_analysis.get('partially_missing'):
        suggestions['optional_improvements'].append({
            'action': 'Consider imputation strategies',
            'reason': f"Columns with partial missing data: {missing_analysis['partially_missing']}",
            'impact': 'Medium - can improve dataset completeness'
        })
        priority_score -= 5
    
    if duplicate_analysis.get('low_uniqueness_columns'):
        suggestions['optional_improvements'].append({
            'action': 'Review low-uniqueness columns',
            'reason': f"Columns with <10% unique values: {duplicate_analysis['low_uniqueness_columns']}",
            'impact': 'Low - may indicate categorical data that needs encoding'
        })
    
    suggestions['data_quality_score'] = max(0, priority_score)
    
    return suggestions


def apply_basic_cleaning(df: pd.DataFrame, actions: list) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply selected cleaning actions and return cleaned dataframe with summary.
    """
    cleaning_summary = {
        'actions_applied': [],
        'rows_before': len(df),
        'cols_before': len(df.columns),
        'rows_after': 0,
        'cols_after': 0,
        'changes_made': {}
    }
    
    cleaned_df = df.copy()
    
    for action in actions:
        if action == 'remove_empty_columns':
            empty_cols = cleaned_df.columns[cleaned_df.isnull().all()].tolist()
            cleaned_df = cleaned_df.drop(columns=empty_cols)
            cleaning_summary['actions_applied'].append(f"Removed {len(empty_cols)} empty columns")
            cleaning_summary['changes_made']['removed_empty_columns'] = empty_cols
            
        elif action == 'remove_duplicates':
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            after_count = len(cleaned_df)
            removed_count = before_count - after_count
            cleaning_summary['actions_applied'].append(f"Removed {removed_count} duplicate rows")
            cleaning_summary['changes_made']['removed_duplicates'] = removed_count
            
        elif action == 'remove_high_missing_columns':
            missing_percentages = (cleaned_df.isnull().sum() / len(cleaned_df)) * 100
            high_missing_cols = missing_percentages[missing_percentages > 80].index.tolist()
            cleaned_df = cleaned_df.drop(columns=high_missing_cols)
            cleaning_summary['actions_applied'].append(f"Removed {len(high_missing_cols)} high-missing columns")
            cleaning_summary['changes_made']['removed_high_missing_columns'] = high_missing_cols
    
    cleaning_summary['rows_after'] = len(cleaned_df)
    cleaning_summary['cols_after'] = len(cleaned_df.columns)
    cleaning_summary['rows_removed'] = cleaning_summary['rows_before'] - cleaning_summary['rows_after']
    cleaning_summary['cols_removed'] = cleaning_summary['cols_before'] - cleaning_summary['cols_after']
    
    return cleaned_df, cleaning_summary


def suggest_imputation_strategies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Suggest appropriate imputation strategies for each column with missing data.
    Returns recommendations based on data type and missing patterns.
    """
    imputation_strategies = {}
    missing_analysis = analyze_missing_data(df)
    
    for col, _ in missing_analysis['columns_with_missing'].items():
        missing_percentage = missing_analysis['missing_percentages'][col]
        
        # Skip if too much missing data
        if missing_percentage > 80:
            imputation_strategies[col] = {
                'recommended_strategy': 'remove_column',
                'reason': f'Too much missing data ({missing_percentage:.1f}%)',
                'alternatives': []
            }
            continue
        
        col_data = df[col].dropna()
        strategies = []
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric columns
            skewness = col_data.skew() if len(col_data) > 0 else 0
            
            if abs(skewness) < 0.5:  # Roughly normal distribution
                strategies = [
                    {'method': 'mean', 'reason': 'Data is roughly normally distributed'},
                    {'method': 'median', 'reason': 'Robust to outliers'},
                    {'method': 'mode', 'reason': 'Most frequent value'}
                ]
            else:
                strategies = [
                    {'method': 'median', 'reason': 'Data is skewed, median is more robust'},
                    {'method': 'mode', 'reason': 'Most frequent value'},
                    {'method': 'mean', 'reason': 'Simple average (may be affected by skewness)'}
                ]
        else:
            # Categorical columns
            unique_ratio = len(col_data.unique()) / len(col_data) if len(col_data) > 0 else 0
            
            if unique_ratio > 0.5:  # High cardinality
                strategies = [
                    {'method': 'mode', 'reason': 'Most frequent category'},
                    {'method': 'constant', 'reason': 'Fill with "Unknown" or "Missing"'},
                    {'method': 'forward_fill', 'reason': 'Use previous valid value'}
                ]
            else:
                strategies = [
                    {'method': 'mode', 'reason': 'Most frequent category (low cardinality)'},
                    {'method': 'constant', 'reason': 'Fill with "Unknown" or "Missing"'}
                ]
        
        # Add advanced strategies for all types
        if missing_percentage < 20:  # Low missing percentage
            strategies.append({'method': 'interpolation', 'reason': 'Linear interpolation for time series data'})
            strategies.append({'method': 'knn', 'reason': 'Use similar rows to predict missing values'})
        
        imputation_strategies[col] = {
            'recommended_strategy': strategies[0]['method'] if strategies else 'remove_column',
            'reason': strategies[0]['reason'] if strategies else 'No suitable strategy',
            'alternatives': strategies[1:] if len(strategies) > 1 else [],
            'missing_percentage': missing_percentage,
            'data_type': 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
        }
    
    return imputation_strategies


def apply_imputation(df: pd.DataFrame, imputation_config: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply imputation to specified columns using specified methods.
    
    Args:
        df: DataFrame to impute
        imputation_config: Dict mapping column names to imputation methods
                          e.g., {'age': 'mean', 'category': 'mode', 'income': 'median'}
    
    Returns:
        Tuple of (imputed_dataframe, imputation_summary)
    """
    imputed_df = df.copy()
    imputation_summary = {
        'imputed_columns': {},
        'total_values_imputed': 0,
        'rows_before': len(df),
        'rows_after': 0,
        'imputation_methods_used': {}
    }
    
    for col, method in imputation_config.items():
        if col not in df.columns:
            continue
            
        original_missing = imputed_df[col].isnull().sum()
        if original_missing == 0:
            continue
        
        try:
            if method == 'mean' and pd.api.types.is_numeric_dtype(imputed_df[col]):
                fill_value = imputed_df[col].mean()
                imputed_df[col] = imputed_df[col].fillna(fill_value)
                
            elif method == 'median' and pd.api.types.is_numeric_dtype(imputed_df[col]):
                fill_value = imputed_df[col].median()
                imputed_df[col] = imputed_df[col].fillna(fill_value)
                
            elif method == 'mode':
                mode_values = imputed_df[col].mode()
                if len(mode_values) > 0:
                    fill_value = mode_values[0]
                    imputed_df[col] = imputed_df[col].fillna(fill_value)
                    
            elif method == 'constant':
                # Use appropriate constant based on data type
                if pd.api.types.is_numeric_dtype(imputed_df[col]):
                    fill_value = 0
                else:
                    fill_value = 'Unknown'
                imputed_df[col] = imputed_df[col].fillna(fill_value)
                
            elif method == 'forward_fill':
                imputed_df[col] = imputed_df[col].fillna(method='ffill')
                fill_value = 'Forward Fill'
                
            elif method == 'backward_fill':
                imputed_df[col] = imputed_df[col].fillna(method='bfill')
                fill_value = 'Backward Fill'
                
            elif method == 'interpolation' and pd.api.types.is_numeric_dtype(imputed_df[col]):
                imputed_df[col] = imputed_df[col].interpolate()
                fill_value = 'Interpolated'
                
            elif method == 'drop_rows':
                # Remove rows with missing values in this column
                imputed_df = imputed_df.dropna(subset=[col])
                fill_value = 'Rows Dropped'
                
            else:
                continue  # Skip unsupported methods
            
            final_missing = imputed_df[col].isnull().sum()
            values_imputed = original_missing - final_missing
            
            imputation_summary['imputed_columns'][col] = {
                'original_missing': original_missing,
                'values_imputed': values_imputed,
                'final_missing': final_missing,
                'method_used': method,
                'fill_value': str(fill_value) if method not in ['forward_fill', 'backward_fill', 'interpolation', 'drop_rows'] else fill_value
            }
            
            imputation_summary['total_values_imputed'] += values_imputed
            imputation_summary['imputation_methods_used'][method] = imputation_summary['imputation_methods_used'].get(method, 0) + 1
            
        except Exception as e:
            imputation_summary['imputed_columns'][col] = {
                'error': str(e),
                'method_attempted': method
            }
    
    imputation_summary['rows_after'] = len(imputed_df)
    imputation_summary['rows_dropped'] = imputation_summary['rows_before'] - imputation_summary['rows_after']
    
    return imputed_df, imputation_summary


def auto_impute_missing_data(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Automatically impute missing data using recommended strategies.
    
    Args:
        df: DataFrame to impute
        aggressive: If True, apply imputation to columns with higher missing percentages
    
    Returns:
        Tuple of (imputed_dataframe, imputation_summary)
    """
    strategies = suggest_imputation_strategies(df)
    
    # Build imputation config based on recommendations
    imputation_config = {}
    
    for col, strategy_info in strategies.items():
        missing_pct = strategy_info['missing_percentage']
        recommended = strategy_info['recommended_strategy']
        
        # Apply conservative or aggressive strategy
        if aggressive:
            threshold = 50  # More aggressive threshold
        else:
            threshold = 30  # Conservative threshold
        
        if missing_pct <= threshold and recommended != 'remove_column':
            imputation_config[col] = recommended
    
    if not imputation_config:
        return df.copy(), {
            'message': 'No columns selected for imputation',
            'strategies_suggested': strategies
        }
    
    return apply_imputation(df, imputation_config)
