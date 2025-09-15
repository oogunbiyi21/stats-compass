# ds_auto_insights/tools/data_cleaning_tools.py
"""
Data cleaning tools for DS Auto Insights.
These tools provide systematic data cleaning capabilities that give us competitive advantage over ChatGPT.
"""

from typing import Type
import pandas as pd
import json
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools.base import BaseTool
import streamlit as st
from ds_auto_insights.utils.data_cleaning import (
    analyze_missing_data,
    detect_outliers,
    find_duplicates,
    suggest_data_cleaning_actions,
    apply_basic_cleaning,
    suggest_imputation_strategies,
    apply_imputation,
    auto_impute_missing_data
)


class AnalyzeMissingDataInput(BaseModel):
    pass


class AnalyzeMissingDataTool(BaseTool):
    name: str = "analyze_missing_data"
    description: str = "Performs comprehensive missing data analysis including patterns, correlations, and insights. Use this when you want to understand missing data structure in the dataset."
    args_schema: Type[BaseModel] = AnalyzeMissingDataInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self) -> str:
        
        try:
            missing_info = analyze_missing_data(self._df)
            
            # Format the analysis for the LLM
            summary = "🔍 **Missing Data Analysis**\n\n"
            
            summary += f"📊 **Overview:**\n"
            summary += f"- Total rows: {missing_info['total_rows']:,}\n"
            summary += f"- Columns with missing data: {len(missing_info['columns_with_missing'])}\n"
            summary += f"- Average missing values per row: {missing_info['avg_missing_per_row']}\n\n"
            
            if missing_info['completely_empty_columns']:
                summary += f"🚨 **Empty Columns:** {', '.join(missing_info['completely_empty_columns'])}\n\n"
            
            if missing_info['mostly_missing']:
                summary += f"⚠️ **High Missing (>80%):** {', '.join(missing_info['mostly_missing'])}\n\n"
            
            if missing_info['columns_with_missing']:
                summary += f"📋 **Missing Data by Column:**\n"
                for col, count in missing_info['columns_with_missing'].items():
                    pct = missing_info['missing_percentages'][col]
                    summary += f"- {col}: {count:,} missing ({pct:.1f}%)\n"
                summary += "\n"
            
            if missing_info.get('correlated_missing_patterns'):
                summary += f"🔗 **Correlated Missing Patterns:**\n"
                for pattern in missing_info['correlated_missing_patterns']:
                    summary += f"- {pattern['col1']} ↔ {pattern['col2']} (correlation: {pattern['correlation']})\n"
                summary += "\n"
            
            summary += f"💡 **Recommendations:**\n"
            if missing_info['completely_empty_columns']:
                summary += f"- Remove completely empty columns: {', '.join(missing_info['completely_empty_columns'])}\n"
            if missing_info['mostly_missing']:
                summary += f"- Consider removing high-missing columns: {', '.join(missing_info['mostly_missing'])}\n"
            if missing_info['partially_missing']:
                summary += f"- Consider imputation for partially missing columns: {', '.join(missing_info['partially_missing'])}\n"
            
            return summary
            
        except Exception as e:
            return f"❌ Error analyzing missing data: {str(e)}"

    def _arun(self):
        raise NotImplementedError("Async not supported")


class DetectOutliersInput(BaseModel):
    method: str = Field(default="iqr", description="Method for outlier detection: 'iqr', 'zscore', or 'modified_zscore'")


class DetectOutliersTool(BaseTool):
    name: str = "detect_outliers"
    description: str = "Detects outliers in numeric columns using various statistical methods. Useful for understanding data quality and identifying potential anomalies."
    args_schema: Type[BaseModel] = DetectOutliersInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, method: str = "iqr") -> str:
        
        try:
            outlier_info = detect_outliers(self._df, method=method)
            
            if 'message' in outlier_info:
                return f"ℹ️ {outlier_info['message']}"
            
            # Format the analysis for the LLM
            summary = f"🎯 **Outlier Detection ({method.upper()} method)**\n\n"
            
            summary += f"📊 **Overview:**\n"
            summary += f"- Total outliers found: {outlier_info['total_outliers_found']}\n"
            summary += f"- Columns with outliers: {len(outlier_info['columns_with_outliers'])}\n"
            summary += f"- Numeric columns analyzed: {len(outlier_info['numeric_columns_analyzed'])}\n\n"
            
            if outlier_info['outliers_by_column']:
                summary += f"📋 **Outliers by Column:**\n"
                for col, info in outlier_info['outliers_by_column'].items():
                    if info['count'] > 0:
                        summary += f"- **{col}**: {info['count']} outliers ({info['percentage']:.2f}%)\n"
                        if info['min_outlier'] is not None:
                            summary += f"  Range: {info['min_outlier']:.3f} to {info['max_outlier']:.3f}\n"
                        if len(info['values']) <= 10:
                            summary += f"  Values: {info['values']}\n"
                        else:
                            summary += f"  Sample values: {info['values'][:5]}...\n"
                summary += "\n"
            
            summary += f"💡 **Recommendations:**\n"
            if outlier_info['total_outliers_found'] > 0:
                summary += f"- Investigate the {outlier_info['total_outliers_found']} outliers to determine if they're errors or valid extreme values\n"
                summary += f"- Consider the impact of outliers on your analysis (they may skew statistical measures)\n"
                summary += f"- For machine learning, you may want to handle outliers through transformation or removal\n"
            else:
                summary += f"- No outliers detected using {method} method - data appears well-behaved\n"
            
            return summary
            
        except Exception as e:
            return f"❌ Error detecting outliers: {str(e)}"

    def _arun(self, method: str = "iqr"):
        raise NotImplementedError("Async not supported")


class FindDuplicatesInput(BaseModel):
    subset_columns: list = Field(default=None, description="List of column names to check for duplicates (optional)")


class FindDuplicatesTool(BaseTool):
    name: str = "find_duplicates"
    description: str = "Analyzes duplicate rows and provides insights about data uniqueness patterns. Can check complete duplicates or duplicates in specific columns."
    args_schema: Type[BaseModel] = FindDuplicatesInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, subset_columns: list = None) -> str:
        
        try:
            duplicate_info = find_duplicates(self._df, subset_columns)
            
            # Format the analysis for the LLM
            summary = f"🔄 **Duplicate Analysis**\n\n"
            
            # Complete duplicates
            summary += f"📊 **Complete Row Duplicates:**\n"
            summary += f"- Count: {duplicate_info['complete_duplicates']['count']}\n"
            summary += f"- Percentage: {duplicate_info['complete_duplicates']['percentage']:.2f}%\n\n"
            
            # Subset duplicates if specified
            if 'subset_duplicates' in duplicate_info:
                cols_checked = duplicate_info['subset_duplicates']['columns_checked']
                summary += f"📋 **Duplicates in {', '.join(cols_checked)}:**\n"
                summary += f"- Count: {duplicate_info['subset_duplicates']['count']}\n"
                summary += f"- Percentage: {duplicate_info['subset_duplicates']['percentage']:.2f}%\n\n"
            
            # Column uniqueness insights
            summary += f"🔍 **Column Uniqueness Analysis:**\n"
            for col, info in duplicate_info['column_analysis'].items():
                summary += f"- **{col}**: {info['unique_values']} unique values "
                summary += f"(uniqueness ratio: {info['uniqueness_ratio']:.3f})\n"
            summary += "\n"
            
            # Key insights
            if duplicate_info['potential_key_columns']:
                summary += f"🗝️ **Potential Key Columns** (high uniqueness): {', '.join(duplicate_info['potential_key_columns'])}\n\n"
            
            if duplicate_info['low_uniqueness_columns']:
                summary += f"📊 **Low Uniqueness Columns** (may be categorical): {', '.join(duplicate_info['low_uniqueness_columns'])}\n\n"
            
            # Recommendations
            summary += f"💡 **Recommendations:**\n"
            if duplicate_info['complete_duplicates']['count'] > 0:
                summary += f"- Remove {duplicate_info['complete_duplicates']['count']} duplicate rows to clean the dataset\n"
            if duplicate_info['potential_key_columns']:
                summary += f"- Consider using {', '.join(duplicate_info['potential_key_columns'])} as unique identifiers\n"
            if duplicate_info['low_uniqueness_columns']:
                summary += f"- Review {', '.join(duplicate_info['low_uniqueness_columns'])} for proper categorical encoding\n"
            
            return summary
            
        except Exception as e:
            return f"❌ Error analyzing duplicates: {str(e)}"

    def _arun(self, subset_columns: list = None):
        raise NotImplementedError("Async not supported")


class ApplyDataCleaningInput(BaseModel):
    actions: list = Field(description="List of cleaning actions: 'remove_empty_columns', 'remove_duplicates', 'remove_high_missing_columns'")


class ApplyBasicCleaningTool(BaseTool):
    name: str = "apply_data_cleaning"
    description: str = "Applies specified data cleaning actions and updates the dataset. Use this after analyzing data quality issues to clean the data."
    args_schema: Type[BaseModel] = ApplyDataCleaningInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, actions: list) -> str:
 
        
        try:
            cleaned_df, summary = apply_basic_cleaning(self._df, actions)
            
            # Update the session state with cleaned data
            st.session_state.df = cleaned_df
            
            # Format the cleaning summary for the LLM
            result = f"✅ **Data Cleaning Applied**\n\n"
            
            result += f"📊 **Summary:**\n"
            result += f"- Rows: {summary['rows_before']:,} → {summary['rows_after']:,} "
            result += f"(removed {summary['rows_removed']:,})\n"
            result += f"- Columns: {summary['cols_before']:,} → {summary['cols_after']:,} "
            result += f"(removed {summary['cols_removed']:,})\n\n"
            
            if summary['actions_applied']:
                result += f"🛠️ **Actions Applied:**\n"
                for action in summary['actions_applied']:
                    result += f"- {action}\n"
                result += "\n"
            
            if summary['changes_made']:
                result += f"📋 **Detailed Changes:**\n"
                changes = summary['changes_made']
                if 'removed_empty_columns' in changes:
                    cols = changes['removed_empty_columns']
                    result += f"- Removed empty columns: {', '.join(cols) if cols else 'None'}\n"
                if 'removed_duplicates' in changes:
                    count = changes['removed_duplicates']
                    result += f"- Removed duplicate rows: {count}\n"
                if 'removed_high_missing_columns' in changes:
                    cols = changes['removed_high_missing_columns']
                    result += f"- Removed high-missing columns: {', '.join(cols) if cols else 'None'}\n"
            
            result += f"\n🎉 **Dataset updated!** All subsequent analysis will use the cleaned data."
            
            return result
            
        except Exception as e:
            return f"❌ Error applying data cleaning: {str(e)}"

    def _arun(self, actions: list):
        raise NotImplementedError("Async not supported")


class SuggestDataCleaningInput(BaseModel):
    pass


class SuggestDataCleaningActionsTool(BaseTool):
    name: str = "suggest_data_cleaning"
    description: str = "Provides AI-powered data cleaning recommendations based on comprehensive analysis. Use this to get intelligent suggestions for improving data quality."
    args_schema: Type[BaseModel] = SuggestDataCleaningInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self) -> str:
        
        try:
            suggestions = suggest_data_cleaning_actions(self._df)
            
            # Format suggestions for the LLM
            result = f"🤖 **AI Data Cleaning Recommendations**\n\n"
            
            # Data quality score
            score = suggestions['data_quality_score']
            score_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            result += f"{score_emoji} **Data Quality Score: {score}/100**\n\n"
            
            # Priority actions
            if suggestions['priority_actions']:
                result += f"🚨 **Priority Actions (Recommended):**\n"
                for i, action in enumerate(suggestions['priority_actions'], 1):
                    result += f"{i}. **{action['action']}**\n"
                    result += f"   - Reason: {action['reason']}\n"
                    result += f"   - Impact: {action['impact']}\n"
                    if 'columns' in action:
                        result += f"   - Columns: {', '.join(action['columns'])}\n"
                    result += f"\n"
            
            # Optional improvements
            if suggestions['optional_improvements']:
                result += f"💡 **Optional Improvements:**\n"
                for i, action in enumerate(suggestions['optional_improvements'], 1):
                    result += f"{i}. **{action['action']}**\n"
                    result += f"   - Reason: {action['reason']}\n"
                    result += f"   - Impact: {action['impact']}\n\n"
            
            # Next steps
            result += f"🎯 **Recommended Next Steps:**\n"
            if suggestions['priority_actions']:
                result += f"1. Use the `apply_data_cleaning` tool with these actions: "
                actions = []
                for action in suggestions['priority_actions']:
                    if 'empty columns' in action['action'].lower():
                        actions.append('remove_empty_columns')
                    elif 'duplicate' in action['action'].lower():
                        actions.append('remove_duplicates')
                    elif 'high-missing' in action['action'].lower():
                        actions.append('remove_high_missing_columns')
                result += f"{actions}\n"
                result += f"2. Re-analyze the data quality after cleaning\n"
                result += f"3. Proceed with your analysis on the cleaned dataset\n"
            else:
                result += f"- Your data quality is already good! Proceed with analysis.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error generating cleaning suggestions: {str(e)}"

    def _arun(self):
        raise NotImplementedError("Async not supported")


class SuggestImputationStrategiesInput(BaseModel):
    pass


class SuggestImputationStrategiesTool(BaseTool):
    name: str = "suggest_imputation_strategies"
    description: str = "Analyzes missing data and suggests appropriate imputation strategies for each column. Use this when you need to decide how to handle missing values."
    args_schema: Type[BaseModel] = SuggestImputationStrategiesInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self) -> str:
        try:
            strategies = suggest_imputation_strategies(self._df)
            
            if not strategies:
                return "✅ No missing data found - no imputation needed!"
            
            result = "🔧 **Imputation Strategy Recommendations**\n\n"
            
            for col, strategy_info in strategies.items():
                result += f"**{col}** ({strategy_info['data_type']}, {strategy_info['missing_percentage']:.1f}% missing):\n"
                result += f"- 🎯 **Recommended:** {strategy_info['recommended_strategy']} - {strategy_info['reason']}\n"
                
                if strategy_info['alternatives']:
                    result += f"- 🔄 **Alternatives:**\n"
                    for alt in strategy_info['alternatives']:
                        result += f"  - {alt['method']}: {alt['reason']}\n"
                result += "\n"
            
            result += "💡 **Next Steps:**\n"
            result += "1. Use apply_imputation with mode='auto' for automatic imputation\n"
            result += "2. Or use apply_imputation with mode='custom' and your chosen methods\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error suggesting imputation strategies: {str(e)}"

    def _arun(self):
        raise NotImplementedError("Async not supported")


class ApplyImputationInput(BaseModel):
    mode: str = Field(default="auto", description="Imputation mode: 'auto' for AI recommendations or 'custom' for specific methods")
    imputation_config: str = Field(default="", description="For custom mode: JSON string mapping column names to methods, e.g., '{\"age\": \"mean\", \"category\": \"mode\"}'")
    aggressive: bool = Field(default=False, description="For auto mode: whether to use aggressive imputation (higher missing data thresholds)")


class ApplyImputationTool(BaseTool):
    name: str = "apply_imputation"
    description: str = "Apply imputation to missing data. Use 'auto' mode for AI recommendations or 'custom' mode for specific methods. Covers both automatic and manual imputation needs."
    args_schema: Type[BaseModel] = ApplyImputationInput

    _df: pd.DataFrame = PrivateAttr()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def _run(self, mode: str = "auto", imputation_config: str = "", aggressive: bool = False) -> str:
        try:
            if mode == "auto":
                # Use automatic imputation with AI recommendations
                imputed_df, summary = auto_impute_missing_data(self._df, aggressive=aggressive)
                
                if 'message' in summary:
                    return f"ℹ️ {summary['message']}\n\nSuggested strategies available - use suggest_imputation_strategies tool to see options."
                
                result = "🤖 **Automatic Imputation Results**\n\n"
                result += f"📊 **Summary:**\n"
                result += f"- Total values imputed: {summary['total_values_imputed']:,}\n"
                result += f"- Rows before: {summary['rows_before']:,}\n"
                result += f"- Rows after: {summary['rows_after']:,}\n"
                
                if summary['rows_dropped'] > 0:
                    result += f"- Rows dropped: {summary['rows_dropped']:,}\n"
                
                result += f"- Strategy: {'Aggressive' if aggressive else 'Conservative'}\n"
                
                result += f"\n📋 **Methods Used:**\n"
                for method, count in summary['imputation_methods_used'].items():
                    result += f"- {method}: {count} columns\n"
                
                result += f"\n🔍 **Imputed Columns:**\n"
                for col, details in summary['imputed_columns'].items():
                    if 'error' in details:
                        result += f"- ❌ {col}: Error - {details['error']}\n"
                    else:
                        result += f"- ✅ {col}: {details['values_imputed']:,} values ({details['method_used']})\n"
                
                result += f"\n💡 **Data has been automatically imputed!**\n"
                
            elif mode == "custom":
                # Use custom imputation configuration
                if not imputation_config:
                    return "❌ Custom mode requires imputation_config parameter with column-method mappings."
                
                
                config = json.loads(imputation_config)
                
                imputed_df, summary = apply_imputation(self._df, config)
                
                result = "🔧 **Custom Imputation Results**\n\n"
                result += f"📊 **Summary:**\n"
                result += f"- Total values imputed: {summary['total_values_imputed']:,}\n"
                result += f"- Rows before: {summary['rows_before']:,}\n"
                result += f"- Rows after: {summary['rows_after']:,}\n"
                
                if summary['rows_dropped'] > 0:
                    result += f"- Rows dropped: {summary['rows_dropped']:,}\n"
                
                result += f"\n📋 **Methods Used:**\n"
                for method, count in summary['imputation_methods_used'].items():
                    result += f"- {method}: {count} columns\n"
                
                result += f"\n🔍 **Column Details:**\n"
                for col, details in summary['imputed_columns'].items():
                    if 'error' in details:
                        result += f"- ❌ {col}: Error - {details['error']}\n"
                    else:
                        result += f"- ✅ {col}: {details['values_imputed']:,} values imputed using {details['method_used']}\n"
                        if 'fill_value' in details and details['method_used'] not in ['forward_fill', 'backward_fill', 'interpolation']:
                            result += f"  Fill value: {details['fill_value']}\n"
                
                result += f"\n💡 **Data has been imputed with your custom settings!**\n"
                
            else:
                return f"❌ Invalid mode '{mode}'. Use 'auto' for automatic imputation or 'custom' for specific methods."
            
            # Update the session state with imputed data
            st.session_state.df = imputed_df
            result += f"You can now proceed with analysis on the cleaned dataset.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error applying imputation: {str(e)}"

    def _arun(self):
        raise NotImplementedError("Async not supported")
