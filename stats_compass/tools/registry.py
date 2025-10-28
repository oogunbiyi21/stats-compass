# tools/registry.py
"""
Central registry for all analysis tools.
Provides organized access to tools by category.
"""

import pandas as pd
from typing import List

from tools.exploration_tools import (
    GroupByAggregateTool,
    TopCategoriesTool,
    HistogramTool,
    CorrelationMatrixTool,
    CreateHistogramChartTool,
)
from tools.query_tools import (
    InspectDataTool,
    ModifyColumnTool,
    FilterDataframeTool,
    GetSchemaTool,
    GetSampleRowsTool,
    DescribeColumnTool,
    DatasetPreviewTool,
)
from tools.chart_tools import (
    CreateBarChartTool,
    CreateScatterPlotTool,
    CreateLineChartTool,
    CreateColumnTool,
    TimeSeriesAnalysisTool,
    CreateCorrelationHeatmapTool,
    CreateRegressionPlotTool,
    CreateResidualPlotTool,
    CreateCoefficientChartTool
)
from tools.ml_chart_tools import (
    CreateFeatureImportanceChartTool,
    CreateROCCurveTool,
    CreatePrecisionRecallCurveTool,
    CreateARIMAPlotTool,
    CreateARIMAForecastPlotTool
)
from tools.data_cleaning_tools import (
    AnalyzeMissingDataTool,
    DetectOutliersTool,
    FindDuplicatesTool,
    ApplyBasicCleaningTool,
    SuggestDataCleaningActionsTool,
    SuggestImputationStrategiesTool,
    ApplyImputationTool
)
from tools.outlier_tools import (
    HandleOutliersTool
)
from tools.statistical_test_tools import (
    RunTTestTool,
    RunZTestTool,
    RunChiSquareTestTool
)
from tools.ml_regression_tools import (
    RunLinearRegressionTool,
    RunLogisticRegressionTool,
    FindOptimalARIMAParametersTool,
    RunARIMATool
)
from tools.ml_evaluation_tools import (
    EvaluateRegressionModelTool,
    EvaluateClassificationModelTool
)
from tools.ml_util_tools import (
    MeanTargetEncodingTool,
    BinRareCategoriesTool
)


class ToolRegistry:
    """Central registry for all data analysis tools"""
    
    @staticmethod
    def get_exploration_tools(df: pd.DataFrame) -> List:
        """Get data exploration and analysis tools"""
        return [
            # Core query/inspection tools (replaces run_pandas_query)
            InspectDataTool(df=df),
            ModifyColumnTool(df=df),
            FilterDataframeTool(df=df),
            # Basic inspection tools
            GetSchemaTool(df=df),
            GetSampleRowsTool(df=df),
            DescribeColumnTool(df=df),
            DatasetPreviewTool(df=df),
            # Analysis tools
            GroupByAggregateTool(df=df),
            TopCategoriesTool(df=df),
            HistogramTool(df=df),
            CorrelationMatrixTool(df=df),
            CreateHistogramChartTool(df=df),
        ]
    
    @staticmethod
    def get_chart_tools(df: pd.DataFrame) -> List:
        """Get chart creation and visualization tools"""
        return [
            CreateBarChartTool(df=df),
            CreateScatterPlotTool(df=df),
            CreateLineChartTool(df=df),
            TimeSeriesAnalysisTool(df=df),
            CreateCorrelationHeatmapTool(df=df),
        ]
    
    @staticmethod
    def get_transformation_tools(df: pd.DataFrame) -> List:
        """Get data transformation tools"""
        return [
            CreateColumnTool(df=df),
        ]
    
    @staticmethod
    def get_cleaning_tools(df: pd.DataFrame) -> List:
        """Get data cleaning and imputation tools"""
        return [
            AnalyzeMissingDataTool(df=df),
            DetectOutliersTool(df=df),
            HandleOutliersTool(df=df),  # ← New outlier handling tool
            FindDuplicatesTool(df=df),
            SuggestDataCleaningActionsTool(df=df),
            ApplyBasicCleaningTool(df=df),
            SuggestImputationStrategiesTool(df=df),
            ApplyImputationTool(df=df),
        ]
    
    @staticmethod
    def get_statistical_tools(df: pd.DataFrame) -> List:
        """Get statistical analysis tools"""
        return [
            RunTTestTool(df=df),
            RunZTestTool(df=df),
            RunChiSquareTestTool(df=df),
        ]
    
    @staticmethod
    def get_ml_tools(df: pd.DataFrame) -> List:
        """Get machine learning modeling tools"""
        return [
            RunLinearRegressionTool(df=df),
            RunLogisticRegressionTool(df=df),
            FindOptimalARIMAParametersTool(df=df),
            RunARIMATool(df=df),
            BinRareCategoriesTool(df=df),
            MeanTargetEncodingTool(df=df),
        ]
    
    @staticmethod
    def get_ml_evaluation_tools() -> List:
        """Get ML model evaluation tools (don't need df)"""
        return [
            EvaluateRegressionModelTool(),
            EvaluateClassificationModelTool(),
        ]
    
    @staticmethod
    def get_ml_chart_tools() -> List:
        """Get ML visualization tools (don't need df)"""
        return [
            CreateRegressionPlotTool(),
            CreateResidualPlotTool(),
            CreateCoefficientChartTool(),
            CreateFeatureImportanceChartTool(),
            CreateROCCurveTool(),
            CreatePrecisionRecallCurveTool(),
            CreateARIMAPlotTool(),
            CreateARIMAForecastPlotTool(),
        ]
    
    @staticmethod
    def get_all_tools(df: pd.DataFrame) -> List:
        """Get all tools for the agent"""
        return (
            ToolRegistry.get_exploration_tools(df) +
            ToolRegistry.get_chart_tools(df) +
            ToolRegistry.get_transformation_tools(df) +
            ToolRegistry.get_cleaning_tools(df) +
            ToolRegistry.get_statistical_tools(df) +
            ToolRegistry.get_ml_tools(df) +
            ToolRegistry.get_ml_evaluation_tools() +
            ToolRegistry.get_ml_chart_tools()
        )
