# stats_compass/tools/ml_chart_tools.py
"""
Machine Learning chart and visualization tools for DS Auto Insights.
Provides specialized charting capabilities for ML model results using Plotly.
"""

from typing import Type, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool


class CreateFeatureImportanceChartInput(BaseModel):
    model_key: str = Field(default="logistic_regression", description="Key of the ML model results to visualize")
    top_n: int = Field(default=10, description="Number of top features to show")
    show_odds_ratios: bool = Field(default=True, description="Show odds ratios instead of raw coefficients for logistic regression")
    title: str = Field(default="", description="Custom title for the chart")


class CreateFeatureImportanceChartTool(BaseTool):
    name: str = "create_feature_importance_chart"
    description: str = "Creates feature importance chart from ML model results. Supports both linear and logistic regression with appropriate visualizations."
    args_schema: Type[BaseModel] = CreateFeatureImportanceChartInput

    def _run(self, model_key: str = "logistic_regression", top_n: int = 10, show_odds_ratios: bool = True, title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a regression analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            model_type = results['model_type']
            coefficients = results['coefficients']
            target_column = results['target_column']
            
            # Handle different model types
            if model_type == 'logistic_regression':
                # For logistic regression, we can show either coefficients or odds ratios
                if show_odds_ratios and 'odds_ratio' in coefficients.columns:
                    # Sort by absolute coefficient value to get most important features
                    chart_data = coefficients.sort_values('abs_coefficient', ascending=True).tail(top_n)
                    
                    # Create horizontal bar chart showing coefficients (not odds ratios) for better visualization
                    fig = go.Figure()
                    
                    # Color bars based on positive/negative coefficients  
                    colors = ['green' if x > 0 else 'red' for x in chart_data['coefficient']]
                    
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],  # Use coefficient values so negatives go left
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<br>Odds Ratio: %{customdata[0]:.3f}<extra></extra>',
                        customdata=chart_data[['odds_ratio']].values
                    ))
                    
                    chart_title = title or f"Feature Importance (Odds Ratios): {target_column}"
                    
                    # Calculate symmetric range around zero for better visualization
                    max_abs_coef = chart_data['coefficient'].abs().max()
                    x_range = [-max_abs_coef * 1.1, max_abs_coef * 1.1]
                    
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title='Coefficient Value (Log Odds)',
                        yaxis_title='Features',
                        hovermode='closest',
                        xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='black')
                    )
                    
                    # Add vertical line at zero (no effect)
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.8)
                    
                    value_column = 'coefficient'
                    interpretation_type = "odds ratios"
                    
                else:
                    # Show raw coefficients for logistic regression
                    chart_data = coefficients.sort_values('abs_coefficient', ascending=True).tail(top_n)
                    
                    # Create horizontal bar chart
                    fig = go.Figure()
                    
                    # Color bars based on positive/negative coefficients
                    colors = ['green' if x > 0 else 'red' for x in chart_data['coefficient']]
                    
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<br>Odds Ratio: %{customdata[0]:.3f}<extra></extra>',
                        customdata=chart_data[['odds_ratio']].values if 'odds_ratio' in chart_data.columns else [[0]] * len(chart_data)
                    ))
                    
                    chart_title = title or f"Feature Importance (Coefficients): {target_column}"
                    
                    # Calculate symmetric range around zero for better visualization
                    max_abs_coef = chart_data['coefficient'].abs().max()
                    x_range = [-max_abs_coef * 1.1, max_abs_coef * 1.1]
                    
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title='Coefficient Value',
                        yaxis_title='Features',
                        hovermode='closest',
                        xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='black')
                    )
                    
                    # Add vertical line at zero for reference
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.8)
                    
                    value_column = 'coefficient'
                    interpretation_type = "coefficients"
                    
            elif model_type == 'linear_regression':
                # For linear regression, show coefficients
                chart_data = coefficients.sort_values('abs_coefficient', ascending=True).tail(top_n)
                
                # Create horizontal bar chart
                fig = go.Figure()
                
                # Color bars based on positive/negative coefficients
                colors = ['green' if x > 0 else 'red' for x in chart_data['coefficient']]
                
                # Check if we have confidence intervals
                has_confidence_intervals = 'conf_int_lower' in chart_data.columns and 'conf_int_upper' in chart_data.columns
                
                if has_confidence_intervals:
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<br>95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>',
                        customdata=chart_data[['conf_int_lower', 'conf_int_upper']].values
                    ))
                else:
                    fig.add_trace(go.Bar(
                        y=chart_data['feature'],
                        x=chart_data['coefficient'],
                        orientation='h',
                        marker=dict(color=colors, opacity=0.7),
                        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>'
                    ))
                
                chart_title = title or f"Feature Importance: {target_column}"
                
                # Calculate symmetric range around zero for better visualization
                max_abs_coef = chart_data['coefficient'].abs().max()
                x_range = [-max_abs_coef * 1.1, max_abs_coef * 1.1]
                
                fig.update_layout(
                    title=chart_title,
                    xaxis_title='Coefficient Value',
                    yaxis_title='Features',
                    hovermode='closest',
                    xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='black')
                )
                
                # Add vertical line at zero for reference
                fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.8)
                
                value_column = 'coefficient'
                interpretation_type = "coefficients"
                
            else:
                return f"❌ Unsupported model type: {model_type}. Supported types: linear_regression, logistic_regression"
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'feature_importance_chart',
                    'title': chart_title,
                    'figure': fig,
                    'data': chart_data.to_dict('records') if hasattr(chart_data, 'to_dict') else chart_data,
                    'target_column': target_column,
                    'model_type': model_type,
                    'interpretation_type': interpretation_type,
                    'top_n': top_n
                }
                st.session_state.current_response_charts.append(chart_info)
            
            # Create more actionable interpretation
            top_features = chart_data.tail(3)  # Most important 3
            feature_interpretation = ""
            recommendations = ""
            
            if model_type == 'logistic_regression' and show_odds_ratios and 'odds_ratio' in coefficients.columns:
                # Get the most impactful feature for recommendations
                most_impactful = chart_data.tail(1).iloc[0]
                most_impactful_odds = most_impactful['odds_ratio']
                
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    odds_ratio = row['odds_ratio']
                    if odds_ratio > 1:
                        effect = f"increases odds by {((odds_ratio - 1) * 100):.1f}%"
                        action = "increase" if odds_ratio > 1.2 else "consider increasing"
                    else:
                        effect = f"decreases odds by {((1 - odds_ratio) * 100):.1f}%"
                        action = "decrease" if odds_ratio < 0.8 else "consider decreasing"
                    
                    feature_interpretation += f"  • **{row['feature']}**: 1 unit increase {effect} (OR: {odds_ratio:.3f})\n"
                
                # Add specific recommendation
                if most_impactful_odds > 1.5:
                    recommendations = f"\n🎯 **Primary Recommendation**: Focus on maximizing {most_impactful['feature']} - it has the strongest positive impact (OR: {most_impactful_odds:.3f})."
                elif most_impactful_odds < 0.67:
                    recommendations = f"\n🎯 **Primary Recommendation**: Focus on minimizing {most_impactful['feature']} - it has the strongest negative impact (OR: {most_impactful_odds:.3f})."
                else:
                    recommendations = f"\n🎯 **Primary Recommendation**: {most_impactful['feature']} shows the strongest effect. Consider strategic adjustments."
                    
            else:
                # For linear regression or coefficient view
                most_impactful = chart_data.tail(1).iloc[0]
                most_impactful_coef = most_impactful['coefficient']
                
                for _, row in top_features.iterrows():
                    direction = "increases" if row['coefficient'] > 0 else "decreases"
                    significance = ""
                    if 'significant' in row and row['significant']:
                        significance = " (✓ Significant)"
                    elif 'significant' in row:
                        significance = " (Not significant)"
                    
                    feature_interpretation += f"  • **{row['feature']}**: 1 unit increase {direction} {target_column} by {abs(row['coefficient']):.3f} units{significance}\n"
                
                # Add specific recommendation with coefficient comparison
                if len(chart_data) > 1:
                    second_most = chart_data.tail(2).iloc[0]
                    ratio = abs(most_impactful_coef) / abs(second_most['coefficient'])
                    action = "increase" if most_impactful_coef > 0 else "decrease"
                    
                    recommendations = f"\n🎯 **Primary Recommendation**: Prioritize {most_impactful['feature']} - it has {ratio:.1f}x more impact than {second_most['feature']}. Focus on {action}ing {most_impactful['feature']} for maximum {target_column} improvement."
                else:
                    action = "increase" if most_impactful_coef > 0 else "decrease"
                    recommendations = f"\n🎯 **Primary Recommendation**: Focus on {action}ing {most_impactful['feature']} for maximum {target_column} impact."
            
            chart_type_desc = "Odds Ratios" if (model_type == 'logistic_regression' and show_odds_ratios) else "Coefficients"
            
            summary = f"""📊 {chart_title}

🎯 Created feature importance chart showing top {min(top_n, len(coefficients))} features using {interpretation_type}.

🔍 Key Feature Effects:
{feature_interpretation.strip()}{recommendations}

💡 Chart Interpretation:
"""
            
            if model_type == 'logistic_regression' and show_odds_ratios:
                summary += """  • Green bars = increases odds of positive outcome
  • Red bars = decreases odds of positive outcome
  • Values > 1 = increases probability, < 1 = decreases probability
  • Distance from 1 indicates strength of effect"""
            else:
                summary += f"""  • Green bars = positive effect on {target_column}
  • Red bars = negative effect on {target_column}
  • Longer bars = stronger effect"""
                if model_type == 'linear_regression':
                    summary += "\n  • Focus on significant features (✓) for decisions"

            return summary
            
        except Exception as e:
            return f"❌ Error creating feature importance chart: {str(e)}"

    def _arun(self, model_key: str = "logistic_regression", top_n: int = 10, show_odds_ratios: bool = True, title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateROCCurveInput(BaseModel):
    model_key: str = Field(default="logistic_regression", description="Key of the classification model results to visualize")
    title: str = Field(default="", description="Custom title for the chart")


class CreateROCCurveTool(BaseTool):
    """
    Create ROC (Receiver Operating Characteristic) curve for binary classification models.
    Shows true positive rate vs false positive rate at various threshold settings.
    """
    
    name: str = "create_roc_curve"
    description: str = "Create ROC curve visualization for binary classification models showing model discrimination ability"
    args_schema: Type[BaseModel] = CreateROCCurveInput

    def _run(self, model_key: str = "logistic_regression", title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a classification analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            
            # Check if this is a classification model
            if 'y_train_proba' not in results:
                return f"❌ Model '{model_key}' does not appear to be a classification model. ROC curves require probability predictions."
            
            # Extract data
            y_train = results['y_train']
            y_train_proba = results['y_train_proba']
            y_test = results['y_test']
            y_test_proba = results['y_test_proba']
            target_column = results['target_column']
            
            # Import sklearn metrics
            from sklearn.metrics import roc_curve, auc
            
            # Calculate ROC curves
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
            roc_auc_train = auc(fpr_train, tpr_train)
            
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
            roc_auc_test = auc(fpr_test, tpr_test)
            
            # Create ROC curve plot
            fig = go.Figure()
            
            # Training ROC curve
            fig.add_trace(go.Scatter(
                x=fpr_train, y=tpr_train,
                mode='lines',
                name=f'Training ROC (AUC = {roc_auc_train:.3f})',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Training</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
            
            # Test ROC curve (if different from training)
            if len(y_test) != len(y_train) or not np.array_equal(y_test, y_train):
                fig.add_trace(go.Scatter(
                    x=fpr_test, y=tpr_test,
                    mode='lines',
                    name=f'Test ROC (AUC = {roc_auc_test:.3f})',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Test</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                ))
            
            # Random classifier line (diagonal)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier (AUC = 0.5)',
                line=dict(color='gray', dash='dash', width=1),
                hoverinfo='skip'
            ))
            
            chart_title = title or f"ROC Curve: {target_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title='False Positive Rate (1 - Specificity)',
                yaxis_title='True Positive Rate (Sensitivity)',
                hovermode='closest',
                width=600,
                height=500,
                showlegend=True
            )
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'roc_curve',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'train_fpr': fpr_train.tolist(),
                        'train_tpr': tpr_train.tolist(),
                        'train_auc': roc_auc_train,
                        'test_fpr': fpr_test.tolist(),
                        'test_tpr': tpr_test.tolist(),
                        'test_auc': roc_auc_test
                    },
                    'target_column': target_column,
                    'model_key': model_key
                }
                st.session_state.current_response_charts.append(chart_info)
            
            summary = f"""📈 {chart_title}

📊 Created ROC curve showing model's discrimination ability across all classification thresholds.

📈 Model Performance:
  • Training AUC = {roc_auc_train:.3f} ({self._interpret_auc(roc_auc_train)})
  • Test AUC = {roc_auc_test:.3f} ({self._interpret_auc(roc_auc_test)})

💡 AUC Interpretation:
  • AUC = 1.0: Perfect classifier
  • AUC = 0.5: Random classifier (diagonal line)
  • AUC > 0.8: Excellent performance
  • AUC 0.7-0.8: Good performance
  • AUC 0.6-0.7: Fair performance"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating ROC curve: {str(e)}"

    def _interpret_auc(self, auc_score):
        """Interpret AUC score"""
        if auc_score >= 0.9:
            return "Excellent"
        elif auc_score >= 0.8:
            return "Good"
        elif auc_score >= 0.7:
            return "Fair"
        elif auc_score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"

    def _arun(self, model_key: str = "logistic_regression", title: str = ""):
        raise NotImplementedError("Async not supported")


class CreatePrecisionRecallCurveInput(BaseModel):
    model_key: str = Field(default="logistic_regression", description="Key of the classification model results to visualize")
    title: str = Field(default="", description="Custom title for the chart")


class CreatePrecisionRecallCurveTool(BaseTool):
    """
    Create Precision-Recall curve for binary classification models.
    Useful for imbalanced datasets where positive class is rare.
    """
    
    name: str = "create_precision_recall_curve"
    description: str = "Create precision-recall curve for binary classification, especially useful for imbalanced datasets"
    args_schema: Type[BaseModel] = CreatePrecisionRecallCurveInput

    def _run(self, model_key: str = "logistic_regression", title: str = "") -> str:
        try:
            # Check if ML results exist in session state
            if not hasattr(st, 'session_state') or 'ml_model_results' not in st.session_state:
                return "❌ No ML model results found. Run a classification analysis first."
                
            if model_key not in st.session_state.ml_model_results:
                available_keys = list(st.session_state.ml_model_results.keys())
                return f"❌ Model '{model_key}' not found. Available models: {available_keys}"
            
            # Get model results
            results = st.session_state.ml_model_results[model_key]
            
            # Check if this is a classification model
            if 'y_train_proba' not in results:
                return f"❌ Model '{model_key}' does not appear to be a classification model. PR curves require probability predictions."
            
            # Extract data
            y_train = results['y_train']
            y_train_proba = results['y_train_proba']
            y_test = results['y_test']
            y_test_proba = results['y_test_proba']
            target_column = results['target_column']
            
            # Import sklearn metrics
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            # Calculate Precision-Recall curves
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
            ap_score_train = average_precision_score(y_train, y_train_proba)
            
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
            ap_score_test = average_precision_score(y_test, y_test_proba)
            
            # Create Precision-Recall curve plot
            fig = go.Figure()
            
            # Training PR curve
            fig.add_trace(go.Scatter(
                x=recall_train, y=precision_train,
                mode='lines',
                name=f'Training PR (AP = {ap_score_train:.3f})',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Training</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
            ))
            
            # Test PR curve (if different from training)
            if len(y_test) != len(y_train) or not np.array_equal(y_test, y_train):
                fig.add_trace(go.Scatter(
                    x=recall_test, y=precision_test,
                    mode='lines',
                    name=f'Test PR (AP = {ap_score_test:.3f})',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Test</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
                ))
            
            # Baseline (random classifier for imbalanced data)
            positive_ratio = np.mean(y_train)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[positive_ratio, positive_ratio],
                mode='lines',
                name=f'Random Classifier (AP = {positive_ratio:.3f})',
                line=dict(color='gray', dash='dash', width=1),
                hoverinfo='skip'
            ))
            
            chart_title = title or f"Precision-Recall Curve: {target_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title='Recall (Sensitivity)',
                yaxis_title='Precision',
                hovermode='closest',
                width=600,
                height=500,
                showlegend=True
            )
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'precision_recall_curve',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'train_precision': precision_train.tolist(),
                        'train_recall': recall_train.tolist(),
                        'train_ap': ap_score_train,
                        'test_precision': precision_test.tolist(),
                        'test_recall': recall_test.tolist(),
                        'test_ap': ap_score_test,
                        'positive_ratio': positive_ratio
                    },
                    'target_column': target_column,
                    'model_key': model_key
                }
                st.session_state.current_response_charts.append(chart_info)
            
            summary = f"""📈 {chart_title}

📊 Created Precision-Recall curve showing trade-off between precision and recall.

📈 Model Performance:
  • Training AP = {ap_score_train:.3f} ({self._interpret_ap(ap_score_train, positive_ratio)})
  • Test AP = {ap_score_test:.3f} ({self._interpret_ap(ap_score_test, positive_ratio)})

💡 Interpretation:
  • Average Precision (AP) summarizes the curve
  • Higher AP = better performance
  • Baseline (random) = {positive_ratio:.3f} (proportion of positive class)
  • PR curves are better than ROC for imbalanced datasets"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating precision-recall curve: {str(e)}"

    def _interpret_ap(self, ap_score, baseline):
        """Interpret Average Precision score"""
        # AP scores range from 0 to 1, with 1 being perfect
        # Compare to baseline but use absolute thresholds for interpretation
        if ap_score >= 0.9:
            return "Excellent"
        elif ap_score >= 0.8:
            return "Very Good"
        elif ap_score >= 0.7:
            return "Good"
        elif ap_score >= 0.6:
            return "Fair"
        elif ap_score > baseline:
            return "Better than random"
        else:
            return "Poor"

    def _arun(self, model_key: str = "logistic_regression", title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateARIMAPlotInput(BaseModel):
    arima_key: str = Field(default="", description="Key of the ARIMA model results to visualize (if empty, use most recent)")
    title: str = Field(default="", description="Custom title for the chart")


class CreateARIMAPlotTool(BaseTool):
    """
    Create ARIMA model fit visualization showing actual vs fitted values.
    Displays how well the ARIMA model captures the historical time series patterns.
    """
    
    name: str = "create_arima_plot"
    description: str = "Create ARIMA model fit plot showing actual vs fitted values for model performance assessment"
    args_schema: Type[BaseModel] = CreateARIMAPlotInput

    def _run(self, arima_key: str = "", title: str = "") -> str:
        try:
            # Check if ARIMA results exist in session state
            if not hasattr(st, 'session_state') or 'arima_results' not in st.session_state:
                return "❌ No ARIMA model results found. Run ARIMA analysis first using run_arima_analysis."
                
            # Get the ARIMA key to use
            if not arima_key:
                # Use the most recent ARIMA result
                if not st.session_state.arima_results:
                    return "❌ No ARIMA results available."
                arima_key = list(st.session_state.arima_results.keys())[-1]
            
            if arima_key not in st.session_state.arima_results:
                available_keys = list(st.session_state.arima_results.keys())
                return f"❌ ARIMA model '{arima_key}' not found. Available models: {available_keys}"
            
            # Get ARIMA results
            results = st.session_state.arima_results[arima_key]
            time_series = results['time_series']
            fitted_values = results['fitted_values']
            original_dates = results['original_dates']
            value_column = results['value_column']
            order = results['order']
            aic = results['aic']
            residuals = results['residuals']
            
            # Create time index for plotting
            time_index = pd.to_datetime(original_dates)
            
            # Create the plot
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=time_index,
                y=time_series,
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
            
            # Add fitted values (skip first few points that ARIMA can't fit due to differencing)
            fitted_dates = time_index[len(time_index) - len(fitted_values):]
            fig.add_trace(go.Scatter(
                x=fitted_dates,
                y=fitted_values,
                mode='lines',
                name='Fitted',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Fitted</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
            
            chart_title = title or f"ARIMA{order} Model Fit: {value_column}"
            fig.update_layout(
                title=chart_title,
                xaxis_title='Time',
                yaxis_title=value_column,
                hovermode='x unified',
                width=800,
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
                
                chart_info = {
                    'type': 'arima_plot',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'time_index': time_index.tolist(),
                        'actual_values': time_series.tolist(),
                        'fitted_values': fitted_values.tolist(),
                        'fitted_dates': fitted_dates.tolist()
                    },
                    'value_column': value_column,
                    'arima_key': arima_key,
                    'order': order,
                    'aic': aic
                }
                st.session_state.current_response_charts.append(chart_info)
            
            # Calculate model fit statistics
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            
            # Calculate percentage of points where fitted is close to actual
            # Use only overlapping period
            actual_subset = time_series[len(time_series) - len(fitted_values):]
            errors = np.abs(actual_subset - fitted_values)
            good_fit_pct = np.mean(errors <= np.std(actual_subset)) * 100
            
            summary = f"""📈 {chart_title}

📊 Created ARIMA model fit visualization showing actual vs fitted values.

📈 Model Fit Assessment:
  • RMSE: {rmse:.4f}
  • MAE: {mae:.4f}
  • AIC: {aic:.2f}
  • Good fit points: {good_fit_pct:.1f}% (within 1 std dev)

💡 Interpretation:
  • Blue line (actual): Historical time series data
  • Red dashed line (fitted): ARIMA model predictions
  • Closer alignment = better model fit
  • Use for assessing model quality before forecasting"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating ARIMA plot: {str(e)}"

    def _arun(self, arima_key: str = "", title: str = ""):
        raise NotImplementedError("Async not supported")


class CreateARIMAForecastPlotInput(BaseModel):
    arima_key: str = Field(default="", description="Key of the ARIMA results to visualize (leave empty for most recent)")
    forecast_steps: Optional[int] = Field(default=None, description="Number of data points to plot. If not specified, plots all available forecast periods from ARIMA analysis. Check the run_arima_analysis output to see how many periods were computed.")
    confidence_level: float = Field(default=0.95, description="Confidence level for forecast intervals (e.g., 0.95 for 95%)")
    title: str = Field(default="", description="Custom title for the chart")


class CreateARIMAForecastPlotTool(BaseTool):
    name: str = "create_arima_forecast_plot"
    description: str = "Creates ARIMA forecast plot showing future predictions with confidence intervals. Use forecast_steps to specify how many DATA POINTS to plot (check ARIMA output for computed periods). If omitted, shows all available forecast data."
    args_schema: Type[BaseModel] = CreateARIMAForecastPlotInput
    
    def _run(self, arima_key: str = "", forecast_steps: Optional[int] = None, confidence_level: float = 0.95, title: str = "") -> str:
        try:
            # Check if ARIMA results exist in session state
            if not hasattr(st, 'session_state') or 'arima_results' not in st.session_state:
                return "❌ No ARIMA results found. Run ARIMA analysis first using 'run_arima' tool."
            
            # Get the most recent ARIMA result if no key specified
            if not arima_key:
                if not st.session_state.arima_results:
                    return "❌ No ARIMA results available."
                arima_key = list(st.session_state.arima_results.keys())[-1]
            
            if arima_key not in st.session_state.arima_results:
                available_keys = list(st.session_state.arima_results.keys())
                return f"❌ ARIMA result '{arima_key}' not found. Available results: {available_keys}"
            
            # Get ARIMA results
            arima_result = st.session_state.arima_results[arima_key]
            model = arima_result['model']
            time_series = arima_result['time_series']
            p, d, q = arima_result['order']
            
            # Get pre-computed forecast from ARIMA (plot tool is a view, not a forecaster)
            stored_forecast_dates = arima_result.get('forecast_dates', [])
            stored_forecast_values = arima_result.get('forecast', [])
            
            # Default to showing all available forecast if not specified
            warning_msg = ""
            if forecast_steps is None:
                forecast_steps = len(stored_forecast_values)
            elif forecast_steps > len(stored_forecast_values):
                # Be forgiving: show what we have and warn the user
                actual_steps = len(stored_forecast_values)
                warning_msg = f"⚠️ Requested {forecast_steps} steps but only {actual_steps} available. Showing all {actual_steps} computed steps.\n\n"
                forecast_steps = actual_steps
            
            # Simply slice the pre-computed results (no regeneration)
            forecast_dates = pd.to_datetime(stored_forecast_dates[:forecast_steps])
            forecast_values = stored_forecast_values[:forecast_steps]
            
            # Get time series metadata
            last_date = time_series.index[-1]
            last_value = time_series.iloc[-1]
            
            # Get confidence intervals if available
            try:
                forecast_ci = model.get_forecast(steps=forecast_steps, alpha=1-confidence_level).conf_int()
                lower_ci = forecast_ci.iloc[:, 0]
                upper_ci = forecast_ci.iloc[:, 1]
                has_ci = True
            except:
                # If confidence intervals not available, create simple bounds
                forecast_std = np.std(time_series) * 0.5  # Conservative estimate
                lower_ci = forecast_values - 1.96 * forecast_std
                upper_ci = forecast_values + 1.96 * forecast_std
                has_ci = False
            
            # Simple historical data logic: show same number of days as forecast
            historical_data = time_series.tail(forecast_steps) if len(time_series) > forecast_steps else time_series
            
            # Create the forecast plot
            fig = go.Figure()
            
            # Add historical data (balanced with forecast period)
            fig.add_trace(go.Scatter(
                x=historical_data.index if hasattr(historical_data, 'index') else range(len(historical_data)),
                y=historical_data.values if hasattr(historical_data, 'values') else historical_data,
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y:.4f}<extra></extra>'
            ))
            
            # Create continuous forecast by including the last historical point
            # This ensures the forecast line connects seamlessly to the historical data
            forecast_x = [time_series.index[-1]] + list(forecast_dates)
            forecast_y = [last_value] + list(forecast_values)
            
            # Add forecast with connection point
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode='lines',
                name=f'Forecast (ARIMA({p},{d},{q}))',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Date</b>: %{x}<br><b>Forecast</b>: %{y:.4f}<extra></extra>'
            ))
            
            # Add confidence intervals (only for actual forecast, not the connection point)
            confidence_pct = int(confidence_level * 100)
            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(reversed(forecast_dates)),
                y=list(upper_ci) + list(reversed(lower_ci)),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_pct}% Confidence Interval',
                hoverinfo='skip'
            ))
            
            # Set chart title
            chart_title = title if title else f"ARIMA({p},{d},{q}) Forecast - {arima_result.get('column_name', 'Time Series')}"
            
            # Update layout with proper datetime formatting
            fig.update_layout(
                title=dict(text=chart_title, x=0.5, font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title="Value",
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d',
                    tickangle=-45
                ),
                hovermode='x unified',
                height=500,
                showlegend=True,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
            )
            
            # Store chart data for Streamlit rendering
            if hasattr(st, 'session_state'):
                if 'current_response_charts' not in st.session_state:
                    st.session_state.current_response_charts = []
            
                chart_info = {
                    'type': 'arima_forecast_plot',
                    'title': chart_title,
                    'figure': fig,
                    'data': {
                        'forecast_steps': forecast_steps,
                        'confidence_level': confidence_level,
                        'arima_order': f"({p},{d},{q})",
                        'has_confidence_intervals': has_ci
                    }
                }
                
                st.session_state.current_response_charts.append(chart_info)
            
            # Calculate forecast statistics
            forecast_mean = np.mean(forecast_values)
            forecast_std = np.std(forecast_values)
            trend_direction = "upward" if forecast_values[-1] > forecast_values[0] else "downward" if forecast_values[-1] < forecast_values[0] else "stable"
            
            # Calculate display information and safe datetime formatting
            historical_points_shown = len(historical_data)
            
            # Safe datetime formatting
            try:
                if hasattr(historical_data, 'index') and len(historical_data) > 0:
                    start_date_str = pd.to_datetime(historical_data.index[0]).strftime('%Y-%m-%d')
                else:
                    start_date_str = "N/A"
                
                if hasattr(forecast_dates, '__len__') and len(forecast_dates) > 0:
                    end_date_str = pd.to_datetime(forecast_dates[-1]).strftime('%Y-%m-%d')
                else:
                    end_date_str = "N/A"
            except:
                start_date_str = "N/A"
                end_date_str = "N/A"
            
            summary = f"""{warning_msg}🔮 {chart_title}

📈 Created ARIMA forecast plot showing {forecast_steps} steps from pre-computed ARIMA results.

📊 Balanced View:
  • Historical data shown: {historical_points_shown} days (matches forecast period)
  • Chart balance: 50/50 recent history vs future forecast
  • Date range: {start_date_str} to {end_date_str}

🔮 Forecast Summary:
  • Forecast steps displayed: {forecast_steps} (from ARIMA results)
  • Total steps available: {len(stored_forecast_values)}
  • Confidence level: {confidence_pct}%
  • Mean forecast: {forecast_mean:.4f}
  • Forecast std dev: {forecast_std:.4f}
  • Trend direction: {trend_direction}

💡 Interpretation:
  • Blue line: Recent historical data (same period as forecast)
  • Red dashed line: Pre-computed ARIMA forecast (continuous connection to historical data)
  • Shaded area: {confidence_pct}% confidence interval
  • ✅ Plot shows exact ARIMA results - no date regeneration
  • ✅ Time consistency: Uses pre-computed forecast dates from ARIMA analysis
  • Note: To plot more steps, re-run ARIMA with higher forecast_periods"""

            return summary
            
        except Exception as e:
            return f"❌ Error creating ARIMA forecast plot: {str(e)}"

    def _arun(self, arima_key: str = "", forecast_steps: int = 10, confidence_level: float = 0.95, title: str = ""):
        raise NotImplementedError("Async not supported")
