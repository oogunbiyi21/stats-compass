# stats_compass/utils/visualization.py

import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional


def display_single_chart(chart_info: Dict[str, Any], chart_id: Optional[str] = None):
    """
    Display a single chart based on chart_info dictionary.
    
    Args:
        chart_info: Dictionary containing chart configuration and data
        chart_id: Optional unique identifier for the chart
    """
    chart_type = chart_info.get('type')
    data = chart_info.get('data')
    title = chart_info.get('title', 'Chart')
    
    # Generate unique ID for this chart
    if chart_id is None:
        chart_id = f"{chart_type}_{hash(title) % 10000}"
    
    # Debug information
    st.caption(f"🔍 Chart Type: {chart_type} | Data Shape: {data.shape if hasattr(data, 'shape') else 'N/A'} | Title: {title}")
    
    if chart_type == 'histogram':
        st.subheader(f"📊 {title}")
        st.bar_chart(data.set_index('bin_range')['count'], use_container_width=True)
        
    elif chart_type == 'bar':
        st.subheader(f"📊 {title}")
        st.bar_chart(data.set_index('category')['count'], use_container_width=True)
        
    elif chart_type == 'scatter':
        st.subheader(f"📊 {title}")
        if chart_info.get('color_column'):
            st.scatter_chart(
                data, 
                x=chart_info['x_column'], 
                y=chart_info['y_column'],
                color=chart_info['color_column'],
                use_container_width=True
            )
        else:
            st.scatter_chart(
                data, 
                x=chart_info['x_column'], 
                y=chart_info['y_column'],
                use_container_width=True
            )
        # Show correlation info
        corr = chart_info.get('correlation', 0)
        if abs(corr) > 0.7:
            st.success(f"🔍 Strong correlation: {corr:.3f}")
        elif abs(corr) > 0.3:
            st.info(f"📈 Moderate correlation: {corr:.3f}")
        else:
            st.caption(f"📊 Weak correlation: {corr:.3f}")
            
    elif chart_type == 'line':
        st.subheader(f"📊 {title}")
        st.line_chart(
            data.set_index(chart_info['x_column'])[chart_info['y_column']], 
            use_container_width=True
        )
        
    elif chart_type == 'time_series':
        st.subheader(f"📈 {title}")
        # Recreate the chart from data to avoid Plotly object serialization issues
        try:
            import plotly.express as px
            # Use the stored data to recreate the chart
            fig = px.line(
                data, 
                x='Date', 
                y='Value',
                title=title
            )
            
            # Apply styling from chart config if available
            chart_config = chart_info.get('chart_config', {})
            line_width = chart_config.get('line_width', 2)
            ylabel = chart_config.get('ylabel', 'Value')
            
            fig.update_traces(
                line=dict(width=line_width),
                hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
            )
            
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title=ylabel,
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{chart_id}")
        except ImportError:
            # Fallback if plotly not available
            st.line_chart(data.set_index('Date')['Value'], use_container_width=True)
            
    elif chart_type == 'correlation_heatmap':
        st.subheader(f"🔥 {title}")
        try:
            import plotly.express as px
            import pandas as pd
            
            # Recreate correlation matrix from stored data
            corr_matrix = chart_info.get('correlation_matrix', {})
            if corr_matrix:
                df_corr = pd.DataFrame(corr_matrix)
                
                # Recreate the heatmap
                fig = px.imshow(
                    df_corr,
                    text_auto=True,
                    aspect="auto",
                    title=title,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                
                # Apply styling
                fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textfont_size=10,
                    hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
                )
                
                cols_count = len(chart_info.get('columns', []))
                fig.update_layout(
                    title=dict(x=0.5, font=dict(size=16)),
                    height=max(400, cols_count * 40),
                    width=max(400, cols_count * 40)
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{chart_id}")
            else:
                st.error("No correlation matrix data available")
                
        except ImportError:
            # Fallback to simple dataframe display
            corr_matrix = chart_info.get('correlation_matrix', {})
            if corr_matrix:
                import pandas as pd
                df_corr = pd.DataFrame(corr_matrix)
                st.dataframe(df_corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))
    
    elif chart_type == 't_test':
        st.subheader(f"📊 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('fig')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"ttest_{chart_id}")
            else:
                st.warning("T-Test chart figure not available for display")
        except Exception as e:
            st.error(f"Error displaying T-Test chart: {e}")
            
    elif chart_type == 'z_test':
        st.subheader(f"📊 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('fig')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"ztest_{chart_id}")
            else:
                st.warning("Z-Test chart figure not available for display")
        except Exception as e:
            st.error(f"Error displaying Z-Test chart: {e}")
            
    elif chart_type == 'chi_square_test':
        st.subheader(f"📊 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('fig')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"chisquare_{chart_id}")
            else:
                st.warning("Chi-Square Test chart figure not available for display")
        except Exception as e:
            st.error(f"Error displaying Chi-Square Test chart: {e}")
    
    elif chart_type == 'regression_plot':
        st.subheader(f"📊 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"regression_{chart_id}")
                
                # Show performance metrics
                train_r2 = chart_info.get('train_r2', 0)
                test_r2 = chart_info.get('test_r2', 0)
                st.caption(f"📈 Training R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
            else:
                st.warning("Regression plot figure not available for display")
        except Exception as e:
            st.error(f"Error displaying regression plot: {e}")
            
    elif chart_type == 'residual_plot':
        st.subheader(f"📊 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"residual_{chart_id}")
            else:
                st.warning("Residual plot figure not available for display")
        except Exception as e:
            st.error(f"Error displaying residual plot: {e}")
            
    elif chart_type == 'coefficient_chart':
        st.subheader(f"📊 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"coefficient_{chart_id}")
                
                # Show interpretation note
                target_column = chart_info.get('target_column', 'target')
                st.caption(f"💡 Green bars = positive effect on {target_column}, Red bars = negative effect")
            else:
                st.warning("Coefficient chart figure not available for display")
        except Exception as e:
            st.error(f"Error displaying coefficient chart: {e}")
            
    elif chart_type == 'feature_importance_chart':
        st.subheader(f"📊 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{chart_id}")
                
                # Show interpretation note based on model type
                model_type = chart_info.get('model_type', 'unknown')
                interpretation_type = chart_info.get('interpretation_type', 'coefficients')
                target_column = chart_info.get('target_column', 'target')
                
                if model_type == 'logistic_regression' and interpretation_type == 'odds ratios':
                    st.caption(f"💡 Values > 1 increase odds, < 1 decrease odds of positive {target_column}")
                else:
                    st.caption(f"💡 Green bars = positive effect on {target_column}, Red bars = negative effect")
            else:
                st.warning("Feature importance chart figure not available for display")
        except Exception as e:
            st.error(f"Error displaying feature importance chart: {e}")
            
    elif chart_type == 'roc_curve':
        st.subheader(f"📈 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"roc_{chart_id}")
                
                # Show performance metrics
                data = chart_info.get('data', {})
                train_auc = data.get('train_auc', 0)
                test_auc = data.get('test_auc', 0)
                st.caption(f"📈 Training AUC: {train_auc:.3f} | Test AUC: {test_auc:.3f}")
            else:
                st.warning("ROC curve figure not available for display")
        except Exception as e:
            st.error(f"Error displaying ROC curve: {e}")
            
    elif chart_type == 'precision_recall_curve':
        st.subheader(f"📈 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"pr_{chart_id}")
                
                # Show performance metrics
                data = chart_info.get('data', {})
                train_ap = data.get('train_ap', 0)
                test_ap = data.get('test_ap', 0)
                baseline = data.get('positive_ratio', 0.5)
                st.caption(f"📈 Training AP: {train_ap:.3f} | Test AP: {test_ap:.3f} | Baseline: {baseline:.3f}")
            else:
                st.warning("Precision-recall curve figure not available for display")
        except Exception as e:
            st.error(f"Error displaying precision-recall curve: {e}")
            
    elif chart_type == 'arima_plot':
        st.subheader(f"📈 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"arima_{chart_id}")
                
                # Show performance metrics
                data = chart_info.get('data', {})
                rmse = data.get('rmse', 0)
                mae = data.get('mae', 0)
                aic = data.get('aic', 0)
                good_fit_pct = data.get('good_fit_percentage', 0)
                st.caption(f"📊 RMSE: {rmse:.3f} | MAE: {mae:.3f} | AIC: {aic:.1f} | Good Fit: {good_fit_pct:.1f}%")
            else:
                st.warning("ARIMA plot figure not available for display")
        except Exception as e:
            st.error(f"Error displaying ARIMA plot: {e}")
            
    elif chart_type == 'arima_forecast_plot':
        st.subheader(f"🔮 {title}")
        try:
            # Get the plotly figure from chart_info
            fig = chart_info.get('figure')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"arima_forecast_{chart_id}")
                
                # Show forecast metrics if available
                data = chart_info.get('data', {})
                forecast_steps = data.get('forecast_steps', 0)
                confidence_level = data.get('confidence_level', 95)
                st.caption(f"🔮 Forecast Steps: {forecast_steps} | Confidence Level: {confidence_level}%")
            else:
                st.warning("ARIMA forecast plot figure not available for display")
        except Exception as e:
            st.error(f"Error displaying ARIMA forecast plot: {e}")
            
    else:
        st.warning(f"Chart type '{chart_type}' not supported yet")
        st.json(chart_info)  # Show raw data for debugging
