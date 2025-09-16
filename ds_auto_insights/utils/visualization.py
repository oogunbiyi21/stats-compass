# ds_auto_insights/utils/visualization.py

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
            
    else:
        st.warning(f"Chart type '{chart_type}' not supported yet")
        st.json(chart_info)  # Show raw data for debugging
