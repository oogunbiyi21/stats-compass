# stats_compass/utils/agent_transcript.py
"""
AI Agent Transcript utilities for logging and displaying agent reasoning process.
Provides insight into the agent's decision-making, tool usage, and thought process.
"""

import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class AgentTranscriptLogger:
    """
    Captures and formats AI agent reasoning and tool execution steps.
    """
    
    @staticmethod
    def format_intermediate_steps(intermediate_steps: List[tuple]) -> List[Dict[str, Any]]:
        """
        Format raw LangChain intermediate steps into readable transcript entries.
        
        Args:
            intermediate_steps: List of (AgentAction, observation) tuples from LangChain
            
        Returns:
            List of formatted transcript entries
        """
        formatted_steps = []
        
        for i, (action, observation) in enumerate(intermediate_steps):
            step_entry = {
                "step_number": i + 1,
                "timestamp": datetime.now().isoformat(),
                "tool_name": getattr(action, 'tool', 'unknown'),
                "tool_input": getattr(action, 'tool_input', {}),
                "reasoning": getattr(action, 'log', ''),
                "observation": str(observation)[:1000] + ("..." if len(str(observation)) > 1000 else ""),  # Truncate long outputs
                "raw_action": str(action)
            }
            formatted_steps.append(step_entry)
        
        return formatted_steps
    
    @staticmethod
    def create_transcript_summary(steps: List[Dict[str, Any]], final_output: str) -> Dict[str, Any]:
        """
        Create a summary of the agent's execution session.
        
        Args:
            steps: Formatted intermediate steps
            final_output: Agent's final response
            
        Returns:
            Transcript summary with metadata
        """
        tools_used = [step["tool_name"] for step in steps]
        unique_tools = list(set(tools_used))
        
        summary = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(steps),
            "tools_used": unique_tools,
            "tool_usage_count": {tool: tools_used.count(tool) for tool in unique_tools},
            "final_output": final_output,
            "steps": steps
        }
        
        return summary


class AgentTranscriptDisplay:
    """
    Streamlit components for displaying agent transcripts in user-friendly formats.
    """
    
    @staticmethod
    def render_transcript_expander(transcript_summary: Dict[str, Any], title: str = "🤖 AI Agent Reasoning"):
        """
        Render agent transcript in an expandable section.
        
        Args:
            transcript_summary: Formatted transcript summary
            title: Title for the expander section
        """
        with st.expander(title, expanded=False):
            # Summary section
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔧 Tools Used", len(transcript_summary["tools_used"]))
            with col2:
                st.metric("⚡ Total Steps", transcript_summary["total_steps"])
            with col3:
                st.metric("🕒 Session Time", transcript_summary["timestamp"].split("T")[1][:8])
            
            # Tools breakdown
            if transcript_summary["tool_usage_count"]:
                st.subheader("🛠️ Tool Usage Breakdown")
                tools_df = pd.DataFrame([
                    {"Tool": tool, "Usage Count": count, "Purpose": AgentTranscriptDisplay._get_tool_purpose(tool)}
                    for tool, count in transcript_summary["tool_usage_count"].items()
                ])
                st.dataframe(tools_df, use_container_width=True)
            
            # Detailed steps
            st.subheader("🧠 Step-by-Step Reasoning")
            for step in transcript_summary["steps"]:
                AgentTranscriptDisplay._render_step_card(step)
    
    @staticmethod
    def render_compact_transcript(transcript_summary: Dict[str, Any]):
        """
        Render a compact version of the transcript suitable for sidebar or small spaces.
        
        Args:
            transcript_summary: Formatted transcript summary
        """
        st.subheader("🤖 Agent Activity")
        
        # Quick stats
        st.write(f"**Tools Used:** {', '.join(transcript_summary['tools_used'])}")
        st.write(f"**Steps:** {transcript_summary['total_steps']}")
        
        # Step summary
        for i, step in enumerate(transcript_summary["steps"], 1):
            with st.container():
                st.write(f"**Step {i}:** {step['tool_name']}")
                if step["reasoning"]:
                    # Extract key reasoning (first sentence)
                    reasoning_preview = step["reasoning"].split('.')[0][:100] + "..."
                    st.caption(reasoning_preview)
    
    @staticmethod
    def _render_step_card(step: Dict[str, Any]):
        """
        Render an individual step in the agent's reasoning process.
        
        Args:
            step: Individual step information
        """
        with st.container():
            st.markdown(f"### Step {step['step_number']}: {step['tool_name']}")
            
            # Tool input
            if step["tool_input"]:
                st.markdown("**🔧 Tool Input:**")
                if isinstance(step["tool_input"], dict):
                    for key, value in step["tool_input"].items():
                        st.write(f"• **{key}:** {value}")
                else:
                    st.write(step["tool_input"])
            
            # Agent reasoning
            if step["reasoning"]:
                st.markdown("**🧠 Agent Reasoning:**")
                st.info(step["reasoning"])
            
            # Tool output
            if step["observation"]:
                st.markdown("**📊 Tool Output:**")
                st.code(step["observation"], language="text")
            
            st.divider()
    
    @staticmethod
    def _get_tool_purpose(tool_name: str) -> str:
        """
        Get human-readable purpose for each tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Human-readable tool purpose
        """
        tool_purposes = {
            "dataset_preview": "Data exploration",
            "run_pandas_query": "Custom data analysis",
            "groupby_aggregate": "Data grouping & aggregation",
            "top_categories": "Find frequent values",
            "histogram": "Distribution analysis",
            "correlation_matrix": "Correlation analysis",
            "create_histogram_chart": "Data visualization",
            "create_bar_chart": "Categorical visualization",
            "create_scatter_plot": "Relationship visualization",
            "create_line_chart": "Trend visualization",
            "time_series_analysis": "Time series analysis",
            "create_correlation_heatmap": "Correlation visualization",
            "create_column": "Data transformation",
            "analyze_missing_data": "Data quality analysis",
            "detect_outliers": "Outlier detection",
            "find_duplicates": "Duplicate analysis",
            "apply_basic_cleaning": "Data cleaning",
            "suggest_data_cleaning": "Cleaning recommendations",
            "run_linear_regression": "Predictive modeling",
            "run_logistic_regression": "Classification modeling",
            "run_arima_analysis": "Time series forecasting",
            "create_arima_plot": "ARIMA model visualization",
            "create_arima_forecast_plot": "ARIMA forecast visualization",
            "mean_target_encoding": "Feature engineering",
            "evaluate_regression_model": "Model evaluation",
            "evaluate_classification_model": "Model evaluation",
        }
        return tool_purposes.get(tool_name, "Analysis tool")


class AgentTranscriptExporter:
    """
    Export agent transcripts in various formats for analysis and documentation.
    """
    
    @staticmethod
    def export_to_json(transcript_summary: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export transcript to JSON format.
        
        Args:
            transcript_summary: Formatted transcript summary
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            JSON string of the transcript
        """
        if filename is None:
            filename = f"agent_transcript_{transcript_summary['session_id']}.json"
        
        return json.dumps(transcript_summary, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_to_markdown(transcript_summary: Dict[str, Any]) -> str:
        """
        Export transcript to Markdown format for documentation.
        
        Args:
            transcript_summary: Formatted transcript summary
            
        Returns:
            Markdown string of the transcript
        """
        md_lines = [
            f"# AI Agent Transcript - {transcript_summary['session_id']}",
            f"",
            f"**Timestamp:** {transcript_summary['timestamp']}",
            f"**Total Steps:** {transcript_summary['total_steps']}",
            f"**Tools Used:** {', '.join(transcript_summary['tools_used'])}",
            f"",
            f"## Tool Usage Summary",
            f""
        ]
        
        for tool, count in transcript_summary["tool_usage_count"].items():
            md_lines.append(f"- **{tool}:** {count} times")
        
        md_lines.extend([
            f"",
            f"## Step-by-Step Reasoning",
            f""
        ])
        
        for step in transcript_summary["steps"]:
            md_lines.extend([
                f"### Step {step['step_number']}: {step['tool_name']}",
                f"",
                f"**Tool Input:** {step['tool_input']}",
                f"",
                f"**Agent Reasoning:**",
                f"{step['reasoning']}",
                f"",
                f"**Tool Output:**",
                f"```",
                f"{step['observation']}",
                f"```",
                f"",
                f"---",
                f""
            ])
        
        md_lines.extend([
            f"## Final Output",
            f"",
            f"{transcript_summary['final_output']}",
        ])
        
        return "\n".join(md_lines)
    
    @staticmethod
    def create_downloadable_transcript(transcript_summary: Dict[str, Any], format_type: str = "json", unique_id: str = None):
        """
        Create a downloadable transcript file in Streamlit.
        
        Args:
            transcript_summary: Formatted transcript summary
            format_type: Export format ('json' or 'markdown')
            unique_id: Optional unique identifier for button key
        """
        session_id = transcript_summary['session_id']
        unique_suffix = f"_{unique_id}" if unique_id else ""
        
        if format_type == "json":
            content = AgentTranscriptExporter.export_to_json(transcript_summary)
            filename = f"agent_transcript_{session_id}.json"
            mime_type = "application/json"
        elif format_type == "markdown":
            content = AgentTranscriptExporter.export_to_markdown(transcript_summary)
            filename = f"agent_transcript_{session_id}.md"
            mime_type = "text/markdown"
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        st.download_button(
            label=f"📥 Download Transcript ({format_type.upper()})",
            data=content,
            file_name=filename,
            mime=mime_type,
            key=f"download_transcript_{session_id}_{format_type}{unique_suffix}"
        )


def store_session_transcripts(transcript_summary: Dict[str, Any]):
    """
    Store transcript in session state for access across the app.
    
    Args:
        transcript_summary: Formatted transcript summary
    """
    if "agent_transcripts" not in st.session_state:
        st.session_state.agent_transcripts = []
    
    st.session_state.agent_transcripts.append(transcript_summary)
    
    # Keep only last 10 transcripts to avoid memory issues
    if len(st.session_state.agent_transcripts) > 10:
        st.session_state.agent_transcripts = st.session_state.agent_transcripts[-10:]


def render_transcript_history():
    """
    Render a history of all agent transcripts from the current session.
    """
    if "agent_transcripts" not in st.session_state or not st.session_state.agent_transcripts:
        st.info("No agent transcripts available yet. Run some analysis to see agent reasoning!")
        return
    
    st.subheader("📜 Agent Transcript History")
    
    for i, transcript in enumerate(reversed(st.session_state.agent_transcripts)):
        transcript_num = len(st.session_state.agent_transcripts) - i
        with st.expander(f"Transcript {transcript_num}: {transcript['session_id']}"):
            AgentTranscriptDisplay.render_transcript_expander(transcript, title="Agent Details")
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                AgentTranscriptExporter.create_downloadable_transcript(transcript, "json", f"hist_{transcript_num}")
            with col2:
                AgentTranscriptExporter.create_downloadable_transcript(transcript, "markdown", f"hist_{transcript_num}")


def get_latest_transcript_summary() -> Optional[Dict[str, Any]]:
    """
    Get the latest agent transcript summary for inclusion in reports.
    
    Returns:
        Latest transcript summary dict or None if no transcripts available
    """
    try:
        if "agent_transcripts" not in st.session_state or not st.session_state.agent_transcripts:
            return None
        
        # Get the most recent transcript
        latest_transcript = st.session_state.agent_transcripts[-1]
        
        # Return a simplified version suitable for reports
        return {
            'session_id': latest_transcript.get('session_id', ''),
            'session_info': latest_transcript.get('session_info', {}),
            'approach_summary': latest_transcript.get('approach_summary', ''),
            'tools_used': latest_transcript.get('tools_used', []),
            'tools_summary': latest_transcript.get('tools_summary', ''),
            'insights': latest_transcript.get('insights', ''),
            'summary': latest_transcript.get('summary', {})
        }
    except Exception as e:
        # Return None if there's any error
        return None