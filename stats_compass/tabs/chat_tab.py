"""
Chat tab for Stats Compass application.

Handles the main chat interface including:
- Smart analysis suggestions
- Chat history rendering
- User query processing
- Real-time agent interaction
"""

import streamlit as st
from smart_suggestions import generate_smart_suggestions


def render_chat_tab(render_chat_message_func, process_user_query_func) -> None:
    """
    Render the chat tab with smart suggestions, chat history, and input.
    
    Args:
        render_chat_message_func: Function to render individual chat messages
        process_user_query_func: Function to process user queries
    """
    st.header("Chat")
    
    # Smart Suggestions Grid (2x3 format)
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        suggestions = generate_smart_suggestions(st.session_state.df)
        
        if suggestions:
            st.markdown("💡 **Quick Analysis Suggestions**")
            
            # Create 2x3 grid of suggestion buttons
            col1, col2, col3 = st.columns(3)
            
            # First row
            if len(suggestions) > 0:
                with col1:
                    if st.button(
                        suggestions[0]['title'], 
                        key="grid_suggest_0",
                        help=suggestions[0]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[0]['query']
                        st.rerun()
            
            if len(suggestions) > 1:
                with col2:
                    if st.button(
                        suggestions[1]['title'], 
                        key="grid_suggest_1",
                        help=suggestions[1]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[1]['query']
                        st.rerun()
            
            if len(suggestions) > 2:
                with col3:
                    if st.button(
                        suggestions[2]['title'], 
                        key="grid_suggest_2",
                        help=suggestions[2]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[2]['query']
                        st.rerun()
            
            # Additional suggestions in expander
            if len(suggestions) > 3:
                with st.expander(f"💡 More suggestions ({len(suggestions) - 3})", expanded=False):
                    for idx, suggestion in enumerate(suggestions[3:], start=3):
                        if st.button(
                            suggestion['title'], 
                            key=f"grid_suggest_{idx}",
                            help=suggestion['description'],
                            use_container_width=True
                        ):
                            st.session_state.to_process = suggestion['query']
                            st.rerun()
            
            st.divider()

    # 1) Replay existing chat history first (before processing new message)
    for i, msg in enumerate(st.session_state.chat_history):
        render_chat_message_func(msg, i, "history")

    # 2) If we have a message queued from the previous run, process it
    queued = st.session_state.pop("to_process", None)
    if queued is not None and hasattr(st.session_state, 'df') and st.session_state.df is not None:
        process_user_query_func(queued, st.session_state.df)

    # 3) Place the chat input at the END. When user submits, queue it + rerun.
    user_query = st.chat_input("Ask a question about your data", key="chat_input_bottom")
    if user_query:
        st.session_state.to_process = user_query
        st.rerun()
