"""
Configuration constants for Stats Compass application.

This module centralizes all application-wide constants to make them
easily discoverable and modifiable in one place.
"""

# ========== Application Settings ==========
PAGE_TITLE = "Stats Compass"
PAGE_LAYOUT = "wide"

# ========== AI/LLM Settings ==========
DEFAULT_MODEL = "gpt-4o"

# ========== Token & Cost Limits ==========
# These match the values in utils/token_tracking.py
TOKEN_WARNING_THRESHOLD = 50000  # 50K tokens
COST_WARNING_THRESHOLD = 5.0     # $5.00

# ========== Display Settings ==========
MAX_CHART_DISPLAY = 10  # Maximum number of charts to display at once
MAX_OBSERVATION_LENGTH = 1000  # Characters before truncating tool output
RECENT_USAGE_DISPLAY_COUNT = 3  # Number of recent usage items to show in sidebar

# ========== File Upload Settings ==========
ALLOWED_FILE_TYPES = ["csv", "xlsx", "xls"]
MAX_FILE_SIZE_MB = 200  # Maximum file size in MB (Streamlit default)

# ========== Session State Keys ==========
# Centralized session state key names for consistency
SESSION_KEY_DF = "df"
SESSION_KEY_CHAT_HISTORY = "chat_history"
SESSION_KEY_CHART_DATA = "chart_data"
SESSION_KEY_CURRENT_RESPONSE_CHARTS = "current_response_charts"
SESSION_KEY_USAGE_HISTORY = "usage_history"
SESSION_KEY_AGENT_TRANSCRIPTS = "agent_transcripts"
SESSION_KEY_TO_PROCESS = "to_process"
SESSION_KEY_UPLOADED_FILENAME = "uploaded_filename"
SESSION_KEY_QUERY_COUNT = "query_count"
