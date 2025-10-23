"""
Tabs package for Stats Compass.

Contains individual tab modules for the main application interface.
"""

from .chat_tab import render_chat_tab
from .reports_tab import render_reports_tab
from .summary_tab import render_summary_tab
from .explore_tab import render_explore_tab
from .logs_tab import render_logs_tab

__all__ = [
    'render_chat_tab',
    'render_reports_tab',
    'render_summary_tab',
    'render_explore_tab',
    'render_logs_tab',
]
