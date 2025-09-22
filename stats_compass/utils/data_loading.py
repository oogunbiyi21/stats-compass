# stats_compass/utils/data_loading.py
"""
Data loading utilities for DS Auto Insights.
Handles file upload, parsing, and validation.
"""

import io
import pandas as pd
import streamlit as st
from typing import Optional


def load_table(uploaded_file):
    """
    Load a CSV or Excel file into a pandas DataFrame.
    Handles multiple Excel sheets and various CSV encodings.
    """
    name = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else ""
    is_excel = name.endswith(".xlsx") or name.endswith(".xls")

    if is_excel:
        try:
            # Check if Excel file has multiple sheets
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Load the first sheet
            df = pd.read_excel(uploaded_file, sheet_name=0)
            
            # Show warning if multiple sheets detected
            if len(sheet_names) > 1:
                first_sheet = sheet_names[0]
                other_sheets = sheet_names[1:]
                other_sheets_str = ", ".join([f"'{sheet}'" for sheet in other_sheets])
                
                warning_message = (
                    f"⚠️ **Multiple sheets detected in Excel file!**\n\n"
                    f"📊 **Currently analyzing:** '{first_sheet}' (first sheet)\n\n"
                    f"📋 **Other sheets found:** {other_sheets_str}\n\n"
                    f"💡 **Tip:** To analyze other sheets, save them as separate Excel files or CSV files and upload individually."
                )
                st.warning(warning_message)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

    # Try different CSV parsing options with multiple encodings
    encoding_options = [
        {},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin1"},
        {"encoding": "iso-8859-1"},
        {"encoding": "cp1252"},
        {"sep": ";"},
        {"on_bad_lines": "skip"}
    ]
    
    for opts in encoding_options:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, **opts)
        except Exception:
            continue

    try:
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        for opts in encoding_options:
            try:
                df = pd.read_csv(io.BytesIO(raw), **opts)
                # Check if DataFrame is empty or has no columns
                if df.empty or df.shape[1] == 0:
                    raise ValueError("The uploaded file appears to be empty or contains no data columns.")
                return df
            except ValueError as ve:
                # Re-raise ValueError to maintain the empty file message
                if "empty" in str(ve).lower():
                    raise ve
            except Exception:
                continue
    except Exception:
        pass

    raise ValueError("Unable to parse file as CSV or Excel. Try re-exporting or cleaning the file.")


def process_uploaded_file(uploaded_file, clear_history=False):
    """
    Process an uploaded file and update session state.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        clear_history: Whether to clear chat history (for new file uploads)
    
    Returns:
        bool: True if file was processed successfully, False otherwise
    """
    if uploaded_file is None:
        return False
        
    try:
        df = load_table(uploaded_file)
        
        # Check if the loaded DataFrame is empty or has no meaningful data
        if df.empty or df.shape[1] == 0:
            st.warning("⚠️ **Empty file detected!**\n\nThe uploaded file appears to be empty or contains no data columns. Please upload a file with actual data to proceed.")
            st.info("💡 **Tips for a successful upload:**\n- Ensure your file has at least one column with data\n- Check that your Excel sheet isn't blank\n- Verify your CSV has headers and data rows")
            st.session_state.df = None
            return False
            
        elif df.shape[0] == 0:
            st.warning("⚠️ **No data rows found!**\n\nYour file has column headers but no data rows. Please upload a file with actual data to analyze.")
            st.info("💡 Make sure your file contains data rows below the headers.")
            st.session_state.df = None
            return False
            
        else:
            # Success - store in session state
            st.session_state.df = df
            st.session_state.uploaded_filename = uploaded_file.name
            
            if clear_history:
                st.session_state.chat_history = []
                st.session_state.chart_data = []
            
            # Success message
            st.success(f"✅ **File uploaded successfully!**\n\n📊 **Dataset:** {uploaded_file.name}\n📈 **Shape:** {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            
            return True
            
    except ValueError as ve:
        # Handle specific file parsing errors
        st.error(f"❌ **File parsing error:** {str(ve)}")
        st.info("💡 **Troubleshooting tips:**\n- Check if your file is corrupted\n- Try saving as a different format (CSV/XLSX)\n- Ensure the file contains actual data")
        st.session_state.df = None
        return False
        
    except Exception as e:
        # Handle other unexpected errors
        st.error(f"❌ **Unexpected error loading file:** {type(e).__name__}")
        st.info("💡 Please check your file format and try again. Supported formats: CSV, XLSX, XLS")
        with st.expander("🔍 Technical details (for debugging)"):
            st.code(str(e))
        st.session_state.df = None
        return False
