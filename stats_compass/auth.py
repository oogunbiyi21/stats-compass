import streamlit as st
import hashlib

def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Skip authentication in local development
    if "localhost" in st.context.headers.get("host", "") or not hasattr(st, "secrets"):
        return True
    
    # Only require password in production (Streamlit Cloud)
    if "app_password" not in st.secrets:
        return True  # No password configured, allow access
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact the app owner for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
