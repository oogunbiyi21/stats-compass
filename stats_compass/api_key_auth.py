# stats_compass/api_key_auth.py
import streamlit as st
import re


def validate_api_key_format(api_key: str) -> bool:
    """Basic format validation for OpenAI API keys."""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # OpenAI API keys typically start with 'sk-' and have specific length
    # We'll be lenient and just check for non-empty string that looks roughly right
    api_key = api_key.strip()
    
    if len(api_key) < 10:  # Too short
        return False
    
    # Basic pattern check (optional - we can accept any non-empty string)
    if api_key.startswith('sk-'):
        return len(api_key) > 20  # Reasonable minimum length
    
    # Accept any reasonable length string (for flexibility)
    return len(api_key) > 10


def check_api_key():
    """Returns `True` if the user has provided an API key."""
    
    # Check if user already has API key set
    if st.session_state.get("api_key_set", False):
        return True
    
    def api_key_entered():
        """Handles API key submission."""
        api_key = st.session_state.get("api_key_input", "").strip()
        
        if validate_api_key_format(api_key):
            st.session_state["openai_api_key"] = api_key
            st.session_state["api_key_set"] = True
            # Clear the input field
            if "api_key_input" in st.session_state:
                del st.session_state["api_key_input"]
            # Trigger rerun to immediately proceed to main app
            st.rerun()
        else:
            st.session_state["api_key_set"] = False
            st.error("Please enter a valid OpenAI API key (starts with 'sk-')")

    # Show API key input page
    render_api_key_input_page(api_key_entered)
    return False


def render_api_key_input_page(callback_func):
    """Renders the API key input page."""
    
    st.markdown("# 🧭 Stats Compass - Public Beta")
    st.markdown("### 🔑 Enter Your OpenAI API Key to Get Started")
    
    # Explanation section
    with st.expander("ℹ️ Why do I need an API key?", expanded=True):
        st.markdown("""
        **Stats Compass uses OpenAI's GPT-4 for intelligent data analysis:**
        
        ✅ **Privacy**: Your API key stays in your browser - we never store it  
        ✅ **Control**: You see exactly what you're paying for  
        ✅ **Cost**: Typically $0.01-$0.50 per analysis session  
        ✅ **Quality**: Direct access to the latest GPT-4 models  
        
        **Don't have an API key?** Get one in 2 minutes:
        1. Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        2. Create account and add payment method ($5 minimum)
        3. Generate new API key
        4. Copy and paste it below
        """)
    
    # API key input
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            key="api_key_input",
            help="Your API key starts with 'sk-' and is about 50+ characters long"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Alignment spacer
        if st.button("Continue", type="primary"):
            callback_func()
    
    # Show error if validation failed
    if st.session_state.get("api_key_error"):
        st.error(st.session_state["api_key_error"])
        # Clear error after showing
        del st.session_state["api_key_error"]
    
    # Help links and detailed guide
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("🔗 [Get API Key](https://platform.openai.com/api-keys)")
    
    with col2:
        st.markdown("💰 [Check Pricing](https://openai.com/pricing)")
    
    with col3:
        if st.button("📚 Detailed Guide"):
            st.session_state["show_api_help"] = True
    
    with col4:
        st.markdown("� [Privacy Policy](https://openai.com/privacy/)")
    
    # Show detailed help if requested
    if st.session_state.get("show_api_help", False):
        with st.expander("📚 Complete Setup Guide", expanded=True):
            show_api_key_help()
            if st.button("Close Guide"):
                st.session_state["show_api_help"] = False


def render_sidebar_api_key_widget():
    """Renders API key management widget in sidebar."""
    
    if not st.session_state.get("api_key_set", False):
        return
    
    # st.markdown("---")
    st.markdown("### 🔑 API Key Settings")
    
    # Show masked current key
    current_key = st.session_state.get("openai_api_key", "")
    if current_key:
        if len(current_key) > 12:
            masked_key = current_key[:8] + "****" + current_key[-4:]
        else:
            masked_key = "****"
        st.caption(f"Current: `{masked_key}`")
    
    # Update key functionality
    with st.expander("Update API Key"):
        new_key = st.text_input(
            "New API Key",
            type="password",
            placeholder="sk-proj-...",
            key="sidebar_api_key_update_input"  # Fixed duplicate key
        )
        
        if st.button("Update Key", key="update_api_key_btn"):
            if new_key and validate_api_key_format(new_key):
                st.session_state["openai_api_key"] = new_key
                st.session_state["api_key_set"] = True
                st.success("✅ API Key updated!")
            else:
                st.error("❌ Please enter a valid OpenAI API key (starts with 'sk-')")
    
    # Help links
    st.markdown("📚 [Get API Key](https://platform.openai.com/api-keys)")


def get_user_api_key():
    """Returns the user's API key from session state."""
    return st.session_state.get("openai_api_key", "")





def clear_api_key():
    """Clears the stored API key (for logout functionality)."""
    keys_to_clear = ["openai_api_key", "api_key_set"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def show_api_key_help():
    """Shows helpful information about getting and using OpenAI API keys."""
    st.markdown("""
    ## 🔑 How to Get Your OpenAI API Key

    ### Step 1: Create OpenAI Account
    1. Visit [platform.openai.com](https://platform.openai.com)
    2. Sign up or log in to your account
    3. Complete email verification

    ### Step 2: Add Payment Method
    1. Go to [Billing](https://platform.openai.com/account/billing)
    2. Add a credit/debit card
    3. Add at least $5 credit (minimum)

    ### Step 3: Generate API Key
    1. Navigate to [API Keys](https://platform.openai.com/api-keys)
    2. Click "Create new secret key"
    3. Give it a name (e.g., "Stats Compass")
    4. Copy the key immediately (starts with `sk-`)

    ### Step 4: Use in Stats Compass
    - Paste your key in the input field above
    - Key is stored securely in your browser session
    - We never see or store your API key

    ## 💰 Pricing Information

    **Typical Stats Compass costs:**
    - Simple question: ~$0.01-0.03
    - Complex analysis: ~$0.05-0.15
    - Full session (5-10 questions): ~$0.20-0.50

    **Why so affordable?**
    - Direct API access (no markup)
    - Efficient prompting reduces token usage
    - You only pay for what you use

    ## 🔒 Privacy & Security

    ✅ **Your API key never leaves your browser**  
    ✅ **We don't store or log API keys**  
    ✅ **Direct connection to OpenAI**  
    ✅ **No third-party key handling**  

    ---
    *Having trouble? Contact support or check OpenAI's documentation.*
    """)


def validate_and_suggest_fixes(api_key: str) -> tuple[bool, str]:
    """
    Validates API key and provides specific suggestions if invalid.
    Returns (is_valid, error_message)
    """
    if not api_key or not isinstance(api_key, str):
        return False, "API key cannot be empty"
    
    api_key = api_key.strip()
    
    if len(api_key) < 10:
        return False, "API key is too short (should be 50+ characters)"
    
    if not api_key.startswith('sk-'):
        return False, "OpenAI API keys should start with 'sk-'"
    
    if len(api_key) < 40:
        return False, "API key appears incomplete (should be 50+ characters)"
    
    # Check for common copy-paste issues
    if '...' in api_key or '****' in api_key:
        return False, "Please copy the complete API key (not the masked version)"
    
    if ' ' in api_key:
        return False, "API key contains spaces - please check for copy-paste errors"
    
    return True, ""