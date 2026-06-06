"""Streamlit session state management."""

import streamlit as st

DEFAULTS = {
    "last_result": None,
    "comparison_results": None,
}


def init_state() -> None:
    """Initialize session state with default values."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value