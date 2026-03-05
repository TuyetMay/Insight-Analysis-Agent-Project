"""
ui/components.py
Reusable Streamlit UI components: sidebar filters and AI chat sidebar.
Returns pure data (filters dict, processed input) so app.py stays free of widget logic.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────
# Sidebar: data filters
# ─────────────────────────────────────────────────────────────

def render_filters(filter_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render the filter sidebar and return the current filter state as a dict.
    """
    st.sidebar.markdown("## 🔍 Filters")

    min_date = pd.to_datetime(filter_options["min_date"]).date()
    max_date = pd.to_datetime(filter_options["max_date"]).date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    selected_regions = st.sidebar.multiselect(
        "Region", options=filter_options["region"], default=filter_options["region"]
    )
    selected_segments = st.sidebar.multiselect(
        "Segment", options=filter_options["segment"], default=filter_options["segment"]
    )
    selected_categories = st.sidebar.multiselect(
        "Category", options=filter_options["category"], default=filter_options["category"]
    )

    safe_range = (
        date_range if isinstance(date_range, tuple) and len(date_range) == 2
        else (min_date, max_date)
    )

    return {
        "date_range": safe_range,
        "region":     selected_regions,
        "segment":    selected_segments,
        "category":   selected_categories,
    }


# ─────────────────────────────────────────────────────────────
# Sidebar: AI chat widget
# ─────────────────────────────────────────────────────────────

def render_chat_sidebar(chatbot: Any) -> None:
    """
    Render the AI assistant panel in the sidebar.
    All Streamlit state mutations happen here; app.py just calls this function.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💬 AI Assistant")
    st.sidebar.markdown("### 💭 Chat")

    # Message history
    history: List[Dict[str, str]] = st.session_state.get("chat_history", [])
    if history:
        for msg in history[-10:]:
            css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
            st.sidebar.markdown(
                f'<div class="{css_class}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.markdown(
            '<div class="welcome-msg">👋 Ask me anything about your data!</div>',
            unsafe_allow_html=True,
        )

    # Quick-action buttons
    st.sidebar.markdown("**Quick Questions:**")
    _quick_buttons(chatbot)

    # Free-text input
    _chat_form(chatbot)


# ── Internal helpers ──────────────────────────────────────────

def _quick_buttons(chatbot: Any) -> None:
    quick_qs = [
        ("💰 Total Sales",  "What is the total sales?"),
        ("🌍 By Region",    "Show profit by region"),
        ("📈 Trends",       "Show me sales trends"),
        ("🏆 Top Items",    "What are the top categories?"),
    ]
    cols = st.sidebar.columns(2)
    for i, (label, question) in enumerate(quick_qs):
        with cols[i % 2]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                _process_question(chatbot, question)
                st.rerun()


def _chat_form(chatbot: Any) -> None:
    with st.sidebar.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question…",
            placeholder="E.g., What's the profit margin?",
            key="chat_input",
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            submit = st.form_submit_button("Send 📤", use_container_width=True)
        with col2:
            clear = st.form_submit_button("Clear", use_container_width=True)

        if submit and user_input.strip():
            _process_question(chatbot, user_input.strip())
            st.rerun()

        if clear:
            st.session_state.chat_history  = []
            st.session_state.suggestions   = []
            st.rerun()


def _process_question(chatbot: Any, question: str) -> None:
    """Append user message, get bot response, store suggestions."""
    history = st.session_state.setdefault("chat_history", [])
    history.append({"role": "user", "content": question})

    response = chatbot.get_response(question)
    suggs    = chatbot.get_suggestions(language="vi")
    if suggs:
        response += "\n\n**Suggested next steps**\n" + "\n".join(
            f"- {s['text']}" for s in suggs
        )

    history.append({"role": "assistant", "content": response})
    st.session_state.suggestions = suggs
