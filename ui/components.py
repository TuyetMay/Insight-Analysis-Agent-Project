"""
ui/components.py
Reusable Streamlit UI components: sidebar filters and AI chat sidebar.
ChatGPT / Claude-style chat with thinking animation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────
# Sidebar: data filters
# ─────────────────────────────────────────────────────────────

def render_filters(filter_options: Dict[str, Any]) -> Dict[str, Any]:
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
# Markdown → HTML converter
# ─────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Convert Markdown subset to safe HTML for chat bubbles."""

    # 1. Escape HTML special chars
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 2. Fix escaped dollar signs  \$ -> $ then escape ALL $ to HTML entity
    # (Streamlit renders $ as LaTeX math delimiters, so we must escape them)
    text = text.replace("\\$", "$")
    text = text.replace("$", "&#36;")

    # 3. Horizontal rule ---
    text = re.sub(
        r'\n?-{3,}\n?',
        '<hr style="border:none;border-top:1px solid #e0e0e0;margin:8px 0">',
        text
    )

    # 4. Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # 5. Italic context line *Jan 2014 - Dec 2017*
    text = re.sub(
        r'(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)',
        r'<span style="color:#888;font-size:0.88em;font-style:italic">\1</span>',
        text
    )

    # 6. Numbered list  1. item
    def render_ol(m: re.Match) -> str:
        items = re.findall(r'^\d+\.\s+(.+)', m.group(0), re.MULTILINE)
        lis = "".join(
            f'<li style="margin:4px 0;padding-left:2px">{i}</li>' for i in items
        )
        return f'<ol style="margin:6px 0 6px 4px;padding-left:16px">{lis}</ol>'
    text = re.sub(r'(?:^\d+\..+$\n?)+', render_ol, text, flags=re.MULTILINE)

    # 7. Bullet list  - item  or  * item
    def render_ul(m: re.Match) -> str:
        items = re.findall(r'^[*\-]\s+(.+)', m.group(0), re.MULTILINE)
        lis = "".join(
            f'<li style="margin:4px 0;padding-left:2px">{i}</li>' for i in items
        )
        return f'<ul style="margin:6px 0 6px 4px;padding-left:16px;list-style:disc">{lis}</ul>'
    text = re.sub(r'(?:^[*\-]\s+.+$\n?)+', render_ul, text, flags=re.MULTILINE)

    # 8. Newlines -> <br>
    text = re.sub(r'\n', '<br>', text)

    # 9. Clean up <br> around block tags
    text = re.sub(r'(<br>)+(<(?:ul|ol|hr))', r'\2', text)
    text = re.sub(r'(</(?:ul|ol)>)(<br>)+', r'\1', text)

    # 10. Style the insight block nicely
    text = re.sub(
        r'(💡\s*&lt;strong&gt;Insight:&lt;/strong&gt;|💡\s*<strong>Insight:</strong>)',
        r'<span style="color:#7c3aed;font-weight:600">💡 Insight:</span>',
        text
    )

    return text


# ─────────────────────────────────────────────────────────────
# HTML bubble builders
# ─────────────────────────────────────────────────────────────

def _user_bubble(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<div class="msg-row user-row">
  <div class="msg-avatar user-avatar">🧑</div>
  <div class="msg-bubble user-bubble">{escaped}</div>
</div>"""


def _bot_bubble(text: str) -> str:
    html_text = _md_to_html(text)
    return f"""
<div class="msg-row">
  <div class="msg-avatar bot-avatar">🤖</div>
  <div class="msg-bubble bot-bubble">{html_text}</div>
</div>"""


def _thinking_bubble() -> str:
    return """
<div class="thinking-row">
  <div class="msg-avatar bot-avatar">🤖</div>
  <div class="thinking-bubble">
    <span class="thinking-label">Thinking</span>
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
  </div>
</div>"""


def _welcome_card() -> str:
    return """
<div class="welcome-card">
  <div class="welcome-emoji">🤖</div>
  <div class="welcome-title">Superstore AI Assistant</div>
  <div class="welcome-sub">
    Ask me about Sales, Profit, Orders, or any insight from your data.
  </div>
</div>"""


def _chat_header() -> str:
    return """
<div class="chat-header">
  <div class="chat-header-avatar">🤖</div>
  <div class="chat-header-info">
    <div class="chat-header-name">Superstore AI</div>
    <div class="chat-status">Online</div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────
# Main chat sidebar
# ─────────────────────────────────────────────────────────────

def render_chat_sidebar(chatbot: Any) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💬 AI Assistant")

    st.sidebar.markdown(_chat_header(), unsafe_allow_html=True)

    history: List[Dict[str, str]] = st.session_state.get("chat_history", [])

    if not history:
        st.sidebar.markdown(_welcome_card(), unsafe_allow_html=True)
    else:
        for msg in history[-12:]:
            if msg["role"] == "user":
                st.sidebar.markdown(_user_bubble(msg["content"]), unsafe_allow_html=True)
            else:
                st.sidebar.markdown(_bot_bubble(msg["content"]), unsafe_allow_html=True)

    thinking_placeholder = st.sidebar.empty()

    st.sidebar.markdown('<div class="chip-label">⚡ Quick questions</div>', unsafe_allow_html=True)
    _quick_buttons(chatbot, thinking_placeholder)

    st.sidebar.markdown('<div class="input-area"></div>', unsafe_allow_html=True)
    _chat_form(chatbot, thinking_placeholder)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _quick_buttons(chatbot: Any, thinking_placeholder: Any) -> None:
    quick_qs = [
        ("💰 Total Sales",  "What is the total sales?"),
        ("🌍 By Region",    "Show profit by region"),
        ("📈 Trends",       "Monthly sales trend"),
        ("🏆 Top Products", "Top 5 sub-categories by profit"),
    ]
    cols = st.sidebar.columns(2)
    for i, (label, question) in enumerate(quick_qs):
        with cols[i % 2]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                _process_question(chatbot, question, thinking_placeholder)
                st.rerun()


def _chat_form(chatbot: Any, thinking_placeholder: Any) -> None:
    with st.sidebar.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question…",
            placeholder="E.g. Profit margin by region?",
            key="chat_input",
            label_visibility="collapsed",
        )
        col1, col2 = st.columns([4, 1])
        with col1:
            submit = st.form_submit_button("Send ➤", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🗑️", use_container_width=True)

        if submit and user_input.strip():
            _process_question(chatbot, user_input.strip(), thinking_placeholder)
            st.rerun()

        if clear:
            st.session_state.chat_history = []
            st.session_state.suggestions  = []
            st.rerun()


def _process_question(chatbot: Any, question: str, thinking_placeholder: Any) -> None:
    history = st.session_state.setdefault("chat_history", [])
    history.append({"role": "user", "content": question})

    # Show thinking animation
    thinking_placeholder.markdown(_thinking_bubble(), unsafe_allow_html=True)

    # Get response
    response = chatbot.get_response(question)
    suggs    = chatbot.get_suggestions()
    if suggs:
        response += "\n\n**Suggested follow-ups:**\n" + "\n".join(
            f"- {s['text']}" for s in suggs
        )

    # Clear animation, store answer
    thinking_placeholder.empty()
    history.append({"role": "assistant", "content": response})
    st.session_state.suggestions = suggs