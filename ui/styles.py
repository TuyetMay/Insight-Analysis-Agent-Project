"""
ui/styles.py
All custom CSS injected into the Streamlit app via st.markdown.
Centralising styles here keeps app.py clean and makes theming easy to adjust.
"""

_CSS = """
<style>
/* ── Header ───────────────────────────────────────────────── */
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1rem;
}

/* ── Chat messages ────────────────────────────────────────── */
.user-msg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0 8px auto;
    max-width: 75%;
    width: fit-content;
    box-shadow: 0 2px 8px rgba(102,126,234,0.3);
    animation: slideInRight 0.3s ease;
}

.bot-msg {
    background: white;
    color: #2d3748;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 4px;
    margin: 8px auto 8px 0;
    max-width: 75%;
    width: fit-content;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    animation: slideInLeft 0.3s ease;
}

.welcome-msg {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    margin: 12px 0;
    box-shadow: 0 4px 12px rgba(245,87,108,0.2);
}

/* ── Chat container ───────────────────────────────────────── */
.chat-box {
    background: #f8f9fa;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #e2e8f0;
}

.chat-box::-webkit-scrollbar       { width: 6px; }
.chat-box::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 3px; }

/* ── Slide animations ─────────────────────────────────────── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px);  }
    to   { opacity: 1; transform: translateX(0);     }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0);     }
}
</style>
"""


def inject() -> None:
    """Call once at app startup to inject all custom CSS."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)
