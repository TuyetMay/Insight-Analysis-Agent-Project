"""
ui/styles.py
All custom CSS injected into the Streamlit app via st.markdown.
"""

_CSS = """
<style>
/* ── Layout: sidebar phải ─────────────────────────────────── */
div[data-testid="stAppViewContainer"] {
    display: flex !important;
    flex-direction: row !important;
}
section.main {
    order: 1 !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
}
[data-testid="stSidebarResizeHandle"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    order: 2 !important;
    width: 360px !important;
    min-width: 360px !important;
    max-width: 360px !important;
    height: 100vh !important;
    position: sticky !important;
    top: 0 !important;
    border-left: 1px solid #e2e8f0 !important;
    border-right: none !important;
    box-shadow: -4px 0 20px rgba(0,0,0,0.06) !important;
    background: #ffffff !important;
}

/* ── Header ───────────────────────────────────────────────── */
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1rem;
}

/* ── Chat header ──────────────────────────────────────────── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 12px 0;
    border-bottom: 1px solid #f0f0f0;
    margin-bottom: 12px;
}
.chat-header-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.chat-header-info { flex: 1; }
.chat-header-name {
    font-weight: 700;
    font-size: 14px;
    color: #1a1a2e;
    line-height: 1.2;
}
.chat-status {
    font-size: 11px;
    color: #22c55e;
    display: flex;
    align-items: center;
    gap: 4px;
}
.chat-status::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #22c55e;
    border-radius: 50%;
}

/* ── Message wrapper ──────────────────────────────────────── */
.msg-row {
    display: flex;
    gap: 8px;
    margin: 6px 0;
    align-items: flex-end;
}
.msg-row.user-row { flex-direction: row-reverse; }

/* ── Avatars ──────────────────────────────────────────────── */
.msg-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-bottom: 2px;
}
.bot-avatar {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.user-avatar {
    background: linear-gradient(135deg, #f093fb, #f5576c);
}

/* ── Bubbles ──────────────────────────────────────────────── */
.msg-bubble {
    max-width: 82%;
    padding: 10px 14px;
    border-radius: 18px;
    font-size: 13px;
    line-height: 1.55;
    word-wrap: break-word;
}
.user-bubble {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 18px 18px 4px 18px;
    box-shadow: 0 2px 8px rgba(102,126,234,0.3);
}
.bot-bubble {
    background: #f7f7f8;
    color: #1a1a2e;
    border-radius: 18px 18px 18px 4px;
    border: 1px solid #ebebeb;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Timestamp ────────────────────────────────────────────── */
.msg-time {
    font-size: 10px;
    color: #aaa;
    text-align: center;
    margin: 8px 0 4px;
}

/* ── Thinking animation ───────────────────────────────────── */
.thinking-row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
    margin: 6px 0;
}
.thinking-bubble {
    background: #f7f7f8;
    border: 1px solid #ebebeb;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.thinking-label {
    font-size: 12px;
    color: #888;
    font-style: italic;
    margin-right: 4px;
}
.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #667eea;
    animation: bounce 1.4s infinite ease-in-out;
}
.dot:nth-child(2) { animation-delay: 0.16s; background: #9b72cf; }
.dot:nth-child(3) { animation-delay: 0.32s; background: #764ba2; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
    30%           { transform: translateY(-6px); opacity: 1; }
}

/* ── Welcome card ─────────────────────────────────────────── */
.welcome-card {
    background: linear-gradient(135deg, #667eea15, #764ba215);
    border: 1px solid #667eea30;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    margin: 12px 0;
}
.welcome-card .welcome-emoji { font-size: 32px; margin-bottom: 8px; }
.welcome-card .welcome-title {
    font-weight: 700;
    font-size: 14px;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.welcome-card .welcome-sub {
    font-size: 12px;
    color: #666;
    line-height: 1.5;
}

/* ── Quick chips ──────────────────────────────────────────── */
.chip-label {
    font-size: 11px;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 10px 0 6px;
}

/* ── Input area ───────────────────────────────────────────── */
.input-area {
    border-top: 1px solid #f0f0f0;
    padding-top: 10px;
    margin-top: 8px;
}

/* ── Slide animations ─────────────────────────────────────── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
.msg-row, .thinking-row {
    animation: fadeSlideIn 0.25s ease;
}
</style>
"""

_JS = """
<script>
function moveSidebarRight() {
    const container = document.querySelector('div[data-testid="stAppViewContainer"]');
    const sidebar   = document.querySelector('section[data-testid="stSidebar"]');
    if (container && sidebar && sidebar.parentElement !== container) {
        container.appendChild(sidebar);
    }
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', moveSidebarRight);
} else {
    moveSidebarRight();
}
setTimeout(moveSidebarRight, 300);
setTimeout(moveSidebarRight, 1000);
</script>
"""


def inject() -> None:
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_JS,  unsafe_allow_html=True)