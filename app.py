"""
app.py
Streamlit entry point — UI orchestration only.
No business logic here: all data, AI, and chart concerns live in their own packages.
"""

import hashlib
import json

import streamlit as st

from charts.breakdown import create_profit_by_region, create_profit_by_segment, create_profit_heatmap
from charts.products import create_category_distribution, create_discount_impact, create_top_subcategories
from charts.trends import create_orders_by_month, create_sales_profit_trend
from config import Config
from core.data_loader import load_filtered_data, get_filter_options, calculate_kpis, load_filtered_data_safe
from chatbot import DashboardChatbot
from core.database import reset_pool
from ui import inject_styles, render_filters, render_chat_sidebar
from ui.components import _KPI_QUICK_QUESTIONS

# ─────────────────────────────────────────────────────────────
# Page config & styles
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("suggestions", [])

# ─────────────────────────────────────────────────────────────
# Bootstrap: filter options
# ─────────────────────────────────────────────────────────────

with st.spinner("⏳ Connecting to database…"):
    filter_options = get_filter_options()

if not filter_options:
    st.error("Cannot connect to the database. Check your .env configuration.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🔄 Retry Connection", type="primary", use_container_width=True):
            st.cache_data.clear()
            reset_pool()
            st.rerun()
    with col_b:
        with st.expander("🔍 Debug info"):
            st.code(f"""DB_HOST     = {Config.DB_HOST}
DB_PORT     = {Config.DB_PORT}
DB_NAME     = {Config.DB_NAME}
DB_USER     = {Config.DB_USER}
DB_TABLE    = {Config.DB_TABLE}
DB_PASSWORD = {"(set)" if Config.DB_PASSWORD else "(EMPTY — check .env)"}
GOOGLE_KEY  = {"(set)" if Config.GOOGLE_API_KEY else "(EMPTY)"}
""")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Sidebar: filters
# ─────────────────────────────────────────────────────────────

filters = render_filters(filter_options)

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────

with st.spinner("🚀 Loading data…"):
    df = load_filtered_data_safe(filters)


if df.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

kpis    = calculate_kpis(df)
def _filters_hash(filters):
    f = {
        "dr": [str(filters["date_range"][0]), str(filters["date_range"][1])],
        "r": sorted(filters.get("region", []) or []),
        "s": sorted(filters.get("segment", []) or []),
        "c": sorted(filters.get("category", []) or []),
    }

    return hashlib.md5(
        json.dumps(f, sort_keys=True).encode()
    ).hexdigest()[:8]

current_hash = _filters_hash(filters)

if ("chatbot" not in st.session_state
        or st.session_state.get("_filter_hash") != current_hash):
    with st.spinner("🔧 Building AI knowledge base…"):
        st.session_state["chatbot"]      = DashboardChatbot(df, kpis, filters)
        st.session_state["_filter_hash"] = current_hash

chatbot = st.session_state["chatbot"]
_QUICK_TOKEN_LABELS = {
    "__quick_sales__":   "📊 Sales overview — trends, top products & regions",
    "__quick_profit__":  "💹 Profit overview — trends, margin & top regions",
    "__quick_orders__":  "📦 Orders overview — volume trends & AOV",
    "__quick_margin__":  "📈 Profit margin — trends & category breakdown",
}

if st.session_state.get("pending_question"):
    _q = st.session_state.pop("pending_question")
    _history = st.session_state.setdefault("chat_history", [])
    
    # Dùng label readable, không phải raw token
    _label = _QUICK_TOKEN_LABELS.get(_q, _q)
    _history.append({"role": "user", "content": _label})
    
    try:
        _response = chatbot.get_response(_q)
        _suggs    = chatbot.get_suggestions()
        if _suggs:
            _response += "\n\n**Suggested follow-ups:**\n" + "\n".join(
                f"- {s['text']}" for s in _suggs
            )
    except Exception as _e:
        _response = f"❌ Could not load insight. ({_e})"
    
    _history.append({"role": "assistant", "content": _response})
    st.session_state.chat_history = _history
    st.rerun() 
# ─────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────

st.markdown(f"<div class='main-header'>{Config.APP_ICON} {Config.APP_TITLE}</div>",
            unsafe_allow_html=True)

# KPIs
st.markdown("## 📌 Key Performance Indicators")
kpi_cols = st.columns(4)
_KPI_QUESTIONS = {
    0: ("Total Sales",   f"${kpis['total_sales']:,.0f}",  "__quick_sales__"),
    1: ("Total Profit",  f"${kpis['total_profit']:,.0f}", "__quick_profit__"),
    2: ("Total Orders",  f"{kpis['total_orders']:,}",     "__quick_orders__"),
    3: ("Profit Margin", f"{kpis['profit_margin']:.2f}%", "__quick_margin__"),
}
for i, (label, value, question) in _KPI_QUESTIONS.items():
    with kpi_cols[i]:
        st.metric(label, value)
        if st.button("🔍 Ask AI", key=f"ask_kpi_{i}", use_container_width=True):
            if not df.empty:
                st.session_state["pending_question"] = question
                st.rerun()
            else:
                st.warning("No data to analyze.")

# Time-series section
st.markdown("## 📈 Sales & Profit Over Time")
col1, col2 = st.columns([2, 1])
col1.plotly_chart(create_sales_profit_trend(df), use_container_width=True)
col2.plotly_chart(create_orders_by_month(df),    use_container_width=True)

# Breakdown section
st.markdown("## 🌍 Profit Breakdown")
col1, col2, col3 = st.columns(3)
col1.plotly_chart(create_profit_by_region(df),  use_container_width=True)
col2.plotly_chart(create_profit_by_segment(df), use_container_width=True)
col3.plotly_chart(create_profit_heatmap(df),    use_container_width=True)

# Product insights section
st.markdown("## 💡 Product Insights")
col1, col2, col3 = st.columns(3)
col1.plotly_chart(create_discount_impact(df),      use_container_width=True)
col2.plotly_chart(create_top_subcategories(df),    use_container_width=True)
col3.plotly_chart(create_category_distribution(df), use_container_width=True)

# Raw data expander
with st.expander("📊 View Raw Data (first 100 rows)"):
    st.dataframe(df.head(100), use_container_width=True)

st.markdown("---")
st.caption("Superstore BI Dashboard · Streamlit · Plotly · Gemini AI")

# ─────────────────────────────────────────────────────────────
# Sidebar: AI chat assistant
# ─────────────────────────────────────────────────────────────

render_chat_sidebar(chatbot)