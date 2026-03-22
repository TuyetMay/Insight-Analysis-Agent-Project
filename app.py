"""
app.py
Streamlit entry point — UI orchestration only.
No business logic here: all data, AI, and chart concerns live in their own packages.
"""

import streamlit as st

from charts.breakdown import create_profit_by_region, create_profit_by_segment, create_profit_heatmap
from charts.products import create_category_distribution, create_discount_impact, create_top_subcategories
from charts.trends import create_orders_by_month, create_sales_profit_trend
from config import Config
from core.data_loader import load_filtered_data, get_filter_options, calculate_kpis, load_filtered_data_safe
from chatbot import DashboardChatbot
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
    st.error("❌ Cannot connect to the database. Check your .env configuration.")
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
chatbot = DashboardChatbot(df, kpis, filters)
_QUICK_TOKEN_LABELS = {
    "__quick_sales__":   "📊 Sales overview — trends, top products & regions",
    "__quick_profit__":  "💹 Profit overview — trends, margin & top regions",
    "__quick_orders__":  "📦 Orders overview — volume trends & AOV",
    "__quick_margin__":  "📈 Profit margin — trends & category breakdown",
}

if st.session_state.get("pending_question"):
    q = st.session_state.pop("pending_question")
    history = st.session_state.setdefault("chat_history", [])
    display_q = _QUICK_TOKEN_LABELS.get(q, q)
    history.append({"role": "user", "content": q})
    response = chatbot.get_response(q)
    suggs = chatbot.get_suggestions()
    if suggs:
        response += "\n\n**Suggested follow-ups:**\n" + "\n".join(f"- {s['text']}" for s in suggs)
    history.append({"role": "assistant", "content": response})
    
if "pending_question" in st.session_state and st.session_state["pending_question"]:
    q = st.session_state.pop("pending_question")
    history = st.session_state.setdefault("chat_history", [])
    history.append({"role": "user", "content": q})
    response = chatbot.get_response(q)
    suggs = chatbot.get_suggestions()
    if suggs:
        response += "\n\n**Suggested follow-ups:**\n" + "\n".join(
            f"- {s['text']}" for s in suggs
        )
    history.append({"role": "assistant", "content": response})
    st.session_state.chat_history = history
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
            st.session_state["pending_question"] = question
            st.rerun()

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
