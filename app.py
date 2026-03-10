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

# ─────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────

st.markdown(f"<div class='main-header'>{Config.APP_ICON} {Config.APP_TITLE}</div>",
            unsafe_allow_html=True)

# KPIs
st.markdown("## 📌 Key Performance Indicators")
kpi_cols = st.columns(4)
kpi_cols[0].metric("Total Sales",    f"${kpis['total_sales']:,.0f}")
kpi_cols[1].metric("Total Profit",   f"${kpis['total_profit']:,.0f}")
kpi_cols[2].metric("Total Orders",   f"{kpis['total_orders']:,}")
kpi_cols[3].metric("Profit Margin",  f"{kpis['profit_margin']:.2f}%")

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
