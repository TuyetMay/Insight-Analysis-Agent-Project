import streamlit as st
from config import Config
from src.visualizations import (
    create_sales_profit_trend,
    create_orders_by_month,
    create_profit_by_region,
    create_profit_by_segment,
    create_profit_heatmap,
    create_discount_impact,
    create_top_subcategories,
    create_category_distribution
)
import pandas as pd
from src.data_loader import load_filtered_data, get_filter_options, calculate_kpis
from src.chatbot import DashboardChatbot
import time

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 8px 0 8px auto;
        max-width: 75%;
        width: fit-content;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
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
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .welcome-msg {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.2);
    }
    
    /* Chat container */
    .chat-box {
        background: #f8f9fa;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 16px;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
    }
    
    .chat-box::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-box::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
with st.spinner('⏳ Loading...'):
    filter_options = get_filter_options()

if not filter_options:
    st.error("❌ Cannot connect to Database.")
    st.stop()

# Sidebar Filters
st.sidebar.markdown("## 🔍 Filters")

min_date = pd.to_datetime(filter_options['min_date']).date()
max_date = pd.to_datetime(filter_options['max_date']).date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

selected_regions = st.sidebar.multiselect(
    "Region",
    options=filter_options['region'],
    default=filter_options['region']
)

selected_segments = st.sidebar.multiselect(
    "Segment",
    options=filter_options['segment'],
    default=filter_options['segment']
)

selected_categories = st.sidebar.multiselect(
    "Category",
    options=filter_options['category'],
    default=filter_options['category']
)
filters = {
    'date_range': date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_date, max_date),
    'region': selected_regions,
    'segment': selected_segments,
    'category': selected_categories
}

# Load filtered data
with st.spinner('🚀 Loading data...'):
    filtered_df = load_filtered_data(filters)

if filtered_df.empty:
    st.warning("⚠️ No data available.")
    st.stop()

kpis = calculate_kpis(filtered_df)
chatbot = DashboardChatbot(filtered_df, kpis, filters)

# Main Dashboard
st.markdown(f"<div class='main-header'>{Config.APP_ICON} {Config.APP_TITLE}</div>", unsafe_allow_html=True)

# KPIs
st.markdown("## 📌 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sales", f"${kpis['total_sales']:,.0f}")
with col2:
    st.metric("Total Profit", f"${kpis['total_profit']:,.0f}")
with col3:
    st.metric("Total Orders", f"{kpis['total_orders']:,}")
with col4:
    st.metric("Profit Margin", f"{kpis['profit_margin']:.2f}%")

# Charts
st.markdown("## 📈 Sales and Profit Over Time")
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(create_sales_profit_trend(filtered_df), width="stretch")
with col2:
    st.plotly_chart(create_orders_by_month(filtered_df), width="stretch")

st.markdown("## 🌍 Profit Breakdown")
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(create_profit_by_region(filtered_df), width="stretch")
with col2:
    st.plotly_chart(create_profit_by_segment(filtered_df), width="stretch")
with col3:
    st.plotly_chart(create_profit_heatmap(filtered_df), width="stretch")
st.markdown("## 💡 Product Insights")
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(create_discount_impact(filtered_df), width="stretch")
with col2:
    st.plotly_chart(create_top_subcategories(filtered_df), width="stretch")
with col3:
    st.plotly_chart(create_category_distribution(filtered_df), width="stretch")
with st.expander("📊 View Data"):
    st.dataframe(filtered_df.head(100), width="stretch")

st.markdown("---")
st.markdown("Dashboard by Streamlit • Plotly • AI Assistant")

# ======= CHAT SIDEBAR =======
st.sidebar.markdown("---")
st.sidebar.markdown("## 💬 AI Assistant")
# Chat Interface
st.sidebar.markdown("### 💭 Chat")

# Display messages
if st.session_state.chat_history:
    for msg in st.session_state.chat_history[-10:]:
        if msg["role"] == "user":
            st.sidebar.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="welcome-msg">👋 Ask me anything about your data!</div>', unsafe_allow_html=True)

# Quick action buttons
st.sidebar.markdown("**Quick Questions:**")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("💰 Total Sales", width="stretch"):
        question = "What is the total sales?"
        st.session_state.chat_history.append({"role": "user", "content": question})
        response = chatbot.get_response(question)
        suggs = chatbot.get_suggestions(language="vi")
        if suggs:
            response += "\n\n**Suggested next step**\n" + "\n".join([f"- {s['text']}" for s in suggs])
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.suggestions = chatbot.get_suggestions(language="vi")
        st.rerun()
    
    if st.button("📈 Trends", width="stretch"):
        question = "Show me sales trends"
        st.session_state.chat_history.append({"role": "user", "content": question})
        response = chatbot.get_response(question)
        suggs = chatbot.get_suggestions(language="vi")
        if suggs:
            response += "\n\n**Suggested next step**\n" + "\n".join([f"- {s['text']}" for s in suggs])
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.suggestions = chatbot.get_suggestions(language="vi")
        st.rerun()

with col2:
    if st.button("🌍 By Region", width="stretch"):
        question = "Show profit by region"
        st.session_state.chat_history.append({"role": "user", "content": question})
        response = chatbot.get_response(question)
        suggs = chatbot.get_suggestions(language="vi")
        if suggs:
            response += "\n\n** Suggested next step**\n" + "\n".join([f"- {s['text']}" for s in suggs])
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.suggestions = chatbot.get_suggestions(language="vi")
        st.rerun()
    
    if st.button("🏆 Top Items", width="stretch"):
        question = "What are the top categories?"
        st.session_state.chat_history.append({"role": "user", "content": question})
        response = chatbot.get_response(question)
        suggs = chatbot.get_suggestions(language="vi")
        if suggs:
            response += "\n\n**Suggested next step**\n" + "\n".join([f"- {s['text']}" for s in suggs])
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.suggestions = chatbot.get_suggestions(language="vi")
        st.rerun()

# Chat input
with st.sidebar.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question...", placeholder="E.g., What's the profit margin?", key="chat_input")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.form_submit_button("Send 📤", width="stretch")
    with col2:
        clear = st.form_submit_button("Clear", width="stretch")
    
    if submit and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        
        with st.spinner("Thinking..."):
            response = chatbot.get_response(user_input.strip())
            suggs = chatbot.get_suggestions(language="vi")
            if suggs:
                response += "\n\n**Suggested next step**\n" + "\n".join([f"- {s['text']}" for s in suggs])
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    
    if clear:
        st.session_state.chat_history = []
        st.session_state.insights = None
        st.session_state.suggestions = []

        st.rerun()
