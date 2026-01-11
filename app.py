import streamlit as st
from config import Config
from src.data_loader import load_superstore_data, filter_data, calculate_kpis
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
from src.chatbot import DashboardChatbot

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

# Custom CSS for floating chat
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Floating Chat Button */
    .chat-toggle-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .chat-toggle-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    
    /* Chat Panel */
    .chat-panel {
        position: fixed;
        right: -400px;
        top: 0;
        width: 400px;
        height: 100vh;
        background: white;
        box-shadow: -2px 0 10px rgba(0,0,0,0.1);
        transition: right 0.3s ease;
        z-index: 999;
        display: flex;
        flex-direction: column;
    }
    
    .chat-panel.open {
        right: 0;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        font-size: 18px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-close-btn {
        background: none;
        border: none;
        color: white;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        background: #f8f9fa;
    }
    
    .chat-input-area {
        padding: 15px;
        border-top: 1px solid #e0e0e0;
        background: white;
    }
    
    .message {
        margin-bottom: 15px;
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 85%;
        word-wrap: break-word;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .bot-message {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 4px;
    }
    
    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 15px;
    }
    
    .quick-action-btn {
        background: white;
        border: 1px solid #667eea;
        color: #667eea;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quick-action-btn:hover {
        background: #667eea;
        color: white;
    }
    
    .insights-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-size: 13px;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .chat-panel {
            width: 100%;
            right: -100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load data
with st.spinner('Loading data from database...'):
    df = load_superstore_data()

if df.empty:
    st.error("No data available. Please check your database connection.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("## 🔍 Filters")

# Date range filter
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()

date_range = st.sidebar.date_input(
    "Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Region filter
all_regions = sorted(df['region'].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Region",
    options=all_regions,
    default=all_regions
)

# Segment filter
all_segments = sorted(df['segment'].unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Segment",
    options=all_segments,
    default=all_segments
)

# Category filter
all_categories = sorted(df['category'].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Category",
    options=all_categories,
    default=all_categories
)

# Apply filters
filters = {
    'date_range': date_range if len(date_range) == 2 else (min_date, max_date),
    'region': selected_regions,
    'segment': selected_segments,
    'category': selected_categories
}

filtered_df = filter_data(df, filters)

# Calculate KPIs
kpis = calculate_kpis(filtered_df)

# Initialize chatbot with current context
chatbot = DashboardChatbot(filtered_df, kpis, filters)

# Main title
st.markdown(f"<div class='main-header'>{Config.APP_ICON} {Config.APP_TITLE}</div>", unsafe_allow_html=True)

# Display KPIs
st.markdown("## 📌 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Sales",
        value=f"${kpis['total_sales']:,.0f}",
        delta=None
    )

with col2:
    st.metric(
        label="Total Profit",
        value=f"${kpis['total_profit']:,.0f}",
        delta=None
    )

with col3:
    st.metric(
        label="Total Orders",
        value=f"{kpis['total_orders']:,}",
        delta=None
    )

with col4:
    st.metric(
        label="Profit Margin",
        value=f"{kpis['profit_margin']:.2f}%",
        delta=None
    )

# Sales and Profit Over Time
st.markdown("## 📈 Sales and Profit Over Time")
col1, col2 = st.columns([2, 1])

with col1:
    fig_trend = create_sales_profit_trend(filtered_df)
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    fig_orders = create_orders_by_month(filtered_df)
    st.plotly_chart(fig_orders, use_container_width=True)

# Profit Breakdown
st.markdown("## 🌍 Profit Breakdown by Region and Segment")
col1, col2, col3 = st.columns(3)

with col1:
    fig_region = create_profit_by_region(filtered_df)
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    fig_segment = create_profit_by_segment(filtered_df)
    st.plotly_chart(fig_segment, use_container_width=True)

with col3:
    fig_heatmap = create_profit_heatmap(filtered_df)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Discount Impact and Product Insights
st.markdown("## 💡 Discount Impact and Product Insights")
col1, col2, col3 = st.columns(3)

with col1:
    fig_discount = create_discount_impact(filtered_df)
    st.plotly_chart(fig_discount, use_container_width=True)

with col2:
    fig_subcategories = create_top_subcategories(filtered_df)
    st.plotly_chart(fig_subcategories, use_container_width=True)

with col3:
    fig_category = create_category_distribution(filtered_df)
    st.plotly_chart(fig_category, use_container_width=True)

# Data summary
with st.expander("📊 View Filtered Data Summary"):
    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.info(f"Showing 100 of {len(filtered_df):,} filtered records")

# Footer
st.markdown("---")
st.markdown("Dashboard designed with Streamlit • Interactive visualizations powered by Plotly • AI Assistant • Dataset: Superstore")

# ======= CHAT PANEL (RIGHT SIDE) =======
st.markdown("""
<style>
.chat-panel {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px;
    padding: 14px 14px 10px 14px;
    background: rgba(255,255,255,0.9);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}
.chat-title {
    font-weight: 700;
    font-size: 16px;
    display:flex;
    align-items:center;
    gap:8px;
    margin-bottom: 8px;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px 12px;
    border-radius: 14px;
    border-bottom-right-radius: 6px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.35;
    white-space: pre-wrap;
}
.chat-bubble-bot {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    color: #222;
    padding: 10px 12px;
    border-radius: 14px;
    border-bottom-left-radius: 6px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.35;
    white-space: pre-wrap;
}
.chat-hint {
    color: rgba(0,0,0,0.55);
    font-size: 12px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
left, right = st.columns([3, 1.2], gap="large")

with right:
    # Open / Close
    if not st.session_state.chat_open:
        if st.button("💬 Open Chat", use_container_width=True):
            st.session_state.chat_open = True
            st.rerun()
    else:
        topbar_l, topbar_r = st.columns([3, 1])
        with topbar_l:
            st.markdown(" ")
        with topbar_r:
            if st.button("✕", use_container_width=True):
                st.session_state.chat_open = False
                st.rerun()

        st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='chat-title'>🤖 Dashboard Assistant</div>", unsafe_allow_html=True)

        # Auto insights (optional)
        with st.sidebar.expander("💡 Automatic Insights", expanded=False):
            if "auto_insights" not in st.session_state:
                st.session_state.auto_insights = None

            if st.session_state.auto_insights is None:
                if st.button("Generate insights", use_container_width=True, key="gen_insights"):
                    st.session_state.auto_insights = chatbot.get_insights()
                    st.rerun()
            else:
                st.info(st.session_state.auto_insights)

            if st.button("Clear insights", use_container_width=True, key="clear_insights"):
                st.session_state.auto_insights = None
                st.rerun()


        # Show chat history
        if st.session_state.chat_history:
            for m in st.session_state.chat_history[-12:]:
                if m["role"] == "user":
                    st.markdown(f"<div class='chat-bubble-user'>{m['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bubble-bot'>{m['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='chat-bubble-bot'>Chào bạn! Hãy hỏi về KPI / trend / region / segment / discount / top sub-category nhé.</div>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='chat-hint'>Ví dụ: “Region nào profit cao nhất?”, “Sales trend tháng gần đây?”, “Discount ảnh hưởng profit ra sao?”</div>", unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_msg = st.text_input("Nhập câu hỏi…", placeholder="Hỏi như nhắn tin 🙂")
            c1, c2 = st.columns([3, 1])
            with c1:
                send = st.form_submit_button("Send")
            with c2:
                clear = st.form_submit_button("Clear")

            if send and user_msg.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_msg.strip()})
                ans = chatbot.get_response(user_msg.strip())
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

            if clear:
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
