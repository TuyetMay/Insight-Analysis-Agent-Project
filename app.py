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

# ========== FLOATING CHAT SIDEBAR ==========
# Chat toggle in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## 💬 AI Assistant")

if st.sidebar.button("🤖 Open Chat Assistant", use_container_width=True):
    st.session_state.show_chat = True
    st.rerun()

if st.session_state.show_chat:
    # Create chat interface in sidebar
    with st.sidebar:
        st.markdown("### Chat with Dashboard AI")
        
        # Insights section
        with st.expander("💡 Automatic Insights", expanded=False):
            insights = chatbot.get_insights()
            st.info(insights)
        
        # Quick actions
        st.markdown("**⚡ Quick Questions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Summary", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "Show me a summary"})
                response = chatbot.get_response("show me a summary")
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("📈 Trend", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "What's the sales trend?"})
                response = chatbot.get_response("what's the sales trend")
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        with col2:
            if st.button("🏆 Top 5", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "Show top 5 products"})
                response = chatbot.get_response("show top 5 products")
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("🌍 Compare", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "Compare regions"})
                response = chatbot.get_response("compare regions")
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown("---")
        
        # Chat messages container with fixed height
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                # Display messages in reverse order (newest at bottom)
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**AI:** {message['content']}")
                    st.markdown("")
            else:
                st.info("👋 Hi! I'm your dashboard assistant. Ask me anything!")
                st.markdown("""
                **Try asking:**
                - What are the total sales?
                - Compare regions
                - Show top 5 products
                - What's the trend?
                """)
        
        # Chat input at bottom
        st.markdown("---")
        user_input = st.text_input("Type your question...", key="chat_input", label_visibility="collapsed")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            send_btn = st.button("Send 📤", use_container_width=True)
        with col2:
            if st.button("Clear 🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_btn and user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get bot response
            bot_response = chatbot.get_response(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            
            st.rerun()
        
        # Close button
        st.markdown("---")
        if st.button("✕ Close Chat", use_container_width=True):
            st.session_state.show_chat = False
            st.rerun()