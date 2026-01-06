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
# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .stSelectbox, .stMultiSelect {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown(f"<div class='main-header'>{Config.APP_ICON} {Config.APP_TITLE}</div>", unsafe_allow_html=True)

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
st.markdown("Dashboard designed with Streamlit • Interactive visualizations powered by Plotly • Dataset: Superstore")
