import pandas as pd
import streamlit as st
from src.database import execute_query
from config import Config

@st.cache_data(ttl=3600)
def load_superstore_data():
    """Load all data from superstore table"""
    query = f"SELECT * FROM {Config.DB_TABLE}"
    df = execute_query(query)
    
    if not df.empty:
        # Convert date columns
        date_columns = ['order_date', 'ship_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    
    return df

def filter_data(df, filters):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    # Date range filter
    if 'date_range' in filters:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['order_date'] >= pd.Timestamp(start_date)) &
            (filtered_df['order_date'] <= pd.Timestamp(end_date))
        ]
    
    # Region filter
    if 'region' in filters and filters['region']:
        filtered_df = filtered_df[filtered_df['region'].isin(filters['region'])]
    
    # Segment filter
    if 'segment' in filters and filters['segment']:
        filtered_df = filtered_df[filtered_df['segment'].isin(filters['segment'])]
    
    # Category filter
    if 'category' in filters and filters['category']:
        filtered_df = filtered_df[filtered_df['category'].isin(filters['category'])]
    
    return filtered_df

def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_sales = df['sales'].sum()
    total_profit = df['profit'].sum()
    total_orders = df['order_id'].nunique()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    
    return {
        'total_sales': total_sales,
        'total_profit': total_profit,
        'total_orders': total_orders,
        'profit_margin': profit_margin
    }
