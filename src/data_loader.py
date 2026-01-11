# file: src/data_loader.py
import pandas as pd
import streamlit as st
from src.database import execute_query
from config import Config

def _build_where_clause(filters):
    """
    Helper function: Xây dựng mệnh đề WHERE động từ bộ lọc.
    Trả về: (câu lệnh sql where, dictionary chứa tham số)
    """
    conditions = []
    params = {}

    # 1. Date Range
    if 'date_range' in filters and filters['date_range']:
        # Kiểm tra độ dài tuple (start, end)
        if len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            conditions.append("order_date >= %(start_date)s")
            conditions.append("order_date <= %(end_date)s")
            params['start_date'] = start_date
            params['end_date'] = end_date

    # 2. Region (Dùng tuple cho SQL IN clause)
    if 'region' in filters and filters['region']:
        conditions.append("region IN %(region)s")
        params['region'] = tuple(filters['region'])

    # 3. Segment
    if 'segment' in filters and filters['segment']:
        conditions.append("segment IN %(segment)s")
        params['segment'] = tuple(filters['segment'])

    # 4. Category
    if 'category' in filters and filters['category']:
        conditions.append("category IN %(category)s")
        params['category'] = tuple(filters['category'])

    # Ghép lại thành chuỗi WHERE
    if conditions:
        return "WHERE " + " AND ".join(conditions), params
    return "", params

@st.cache_data(ttl=600) # Giảm TTL xuống vì dữ liệu load nhanh hơn
def load_filtered_data(filters):
    """
    Chỉ tải dữ liệu ĐÃ LỌC từ database.
    Tối ưu hóa hiệu năng RAM và tốc độ mạng.
    """
    where_clause, params = _build_where_clause(filters)
    
    # Query thông minh hơn
    query = f"""
        SELECT * FROM {Config.DB_TABLE}
        {where_clause}
        ORDER BY order_date
    """
    
    df = execute_query(query, params)
    
    # Chuẩn hóa kiểu dữ liệu datetime nếu cần
    if not df.empty:
        date_columns = ['order_date', 'ship_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
    return df

def get_filter_options():
    """
    Tải danh sách unique cho các bộ lọc (Region, Segment...) 
    mà KHÔNG tải toàn bộ bảng sales.
    """
    query = f"""
    SELECT 
        ARRAY_AGG(DISTINCT region) as regions,
        ARRAY_AGG(DISTINCT segment) as segments,
        ARRAY_AGG(DISTINCT category) as categories,
        MIN(order_date) as min_date,
        MAX(order_date) as max_date
    FROM {Config.DB_TABLE}
    """
    df = execute_query(query)
    if not df.empty:
        return {
            'region': sorted(df.iloc[0]['regions']) if df.iloc[0]['regions'] else [],
            'segment': sorted(df.iloc[0]['segments']) if df.iloc[0]['segments'] else [],
            'category': sorted(df.iloc[0]['categories']) if df.iloc[0]['categories'] else [],
            'min_date': df.iloc[0]['min_date'],
            'max_date': df.iloc[0]['max_date']
        }
    return {}

# KPI vẫn tính trên DataFrame đã lọc (nhưng giờ DF này nhỏ hơn nhiều)
# Hoặc bạn có thể viết thêm hàm calculate_kpis_sql để tính trực tiếp dưới DB
def calculate_kpis(df):
    if df.empty:
        return {
            'total_sales': 0, 'total_profit': 0, 
            'total_orders': 0, 'profit_margin': 0
        }
        
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