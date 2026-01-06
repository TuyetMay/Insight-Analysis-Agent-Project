import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_sales_profit_trend(df):
    """Create sales and profit trend chart"""
    monthly_data = df.groupby(df['order_date'].dt.to_period('M')).agg({
        'sales': 'sum',
        'profit': 'sum'
    }).reset_index()
    monthly_data['order_date'] = monthly_data['order_date'].dt.to_timestamp()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data['order_date'],
        y=monthly_data['sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_data['order_date'],
        y=monthly_data['profit'],
        mode='lines+markers',
        name='Profit',
        line=dict(color='#17becf', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title='Sales and Profit Trends by Month',
        xaxis_title='Order Date',
        yaxis_title='Value',
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def create_orders_by_month(df):
    """Create orders per month bar chart"""
    monthly_orders = df.groupby(df['order_date'].dt.to_period('M')).agg({
        'order_id': 'nunique'
    }).reset_index()
    monthly_orders['order_date'] = monthly_orders['order_date'].dt.to_timestamp()
    
    fig = px.bar(
        monthly_orders,
        x='order_date',
        y='order_id',
        title='Total Orders per Month',
        labels={'order_id': 'Total Orders', 'order_date': 'Month'}
    )
    
    fig.update_traces(marker_color='#1f77b4')
    fig.update_layout(height=400)
    
    return fig

def create_profit_by_region(df):
    """Create profit by region bar chart"""
    region_profit = df.groupby('region')['profit'].sum().reset_index()
    
    fig = px.bar(
        region_profit,
        x='region',
        y='profit',
        title='Profit by Region',
        labels={'profit': 'Profit', 'region': 'Region'}
    )
    
    fig.update_traces(marker_color='#1f77b4')
    fig.update_layout(height=400)
    
    return fig

def create_profit_by_segment(df):
    """Create profit by segment bar chart"""
    segment_profit = df.groupby('segment')['profit'].sum().reset_index()
    
    fig = px.bar(
        segment_profit,
        x='segment',
        y='profit',
        title='Profit by Segment',
        labels={'profit': 'Profit', 'segment': 'Segment'}
    )
    
    fig.update_traces(marker_color='#1f77b4')
    fig.update_layout(height=400)
    
    return fig

def create_discount_impact(df):
    """Create discount impact chart"""
    df_copy = df.copy()
    df_copy['discount_bucket'] = pd.cut(
        df_copy['discount'],
        bins=[0, 0.1, 0.2, 0.3, 1.0],
        labels=['0-10%', '10-20%', '20-30%', '>30%'],
        include_lowest=True
    )
    
    discount_data = df_copy.groupby('discount_bucket', observed=True).agg({
        'sales': 'mean',
        'profit': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=discount_data['discount_bucket'],
        y=discount_data['sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=discount_data['discount_bucket'],
        y=discount_data['profit'],
        mode='lines+markers',
        name='Profit',
        line=dict(color='#17becf', width=2)
    ))
    
    fig.update_layout(
        title='Average Sales and Profit by Discount Level',
        xaxis_title='Discount Bucket',
        yaxis_title='Value',
        height=400
    )
    
    return fig

def create_discount_impact(df):
    """Create discount impact chart"""
    df_copy = df.copy()
    df_copy['discount_bucket'] = pd.cut(
        df_copy['discount'],
        bins=[0, 0.1, 0.2, 0.3, 1.0],
        labels=['0-10%', '10-20%', '20-30%', '>30%'],
        include_lowest=True
    )
    
    discount_data = df_copy.groupby('discount_bucket', observed=True).agg({
        'sales': 'mean',
        'profit': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=discount_data['discount_bucket'],
        y=discount_data['sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=discount_data['discount_bucket'],
        y=discount_data['profit'],
        mode='lines+markers',
        name='Profit',
        line=dict(color='#17becf', width=2)
    ))
    
    fig.update_layout(
        title='Average Sales and Profit by Discount Level',
        xaxis_title='Discount Bucket',
        yaxis_title='Value',
        height=400
    )
    
    return fig

def create_top_subcategories(df, top_n=10):
    """Create top subcategories by profit"""
    subcategory_profit = df.groupby('sub_category')['profit'].sum().nlargest(top_n).reset_index()
    
    fig = px.bar(
        subcategory_profit,
        x='profit',
        y='sub_category',
        orientation='h',
        title=f'Top {top_n} Sub-Categories by Profit',
        labels={'profit': 'Profit', 'sub_category': 'Sub-Category'}
    )
    
    fig.update_traces(marker_color='#1f77b4')
    fig.update_layout(height=400)
    
    return fig

def create_category_distribution(df):
    """Create profit distribution by category pie chart"""
    category_profit = df.groupby('category')['profit'].sum().reset_index()
    
    fig = px.pie(
        category_profit,
        values='profit',
        names='category',
        title='Profit Distribution by Category',
        hole=0.3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_profit_heatmap(df):
    """Create profit heatmap by region and segment"""
    heatmap_data = df.pivot_table(
        values='profit',
        index='segment',
        columns='region',
        aggfunc='sum'
    )
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x='Region', y='Segment', color='Profit'),
        title='Heatmap: Profit by Region & Segment',
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(height=400)
    
    return fig