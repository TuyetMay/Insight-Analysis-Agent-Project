import pandas as pd
import re
from datetime import datetime

class DashboardChatbot:
    """
    Smart chatbot for Superstore Dashboard
    Phase 1: Basic Q&A based on filtered data and KPIs
    Expandable for future enhancements
    """
    
    def __init__(self, df, kpis, filters):
        """
        Initialize chatbot with dashboard context
        
        Args:
            df: Filtered dataframe
            kpis: Dictionary of KPIs
            filters: Active filters
        """
        self.df = df
        self.kpis = kpis
        self.filters = filters
        
    def get_response(self, user_message):
        """
        Main method to process user message and generate response
        """
        user_message = user_message.lower().strip()
        
        # Pattern matching for different question types
        response = None
        
        # KPI Questions
        if self._is_asking_about_kpi(user_message):
            response = self._answer_kpi_question(user_message)
        
        # Comparison Questions
        elif self._is_asking_comparison(user_message):
            response = self._answer_comparison_question(user_message)
        
        # Top/Bottom Questions
        elif self._is_asking_top_bottom(user_message):
            response = self._answer_top_bottom_question(user_message)
        
        # Filter Status Questions
        elif self._is_asking_filters(user_message):
            response = self._answer_filter_question(user_message)
        
        # Trend Questions
        elif self._is_asking_trend(user_message):
            response = self._answer_trend_question(user_message)
        
        # Default: Provide helpful suggestions
        else:
            response = self._get_default_response(user_message)
        
        return response
    
    # ========== Pattern Detection Methods ==========
    
    def _is_asking_about_kpi(self, message):
        """Detect if asking about KPIs"""
        kpi_keywords = ['total sales', 'total profit', 'total order', 'profit margin', 
                       'revenue', 'how much', 'what is the total', 'sales']
        return any(keyword in message for keyword in kpi_keywords)
    
    def _is_asking_comparison(self, message):
        """Detect comparison questions"""
        comparison_keywords = ['compare', 'vs', 'versus', 'difference between', 
                              'better', 'higher', 'lower']
        return any(keyword in message for keyword in comparison_keywords)
    
    def _is_asking_top_bottom(self, message):
        """Detect top/bottom questions"""
        top_bottom_keywords = ['top', 'best', 'highest', 'most', 'bottom', 
                              'worst', 'lowest', 'least']
        return any(keyword in message for keyword in top_bottom_keywords)
    
    def _is_asking_filters(self, message):
        """Detect filter status questions"""
        filter_keywords = ['filter', 'selected', 'current', 'showing', 
                          'what data', 'which region', 'which segment']
        return any(keyword in message for keyword in filter_keywords)
    
    def _is_asking_trend(self, message):
        """Detect trend questions"""
        trend_keywords = ['trend', 'over time', 'growing', 'increasing', 
                         'decreasing', 'change']
        return any(keyword in message for keyword in trend_keywords)
    
    # ========== Answer Generation Methods ==========
    
    def _answer_kpi_question(self, message):
        """Answer KPI-related questions"""
        if 'sales' in message or 'revenue' in message:
            return f"💰 **Total Sales:** ${self.kpis['total_sales']:,.2f}\n\nThis represents all sales from {len(self.df):,} transactions in the filtered data."
        
        elif 'profit' in message and 'margin' not in message:
            return f"📈 **Total Profit:** ${self.kpis['total_profit']:,.2f}\n\nProfit margin: {self.kpis['profit_margin']:.2f}%"
        
        elif 'margin' in message:
            return f"📊 **Profit Margin:** {self.kpis['profit_margin']:.2f}%\n\nThis is calculated as (Total Profit / Total Sales) × 100"
        
        elif 'order' in message:
            return f"🛒 **Total Orders:** {self.kpis['total_orders']:,}\n\nWith an average order value of ${self.kpis['total_sales']/self.kpis['total_orders']:,.2f}"
        
        # Summary if can't determine specific KPI
        return f"""📌 **Key Metrics Summary:**
- Total Sales: ${self.kpis['total_sales']:,.2f}
- Total Profit: ${self.kpis['total_profit']:,.2f}
- Total Orders: {self.kpis['total_orders']:,}
- Profit Margin: {self.kpis['profit_margin']:.2f}%"""
    
    def _answer_comparison_question(self, message):
        """Answer comparison questions"""
        # Region comparison
        if 'region' in message:
            region_stats = self.df.groupby('region').agg({
                'sales': 'sum',
                'profit': 'sum'
            }).sort_values('profit', ascending=False)
            
            response = "🌍 **Profit by Region (Ranked):**\n\n"
            for idx, (region, row) in enumerate(region_stats.iterrows(), 1):
                response += f"{idx}. **{region}**: ${row['profit']:,.2f}\n"
            
            return response
        
        # Segment comparison
        elif 'segment' in message:
            segment_stats = self.df.groupby('segment').agg({
                'sales': 'sum',
                'profit': 'sum'
            }).sort_values('profit', ascending=False)
            
            response = "👥 **Profit by Segment (Ranked):**\n\n"
            for idx, (segment, row) in enumerate(segment_stats.iterrows(), 1):
                response += f"{idx}. **{segment}**: ${row['profit']:,.2f}\n"
            
            return response
        
        # Category comparison
        elif 'category' in message:
            category_stats = self.df.groupby('category').agg({
                'sales': 'sum',
                'profit': 'sum'
            }).sort_values('profit', ascending=False)
            
            response = "📦 **Profit by Category (Ranked):**\n\n"
            for idx, (category, row) in enumerate(category_stats.iterrows(), 1):
                response += f"{idx}. **{category}**: ${row['profit']:,.2f}\n"
            
            return response
        
        return "I can compare regions, segments, or categories. Try asking: 'Compare regions' or 'Which segment is better?'"
    
    def _answer_top_bottom_question(self, message):
        """Answer top/bottom performance questions"""
        # Determine if asking for top or bottom
        is_bottom = any(word in message for word in ['bottom', 'worst', 'lowest', 'least'])
        
        # Extract number if specified (default 5)
        numbers = re.findall(r'\d+', message)
        n = int(numbers[0]) if numbers else 5
        
        # Determine what to rank
        if 'product' in message or 'sub-category' in message or 'subcategory' in message:
            ranking = self.df.groupby('sub_category')['profit'].sum()
            ranking = ranking.nsmallest(n) if is_bottom else ranking.nlargest(n)
            
            title = f"{'Bottom' if is_bottom else 'Top'} {n} Sub-Categories by Profit"
            response = f"🏆 **{title}:**\n\n"
            
            for idx, (subcat, profit) in enumerate(ranking.items(), 1):
                response += f"{idx}. **{subcat}**: ${profit:,.2f}\n"
            
            return response
        
        elif 'customer' in message:
            ranking = self.df.groupby('customer_name')['profit'].sum()
            ranking = ranking.nsmallest(n) if is_bottom else ranking.nlargest(n)
            
            title = f"{'Bottom' if is_bottom else 'Top'} {n} Customers by Profit"
            response = f"👤 **{title}:**\n\n"
            
            for idx, (customer, profit) in enumerate(ranking.items(), 1):
                response += f"{idx}. **{customer}**: ${profit:,.2f}\n"
            
            return response
        
        elif 'region' in message:
            ranking = self.df.groupby('region')['profit'].sum()
            ranking = ranking.nsmallest(n) if is_bottom else ranking.nlargest(n)
            
            title = f"{'Bottom' if is_bottom else 'Top'} Regions by Profit"
            response = f"🌍 **{title}:**\n\n"
            
            for idx, (region, profit) in enumerate(ranking.items(), 1):
                response += f"{idx}. **{region}**: ${profit:,.2f}\n"
            
            return response
        
        return "I can show top/bottom products, customers, or regions. Try: 'Show me top 5 products' or 'What are the worst performing regions?'"
    
    def _answer_filter_question(self, message):
        """Answer questions about current filters"""
        response = "🔍 **Current Filters Applied:**\n\n"
        
        # Date range
        start_date, end_date = self.filters['date_range']
        response += f"📅 **Date Range:** {start_date} to {end_date}\n\n"
        
        # Regions
        if self.filters['region']:
            response += f"🌍 **Regions:** {', '.join(self.filters['region'])}\n\n"
        else:
            response += "🌍 **Regions:** All regions\n\n"
        
        # Segments
        if self.filters['segment']:
            response += f"👥 **Segments:** {', '.join(self.filters['segment'])}\n\n"
        else:
            response += "👥 **Segments:** All segments\n\n"
        
        # Categories
        if self.filters['category']:
            response += f"📦 **Categories:** {', '.join(self.filters['category'])}\n\n"
        else:
            response += "📦 **Categories:** All categories\n\n"
        
        response += f"📊 **Total Records:** {len(self.df):,} transactions"
        
        return response
    
    def _answer_trend_question(self, message):
        """Answer trend-related questions"""
        # Calculate monthly trends
        monthly_data = self.df.groupby(self.df['order_date'].dt.to_period('M')).agg({
            'sales': 'sum',
            'profit': 'sum'
        })
        
        if len(monthly_data) < 2:
            return "Not enough data to determine trends. Try adjusting your date range filter."
        
        # Calculate growth
        first_month_sales = monthly_data['sales'].iloc[0]
        last_month_sales = monthly_data['sales'].iloc[-1]
        sales_growth = ((last_month_sales - first_month_sales) / first_month_sales) * 100
        
        first_month_profit = monthly_data['profit'].iloc[0]
        last_month_profit = monthly_data['profit'].iloc[-1]
        profit_growth = ((last_month_profit - first_month_profit) / first_month_profit) * 100
        
        avg_monthly_sales = monthly_data['sales'].mean()
        avg_monthly_profit = monthly_data['profit'].mean()
        
        response = f"""📈 **Trend Analysis:**

**Sales Trend:**
- First Month: ${first_month_sales:,.2f}
- Last Month: ${last_month_sales:,.2f}
- Growth: {sales_growth:+.1f}%
- Monthly Average: ${avg_monthly_sales:,.2f}

**Profit Trend:**
- First Month: ${first_month_profit:,.2f}
- Last Month: ${last_month_profit:,.2f}
- Growth: {profit_growth:+.1f}%
- Monthly Average: ${avg_monthly_profit:,.2f}

{'📊 Sales are increasing!' if sales_growth > 0 else '📉 Sales are decreasing.'}
{'💰 Profit is growing!' if profit_growth > 0 else '⚠️ Profit is declining.'}
"""
        
        return response
    
    def _get_default_response(self, message):
        """Provide helpful suggestions when question is not recognized"""
        return """👋 I can help you understand the dashboard data! Try asking:

**KPIs & Metrics:**
- "What are the total sales?"
- "Show me the profit margin"
- "How many orders do we have?"

**Comparisons:**
- "Compare regions"
- "Which segment performs better?"
- "Compare categories"

**Rankings:**
- "Show top 5 products"
- "What are the best customers?"
- "Which regions perform worst?"

**Filters & Context:**
- "What filters are applied?"
- "What data am I looking at?"

**Trends:**
- "What's the sales trend?"
- "Is profit growing?"

Feel free to ask anything about the current dashboard data! 📊"""

    def get_insights(self):
        """Generate automatic insights from the data"""
        insights = []
        
        # Profit margin insight
        if self.kpis['profit_margin'] < 10:
            insights.append("⚠️ **Low Profit Margin Alert**: Your profit margin is below 10%. Consider reviewing pricing strategy or cost optimization.")
        elif self.kpis['profit_margin'] > 20:
            insights.append("✅ **Healthy Profit Margin**: Your profit margin above 20% indicates strong profitability!")
        
        # Best performing segment
        segment_profit = self.df.groupby('segment')['profit'].sum()
        best_segment = segment_profit.idxmax()
        insights.append(f"🎯 **Top Segment**: {best_segment} is your most profitable segment (${segment_profit.max():,.2f})")
        
        # Discount impact
        high_discount = self.df[self.df['discount'] > 0.3]
        if len(high_discount) > 0:
            avg_profit_high_discount = high_discount['profit'].mean()
            avg_profit_low_discount = self.df[self.df['discount'] <= 0.3]['profit'].mean()
            
            if avg_profit_high_discount < avg_profit_low_discount:
                insights.append(f"💡 **Discount Insight**: High discounts (>30%) are reducing profitability. Average profit with high discount: ${avg_profit_high_discount:.2f} vs ${avg_profit_low_discount:.2f} with lower discounts.")
        
        return "\n\n".join(insights)