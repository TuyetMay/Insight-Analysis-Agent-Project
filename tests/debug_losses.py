# test/debug_losses.py

# Check 1: Có bao nhiêu orders profit < 0?
from core.database import execute_query


df1 = execute_query("SELECT COUNT(*) as cnt, SUM(profit) as total FROM superstore WHERE profit < 0")
print("Loss orders:", df1)

# Check 2: Sub-categories nào có tổng profit < 0?
df2 = execute_query("""
    SELECT sub_category, SUM(profit) as total_profit
    FROM superstore
    GROUP BY sub_category
    HAVING SUM(profit) < 0
    ORDER BY SUM(profit) ASC
""")
print("Loss sub-categories:", df2)

# Check 3: Categories nào có tổng profit < 0?
df3 = execute_query("""
    SELECT sub_category, category, SUM(profit) as total_profit
    FROM superstore
    GROUP BY sub_category, category
    HAVING SUM(profit) < 0
    ORDER BY SUM(profit) ASC
""")
print("Loss sub+category combos:", df3)