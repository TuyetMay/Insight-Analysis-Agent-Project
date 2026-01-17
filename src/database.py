# # file: database.py
# import psycopg2
# from psycopg2 import pool
# import pandas as pd
# import streamlit as st
# from contextlib import contextmanager
# from config import Config

# # 1. Khởi tạo Connection Pool và cache nó (chứ không cache connection lẻ)
# @st.cache_resource
# def get_connection_pool():
#     try:
#         # Tạo pool với tối thiểu 1 và tối đa 10 kết nối
#         db_pool = psycopg2.pool.SimpleConnectionPool(
#             1, 10,
#             host=Config.DB_HOST,
#             port=Config.DB_PORT,
#             database=Config.DB_NAME,
#             user=Config.DB_USER,
#             password=Config.DB_PASSWORD
#         )
#         return db_pool
#     except Exception as e:
#         st.error(f"❌ Error creating connection pool: {e}")
#         return None

# # 2. Context Manager để lấy và trả kết nối tự động
# @contextmanager
# def get_db_connection():
#     db_pool = get_connection_pool()
#     conn = None
#     try:
#         if db_pool:
#             conn = db_pool.getconn()
#             yield conn
#     except Exception as e:
#         st.error(f"❌ Database connection error: {e}")
#         raise e
#     finally:
#         if db_pool and conn:
#             # Quan trọng: Trả kết nối về pool để tái sử dụng
#             db_pool.putconn(conn)

# # 3. Hàm execute query cải tiến
# def execute_query(query, params=None):
#     """Execute query and return DataFrame safely"""
#     with get_db_connection() as conn:
#         if conn:
#             try:
#                 # pandas read_sql_query tự động dùng conn và đóng cursor
#                 df = pd.read_sql_query(query, conn, params=params)
#                 return df
#             except Exception as e:
#                 st.error(f"❌ Query execution error: {e}")
#                 return pd.DataFrame()
#     return pd.DataFrame()


from contextlib import contextmanager
import socket
import psycopg2
from psycopg2 import pool
import pandas as pd
import streamlit as st
from config import Config

def resolve_ipv4(host: str) -> str:
    # Lấy IPv4 (AF_INET) để tránh psycopg2 chọn IPv6 trước
    infos = socket.getaddrinfo(host, None, socket.AF_INET)
    return infos[0][4][0]

@st.cache_resource
def get_connection_pool():
    try:
        ipv4 = resolve_ipv4(Config.DB_HOST)
        return psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=Config.DB_HOST,
            hostaddr=ipv4,              # ✅ ép IPv4
            port=Config.DB_PORT,
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            sslmode="require"           # ✅ Supabase cần SSL
        )
    except Exception as e:
        st.error(f"❌ Error creating connection pool: {e}")
        return None

@contextmanager
def get_db_connection():
    p = get_connection_pool()
    if p is None:
        yield None                     # ✅ luôn yield để tránh "generator didn't yield"
        return
    conn = None
    try:
        conn = p.getconn()
        yield conn
    finally:
        if conn is not None:
            p.putconn(conn)

def execute_query(query, params=None):
    with get_db_connection() as conn:
        if conn is None:
            return pd.DataFrame()
        return pd.read_sql_query(query, conn, params=params)
