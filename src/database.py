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


# file: src/database.py
from contextlib import contextmanager
import socket
import psycopg2
from psycopg2 import pool
import pandas as pd
import streamlit as st
from config import Config

def resolve_to_ipv4(hostname):
    """Force IPv4 resolution for Supabase"""
    try:
        # Get all addresses
        addr_info = socket.getaddrinfo(
            hostname, 
            None, 
            socket.AF_INET,  # Force IPv4
            socket.SOCK_STREAM
        )
        if addr_info:
            ipv4 = addr_info[0][4][0]
            st.success(f"✅ Resolved {hostname} → {ipv4}")
            return ipv4
    except socket.gaierror as e:
        st.error(f"❌ DNS resolution failed: {e}")
        return None

@st.cache_resource
def get_connection_pool():
    """Create connection pool for Supabase PostgreSQL"""
    try:
        # Force IPv4 resolution
        ipv4_host = resolve_to_ipv4(Config.DB_HOST)
        
        if not ipv4_host:
            st.error("Cannot resolve hostname to IPv4")
            st.info("💡 Solution: Switch to 'Session Pooler' in Supabase connection settings")
            return None
        
        connection_params = {
            'host': ipv4_host,  # Use resolved IPv4
            'port': int(Config.DB_PORT),
            'dbname': Config.DB_NAME,
            'user': Config.DB_USER,
            'password': Config.DB_PASSWORD,
            'sslmode': 'require',
            'connect_timeout': 10
        }
        
        db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            **connection_params
        )
        
        st.success("✅ Connection pool created successfully!")
        return db_pool
        
    except Exception as e:
        st.error(f"❌ Error creating connection pool: {e}")
        st.error(f"Connection: {Config.DB_HOST}:{Config.DB_PORT} → DB: {Config.DB_NAME}")
        st.info("💡 Try switching to Session Pooler in Supabase settings")
        return None

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    db_pool = get_connection_pool()
    conn = None
    try:
        if db_pool:
            conn = db_pool.getconn()
            yield conn
        else:
            yield None
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        yield None
    finally:
        if db_pool and conn:
            db_pool.putconn(conn)

def execute_query(query, params=None):
    """Execute query and return DataFrame safely"""
    with get_db_connection() as conn:
        if conn is None:
            return pd.DataFrame()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            st.error(f"❌ Query execution error: {e}")
            return pd.DataFrame()