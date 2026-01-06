import psycopg2
import pandas as pd
import streamlit as st
from config import Config

@st.cache_resource
def get_database_connection():
    """Create and cache database connection"""
    try:
        conn = psycopg2.connect(Config.get_db_connection_string())
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def execute_query(query, params=None):
    """Execute query and return DataFrame"""
    conn = get_database_connection()
    if conn:
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            st.error(f"Query execution error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()