"""
core/database.py
Infrastructure layer — connection pool management and query execution.
No UI logic here; callers handle error display.
"""

from __future__ import annotations

import logging
import socket
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import pandas as pd
import psycopg2
from psycopg2 import pool
import streamlit as st

from config import Config

logger = logging.getLogger(__name__)


def _resolve_ipv4(hostname: str) -> Optional[str]:
    """Resolve hostname to IPv4 address (required for Supabase)."""
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        if addr_info:
            return addr_info[0][4][0]
    except socket.gaierror as exc:
        logger.warning("DNS resolution failed for %s: %s", hostname, exc)
    return None


@st.cache_resource
def _get_pool() -> Optional[psycopg2.pool.SimpleConnectionPool]:
    """Create and cache the connection pool (one per Streamlit session)."""
    host = _resolve_ipv4(Config.DB_HOST) or Config.DB_HOST
    try:
        return psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=host,
            port=int(Config.DB_PORT),
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            sslmode="require",
            connect_timeout=10,
        )
    except Exception as exc:
        logger.error("Failed to create connection pool: %s", exc)
        return None


@contextmanager
def get_connection() -> Generator:
    """Context manager: borrow a connection from the pool and return it afterwards."""
    db_pool = _get_pool()
    conn = None
    try:
        if db_pool:
            conn = db_pool.getconn()
        yield conn
    except Exception as exc:
        logger.error("DB connection error: %s", exc)
        yield None
    finally:
        if db_pool and conn:
            db_pool.putconn(conn)


def execute_query(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute a parameterised query and return a DataFrame. Returns empty DF on failure."""
    with get_connection() as conn:
        if conn is None:
            return pd.DataFrame()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except Exception as exc:
            logger.error("Query execution error: %s", exc)
            return pd.DataFrame()


def is_connected() -> bool:
    """Quick health-check — returns True if a connection can be borrowed."""
    with get_connection() as conn:
        return conn is not None
