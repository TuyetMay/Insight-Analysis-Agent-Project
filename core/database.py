"""
core/database.py  — fixed version
Key change: _get_pool() no longer uses @st.cache_resource
because caching None permanently breaks all subsequent connections.
Instead, pool is stored in st.session_state with retry logic.
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
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        if addr_info:
            return addr_info[0][4][0]
    except socket.gaierror as exc:
        logger.warning("DNS resolution failed for %s: %s", hostname, exc)
    return None


def _get_pool() -> Optional[psycopg2.pool.SimpleConnectionPool]:
    """
    Get or create connection pool.
    FIXED: No longer uses @st.cache_resource to avoid caching None permanently.
    Pool is stored in st.session_state so it can be reset on retry.
    """
    # Check if we have a valid cached pool
    if "db_pool" in st.session_state and st.session_state["db_pool"] is not None:
        return st.session_state["db_pool"]

    # Try to create a new pool
    host = _resolve_ipv4(Config.DB_HOST) or Config.DB_HOST
    try:
        p = psycopg2.pool.SimpleConnectionPool(
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
        st.session_state["db_pool"] = p
        return p
    except Exception as exc:
        logger.error("Failed to create connection pool: %s", exc)
        st.session_state["db_pool"] = None   # store None but don't cache permanently
        return None


@contextmanager
def get_connection() -> Generator:
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
            try:
                db_pool.putconn(conn)
            except Exception:
                pass


def execute_query(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    with get_connection() as conn:
        if conn is None:
            return pd.DataFrame()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except Exception as exc:
            logger.error("Query execution error: %s", exc)
            return pd.DataFrame()


def is_connected() -> bool:
    with get_connection() as conn:
        return conn is not None