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
import threading
from typing import Any, Dict, Generator, Optional

import pandas as pd
import psycopg2
from psycopg2 import pool
import streamlit as st

from config import Config

logger = logging.getLogger(__name__)

_pool_lock = threading.Lock()
_shared_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

def _resolve_ipv4(hostname: str) -> Optional[str]:
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        if addr_info:
            return addr_info[0][4][0]
    except socket.gaierror as exc:
        logger.warning("DNS resolution failed for %s: %s", hostname, exc)
    return None


def _get_pool() -> Optional[psycopg2.pool.ThreadedConnectionPool]:
    """
    Module-level singleton pool — shared across all Streamlit sessions.
    ThreadedConnectionPool thay vì SimpleConnectionPool để thread-safe.
    """
    global _shared_pool
    if _shared_pool is not None:
        return _shared_pool
 
    with _pool_lock:
        if _shared_pool is not None:          # double-check sau khi acquire lock
            return _shared_pool
        host = _resolve_ipv4(Config.DB_HOST) or Config.DB_HOST
        try:
            _shared_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=15,                   # tổng connections dùng chung
                host=host,
                port=int(Config.DB_PORT),
                dbname=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                sslmode="require",
                connect_timeout=10,
            )
            logger.info("DB pool created (shared, threaded)")
            return _shared_pool
        except Exception as exc:
            logger.error("Failed to create shared pool: %s", exc)
            return None
 
def reset_pool() -> None:
    """Gọi từ UI retry button để force-rebuild pool."""
    global _shared_pool
    with _pool_lock:
        if _shared_pool:
            try:
                _shared_pool.closeall()
            except Exception:
                pass
        _shared_pool = None


@contextmanager
def get_connection() -> Generator:
    db_pool = _get_pool()
    conn = None
    try:
        if db_pool:
            conn = db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except Exception:
                logger.warning("Stale connection detected, replacing...")
                try:
                    db_pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = db_pool.getconn()
        yield conn
    except Exception as exc:
        logger.error("DB connection error: %s", exc)
        yield None
    finally:
        if db_pool and conn:
            try:
                if not conn.closed:
                    conn.rollback()   # reset transaction state
                db_pool.putconn(conn)
            except Exception:
                pass


def execute_query(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    with get_connection() as conn:
        if conn is None:
            return pd.DataFrame()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                return pd.DataFrame(rows, columns=cols)
        except Exception as exc:
            logger.error("Query execution error: %s", exc)
            return pd.DataFrame()


def is_connected() -> bool:
    with get_connection() as conn:
        return conn is not None