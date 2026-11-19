"""Bounded, read-only connections for hosted corpus queries."""
from contextlib import contextmanager
import os

from .request_metrics import stage


def open_pool():
    from psycopg_pool import ConnectionPool

    dsn = os.environ["DATABASE_URL"]
    pool = ConnectionPool(
        dsn, min_size=1, max_size=3, max_waiting=4, timeout=1.5,
        kwargs={"autocommit": True, "connect_timeout": 3},
        configure=_read_only, open=False, name="ayurip-search",
    )
    pool.open()
    return pool


def _read_only(connection):
    connection.execute("SET default_transaction_read_only = on")
    connection.execute("SET statement_timeout = '1800ms'")


@contextmanager
def connection(pool, connect):
    if pool is None:
        with stage("db_connect"):
            db = connect()
        try:
            yield db
        finally:
            db.close()
        return
    with stage("db_checkout"):
        db = pool.getconn()
    try:
        yield db
    finally:
        pool.putconn(db)
