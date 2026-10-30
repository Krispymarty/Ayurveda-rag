"""Explicit initialization for developer CLIs, outside their timed query loops."""
import time


def initialize_for_cli(timeout=60):
    from . import search_api
    search_api.readiness.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = search_api.readiness.status()
        if status["state"] == "ready":
            return search_api._ensure_ready()
        if status["state"] == "failed":
            raise ValueError(status["reason"])
        time.sleep(.025)
    raise ValueError("initialization_timeout")
