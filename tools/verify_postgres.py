"""Create an isolated test database, run verification, and remove that database."""
import os
import subprocess
import sys
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import psycopg
from psycopg import sql


def main():
    dsn = os.environ["DATABASE_URL"]
    parsed = urlsplit(dsn)
    name = "ayurip_check_" + uuid.uuid4().hex
    if not name.startswith("ayurip_check_") or name == parsed.path.lstrip("/"):
        raise ValueError("invalid disposable database name")
    connection = psycopg.connect(dsn, autocommit=True)
    before = connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0]
    created = False
    try:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        environment = dict(os.environ)
        environment["AYURIP_TEST_DATABASE_URL"] = urlunsplit((parsed.scheme, parsed.netloc, "/" + name, parsed.query, parsed.fragment))
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_postgres_integration.py", "-q"], env=environment)
        after = connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0]
        report = {"pass": result.returncode == 0 and before == after, "isolated_database": True,
                  "production_active_generation_unchanged": before == after,
                  "active_generation_id": after, "checked_at": datetime.now(timezone.utc).isoformat()}
        target = Path(__file__).resolve().parents[1] / "reports/verification/postgres.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["pass"]:
            raise ValueError("PostgreSQL verification failed")
        return result.returncode
    finally:
        if created:
            connection.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
