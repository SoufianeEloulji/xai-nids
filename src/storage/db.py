import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any, Dict, List
from src import logger

DB_PATH = "artifacts/predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    input_json TEXT NOT NULL,
    top_features_json TEXT NOT NULL
);
"""


@contextmanager
def get_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes the SQLite database and creates the predictions table if it doesn't exist."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(SCHEMA)
            conn.commit()
        logger.info(f"Initialized database: {db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize the database: {e}")
        raise


def insert_prediction(
    class_id: int,
    label: str,
    confidence: float,
    input_data: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> int:
    """Inserts a prediction record into the SQLite database.
    Returns the ID of the inserted record."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (timestamp, class_id, label, confidence, input_json, top_features_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    class_id,
                    label,
                    confidence,
                    json.dumps(input_data, default=str),
                    json.dumps(top_features),
                ),
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to insert prediction: {e}")
        raise


def fetch_recent(limit: int = 200, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Returns the `limit` most recent predictions (descending order)."""
    try:
        with get_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch recent predictions: {e}")
        raise


def fetch_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Returns aggregated statistics on all stored predictions."""
    try:
        with get_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
            by_label = conn.execute(
                "SELECT label, COUNT(*) as c FROM predictions GROUP BY label ORDER BY c DESC"
            ).fetchall()
            return {"total": total, "by_label": [dict(r) for r in by_label]}
    except Exception as e:
        logger.error(f"Failed to fetch statistics: {e}")
        raise