from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("backend/data/training_metrics.db")


def init_training_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                loss REAL NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                map REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_run_id_epoch
            ON runs(run_id, epoch)
            """
        )
        conn.commit()


def insert_training_metric(
    run_id: str,
    epoch: int,
    loss: float,
    accuracy: float,
    precision: float,
    recall: float,
    map_score: float,
    timestamp: str,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, epoch, loss, accuracy, precision, recall, map, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, epoch, loss, accuracy, precision, recall, map_score, timestamp),
        )
        conn.commit()


def get_training_metrics(run_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT run_id, epoch, loss, accuracy, precision, recall, map, timestamp
            FROM runs
            WHERE run_id = ?
            ORDER BY epoch ASC
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_training_runs() -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                run_id,
                COUNT(*) AS epochs_recorded,
                MIN(timestamp) AS started_at,
                MAX(timestamp) AS updated_at,
                MIN(loss) AS best_loss,
                MAX(accuracy) AS best_accuracy,
                MAX(map) AS best_map
            FROM runs
            GROUP BY run_id
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]

