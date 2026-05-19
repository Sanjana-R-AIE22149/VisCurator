import pytest
import sqlite3
from pathlib import Path
from backend.training.db import init_training_db, insert_training_metric, get_training_metrics, list_training_runs

# We will patch DB_PATH to use an in-memory or temporary database for unit testing
@pytest.fixture(autouse=True)
def mock_db_path(monkeypatch, tmp_path):
    temp_db = tmp_path / "test_metrics.db"
    monkeypatch.setattr("backend.training.db.DB_PATH", temp_db)
    init_training_db()
    yield temp_db

def test_init_training_db_creates_tables(mock_db_path):
    """UNIT TEST: Verifies the database initialization creates the proper schema."""
    conn = sqlite3.connect(mock_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs';")
    assert cursor.fetchone() is not None

def test_insert_and_get_training_metric():
    """UNIT TEST: Verifies we can insert a metric and retrieve it accurately."""
    run_id = "test_run_123"
    insert_training_metric(
        run_id=run_id,
        epoch=1,
        loss=0.45,
        accuracy=0.88,
        precision=0.85,
        recall=0.89,
        map_score=0.87,
        timestamp="2026-05-19T10:00:00"
    )
    
    metrics = get_training_metrics(run_id)
    assert len(metrics) == 1
    assert metrics[0]["run_id"] == run_id
    assert metrics[0]["loss"] == 0.45
    assert metrics[0]["accuracy"] == 0.88

def test_list_training_runs():
    """UNIT TEST: Verifies aggregation logic for listing multiple runs."""
    insert_training_metric("run_A", 1, 0.5, 0.8, 0.8, 0.8, 0.8, "T1")
    insert_training_metric("run_A", 2, 0.4, 0.9, 0.9, 0.9, 0.9, "T2")
    insert_training_metric("run_B", 1, 0.6, 0.7, 0.7, 0.7, 0.7, "T3")
    
    runs = list_training_runs()
    assert len(runs) == 2
    
    run_a_summary = next(r for r in runs if r["run_id"] == "run_A")
    assert run_a_summary["epochs_recorded"] == 2
    assert run_a_summary["best_loss"] == 0.4
    assert run_a_summary["best_accuracy"] == 0.9
