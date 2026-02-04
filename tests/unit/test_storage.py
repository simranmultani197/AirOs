"""
Unit tests for the Storage module - SQLite Persistence.
"""
import pytest
import os
import json
import tempfile
import shutil
from datetime import datetime

from agentcircuit.storage import Storage


class TestStorageBasics:
    """Test basic Storage functionality."""

    def test_storage_init_creates_directory(self, temp_db_path):
        """Test Storage creates directory if not exists."""
        storage = Storage(db_path=temp_db_path)
        assert os.path.exists(os.path.dirname(temp_db_path))

    def test_storage_init_creates_tables(self, temp_db_path):
        """Test Storage creates required tables."""
        storage = Storage(db_path=temp_db_path)

        import sqlite3
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Check traces table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='traces'")
        assert cursor.fetchone() is not None

        # Check global_settings table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_settings'")
        assert cursor.fetchone() is not None

        conn.close()

    def test_storage_default_settings(self, temp_db_path):
        """Test Storage initializes with default settings."""
        storage = Storage(db_path=temp_db_path)

        cost_per_token = storage.get_setting("cost_per_token")
        assert cost_per_token == "0.000005"


class TestStorageTraceLogging:
    """Test trace logging functionality."""

    def test_log_trace_basic(self, temp_db_path):
        """Test basic trace logging."""
        storage = Storage(db_path=temp_db_path)

        storage.log_trace(
            run_id="test-run-1",
            node_id="test_node",
            input_state={"input": "test"},
            output_state={"output": "result"},
            status="success",
            recovery_attempts=0
        )

        history = storage.get_run_history("test-run-1")
        assert len(history) == 1
        assert history[0]["node_id"] == "test_node"
        assert history[0]["status"] == "success"

    def test_log_trace_with_all_fields(self, temp_db_path):
        """Test trace logging with all fields."""
        storage = Storage(db_path=temp_db_path)

        storage.log_trace(
            run_id="test-run-2",
            node_id="complex_node",
            input_state={"messages": ["hello"]},
            output_state={"response": "world"},
            status="repaired",
            cost_tokens=100,
            recovery_attempts=1,
            saved_cost=0.05,
            token_usage=200,
            estimated_cost=0.001,
            diagnosis="Fixed JSON parsing error",
            duration_ms=1500.5
        )

        history = storage.get_run_history("test-run-2")
        assert len(history) == 1
        assert history[0]["status"] == "repaired"
        assert history[0]["diagnosis"] == "Fixed JSON parsing error"

    def test_log_trace_json_serialization(self, temp_db_path):
        """Test trace logging handles complex state serialization."""
        storage = Storage(db_path=temp_db_path)

        complex_state = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "unicode": "Hello, \u4e16\u754c"
        }

        storage.log_trace(
            run_id="test-run-3",
            node_id="json_node",
            input_state=complex_state,
            output_state=complex_state,
            status="success",
            recovery_attempts=0
        )

        history = storage.get_run_history("test-run-3")
        assert history[0]["input_state"] == complex_state

    def test_log_multiple_traces_same_run(self, temp_db_path):
        """Test logging multiple traces for same run."""
        storage = Storage(db_path=temp_db_path)

        for i in range(5):
            storage.log_trace(
                run_id="multi-run",
                node_id=f"node_{i}",
                input_state={"step": i},
                output_state={"result": i},
                status="success",
                recovery_attempts=0
            )

        history = storage.get_run_history("multi-run")
        assert len(history) == 5
        assert [h["node_id"] for h in history] == [f"node_{i}" for i in range(5)]

    def test_log_trace_different_runs(self, temp_db_path):
        """Test traces are separated by run_id."""
        storage = Storage(db_path=temp_db_path)

        storage.log_trace(
            run_id="run-a",
            node_id="node",
            input_state={},
            output_state={},
            status="success",
            recovery_attempts=0
        )

        storage.log_trace(
            run_id="run-b",
            node_id="node",
            input_state={},
            output_state={},
            status="failed",
            recovery_attempts=0
        )

        history_a = storage.get_run_history("run-a")
        history_b = storage.get_run_history("run-b")

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["status"] == "success"
        assert history_b[0]["status"] == "failed"


class TestStorageSettings:
    """Test settings management."""

    def test_get_setting_exists(self, temp_db_path):
        """Test getting existing setting."""
        storage = Storage(db_path=temp_db_path)

        # Default setting should exist
        value = storage.get_setting("cost_per_token")
        assert value is not None

    def test_get_setting_not_exists(self, temp_db_path):
        """Test getting non-existent setting returns None."""
        storage = Storage(db_path=temp_db_path)

        value = storage.get_setting("nonexistent_key")
        assert value is None

    def test_set_setting_new(self, temp_db_path):
        """Test setting a new setting."""
        storage = Storage(db_path=temp_db_path)

        storage.set_setting("custom_setting", "custom_value")

        value = storage.get_setting("custom_setting")
        assert value == "custom_value"

    def test_set_setting_update(self, temp_db_path):
        """Test updating existing setting."""
        storage = Storage(db_path=temp_db_path)

        storage.set_setting("cost_per_token", "0.00001")

        value = storage.get_setting("cost_per_token")
        assert value == "0.00001"


class TestStorageRunCost:
    """Test run cost calculation."""

    def test_get_run_cost_empty(self, temp_db_path):
        """Test run cost for non-existent run."""
        storage = Storage(db_path=temp_db_path)

        cost = storage.get_run_cost("nonexistent-run")
        assert cost == 0.0

    def test_get_run_cost_single_trace(self, temp_db_path):
        """Test run cost with single trace."""
        storage = Storage(db_path=temp_db_path)

        storage.log_trace(
            run_id="cost-run-1",
            node_id="node",
            input_state={},
            output_state={},
            status="success",
            estimated_cost=0.05,
            recovery_attempts=0
        )

        cost = storage.get_run_cost("cost-run-1")
        assert cost == 0.05

    def test_get_run_cost_multiple_traces(self, temp_db_path):
        """Test run cost sums multiple traces."""
        storage = Storage(db_path=temp_db_path)

        for i in range(3):
            storage.log_trace(
                run_id="cost-run-2",
                node_id=f"node_{i}",
                input_state={},
                output_state={},
                status="success",
                estimated_cost=0.1,
                recovery_attempts=0
            )

        cost = storage.get_run_cost("cost-run-2")
        assert abs(cost - 0.3) < 0.001  # Float comparison


class TestStorageRunHistory:
    """Test run history retrieval."""

    def test_get_run_history_empty(self, temp_db_path):
        """Test history for non-existent run."""
        storage = Storage(db_path=temp_db_path)

        history = storage.get_run_history("nonexistent-run")
        assert history == []

    def test_get_run_history_ordered(self, temp_db_path):
        """Test history is ordered by insertion time."""
        storage = Storage(db_path=temp_db_path)

        for i in range(5):
            storage.log_trace(
                run_id="ordered-run",
                node_id=f"node_{i}",
                input_state={"order": i},
                output_state={},
                status="success",
                recovery_attempts=0
            )

        history = storage.get_run_history("ordered-run")
        orders = [h["input_state"]["order"] for h in history]
        assert orders == [0, 1, 2, 3, 4]

    def test_get_run_history_includes_diagnosis(self, temp_db_path):
        """Test history includes diagnosis field."""
        storage = Storage(db_path=temp_db_path)

        storage.log_trace(
            run_id="diagnosis-run",
            node_id="node",
            input_state={},
            output_state={},
            status="repaired",
            diagnosis="Fixed validation error",
            recovery_attempts=1
        )

        history = storage.get_run_history("diagnosis-run")
        assert history[0]["diagnosis"] == "Fixed validation error"


class TestStorageMigrations:
    """Test database migrations."""

    def test_migration_adds_missing_columns(self, temp_db_path):
        """Test migration adds missing columns to existing table."""
        # Create old-style table without new columns
        os.makedirs(os.path.dirname(temp_db_path), exist_ok=True)

        import sqlite3
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                node_id TEXT,
                input_state TEXT,
                output_state TEXT,
                status TEXT,
                cost_tokens INTEGER,
                recovery_attempts INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        # Initialize Storage - should run migrations
        storage = Storage(db_path=temp_db_path)

        # Verify new columns exist by logging with them
        storage.log_trace(
            run_id="migration-test",
            node_id="node",
            input_state={},
            output_state={},
            status="success",
            saved_cost=0.1,
            token_usage=100,
            estimated_cost=0.001,
            diagnosis="test",
            duration_ms=500.0,
            recovery_attempts=0
        )

        # Should not raise
        history = storage.get_run_history("migration-test")
        assert len(history) == 1
