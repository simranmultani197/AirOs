"""
Storage layer for AgentCircuit.

Provides:
- In-memory storage (default, zero dependencies)
- SQLite storage (local-first, persistent)
- PostgreSQL support (optional, for teams)
- Automatic indexing
- Data pruning and archival
- Query optimization
"""
import sqlite3
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

DB_PATH = ".agentcircuit/traces.db"


class StorageBackend(Enum):
    """Available storage backends."""
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


@dataclass
class StorageConfig:
    """Configuration for storage."""
    backend: StorageBackend = StorageBackend.SQLITE
    db_path: str = DB_PATH
    connection_string: Optional[str] = None

    # Pruning settings
    enable_pruning: bool = True
    retention_days: int = 30
    max_traces: int = 100000

    # Performance settings
    enable_wal: bool = True
    batch_size: int = 100

    # PostgreSQL settings
    pool_size: int = 5


class BaseStorage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def log_trace(
        self,
        run_id: str,
        node_id: str,
        input_state: Any,
        output_state: Any,
        status: str,
        cost_tokens: int = 0,
        recovery_attempts: int = 0,
        saved_cost: float = 0.0,
        token_usage: int = 0,
        estimated_cost: float = 0.0,
        diagnosis: Optional[str] = None,
        duration_ms: float = 0.0,
        error_category: Optional[str] = None,
        strategy_used: Optional[str] = None
    ) -> int:
        """Log a trace and return its ID."""
        pass

    @abstractmethod
    def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all traces for a run."""
        pass

    @abstractmethod
    def get_run_cost(self, run_id: str) -> float:
        """Get total cost for a run."""
        pass

    @abstractmethod
    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value."""
        pass

    @abstractmethod
    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        pass

    @abstractmethod
    def prune_old_traces(self, days: int) -> int:
        """Delete traces older than N days. Return count deleted."""
        pass


class InMemoryStorage(BaseStorage):
    """
    In-memory storage backend. Zero dependencies, zero setup.

    This is the default storage backend for the SDK. Traces are stored
    in memory and lost when the process exits. Use SQLite or PostgreSQL
    for persistent storage.
    """

    def __init__(self):
        self._traces: List[Dict[str, Any]] = []
        self._settings: Dict[str, str] = {
            "cost_per_token": "0.000005",
            "retention_days": "30",
        }
        self._next_id = 1

    def log_trace(
        self,
        run_id: str,
        node_id: str,
        input_state: Any,
        output_state: Any,
        status: str,
        cost_tokens: int = 0,
        recovery_attempts: int = 0,
        saved_cost: float = 0.0,
        token_usage: int = 0,
        estimated_cost: float = 0.0,
        diagnosis: Optional[str] = None,
        duration_ms: float = 0.0,
        error_category: Optional[str] = None,
        strategy_used: Optional[str] = None
    ) -> int:
        trace_id = self._next_id
        self._next_id += 1
        self._traces.append({
            "id": trace_id,
            "run_id": run_id,
            "node_id": node_id,
            "input_state": input_state,
            "output_state": output_state,
            "status": status,
            "cost_tokens": cost_tokens,
            "recovery_attempts": recovery_attempts,
            "saved_cost": saved_cost,
            "token_usage": token_usage,
            "estimated_cost": estimated_cost,
            "diagnosis": diagnosis,
            "duration_ms": duration_ms,
            "error_category": error_category,
            "strategy_used": strategy_used,
            "timestamp": datetime.now().isoformat(),
        })
        return trace_id

    def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        return [t for t in self._traces if t["run_id"] == run_id]

    def get_run_cost(self, run_id: str) -> float:
        return sum(t["estimated_cost"] for t in self._traces if t["run_id"] == run_id)

    def get_setting(self, key: str) -> Optional[str]:
        return self._settings.get(key)

    def set_setting(self, key: str, value: str) -> None:
        self._settings[key] = value

    def prune_old_traces(self, days: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        before = len(self._traces)
        self._traces = [
            t for t in self._traces
            if datetime.fromisoformat(t["timestamp"]) >= cutoff
        ]
        return before - len(self._traces)


class Storage(BaseStorage):
    """
    SQLite storage backend with enhanced features.

    Features:
    - Automatic table creation and migrations
    - Indexed queries for performance
    - WAL mode for concurrency
    - Automatic pruning of old data
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        config: Optional[StorageConfig] = None
    ):
        self.db_path = db_path
        self.config = config or StorageConfig(db_path=db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable WAL mode for better concurrency
        if self.config.enable_wal:
            cursor.execute("PRAGMA journal_mode=WAL;")

        # Create traces table with proper types
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                input_state TEXT,
                output_state TEXT,
                status TEXT NOT NULL,
                cost_tokens INTEGER DEFAULT 0,
                recovery_attempts INTEGER DEFAULT 0,
                saved_cost REAL DEFAULT 0.0,
                token_usage INTEGER DEFAULT 0,
                estimated_cost REAL DEFAULT 0.0,
                diagnosis TEXT,
                duration_ms REAL DEFAULT 0.0,
                error_category TEXT,
                strategy_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_traces_run_id
            ON traces(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_traces_status
            ON traces(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_traces_timestamp
            ON traces(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_traces_node_id
            ON traces(node_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_traces_run_node
            ON traces(run_id, node_id)
        """)

        # Run migrations for existing databases
        self._run_migrations(cursor)

        # Global Settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Default settings
        cursor.execute("""
            INSERT OR IGNORE INTO global_settings (key, value)
            VALUES ('cost_per_token', '0.000005')
        """)
        cursor.execute("""
            INSERT OR IGNORE INTO global_settings (key, value)
            VALUES ('retention_days', '30')
        """)

        conn.commit()
        conn.close()

    def _run_migrations(self, cursor: sqlite3.Cursor):
        """Run database migrations for backward compatibility."""
        cursor.execute("PRAGMA table_info(traces)")
        columns = {info[1] for info in cursor.fetchall()}

        migrations = [
            ("saved_cost", "ALTER TABLE traces ADD COLUMN saved_cost REAL DEFAULT 0.0"),
            ("token_usage", "ALTER TABLE traces ADD COLUMN token_usage INTEGER DEFAULT 0"),
            ("estimated_cost", "ALTER TABLE traces ADD COLUMN estimated_cost REAL DEFAULT 0.0"),
            ("diagnosis", "ALTER TABLE traces ADD COLUMN diagnosis TEXT"),
            ("duration_ms", "ALTER TABLE traces ADD COLUMN duration_ms REAL DEFAULT 0.0"),
            ("error_category", "ALTER TABLE traces ADD COLUMN error_category TEXT"),
            ("strategy_used", "ALTER TABLE traces ADD COLUMN strategy_used TEXT"),
        ]

        for column_name, migration_sql in migrations:
            if column_name not in columns:
                try:
                    cursor.execute(migration_sql)
                except sqlite3.OperationalError:
                    pass  # Column might already exist

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log_trace(
        self,
        run_id: str,
        node_id: str,
        input_state: Any,
        output_state: Any,
        status: str,
        cost_tokens: int = 0,
        recovery_attempts: int = 0,
        saved_cost: float = 0.0,
        token_usage: int = 0,
        estimated_cost: float = 0.0,
        diagnosis: Optional[str] = None,
        duration_ms: float = 0.0,
        error_category: Optional[str] = None,
        strategy_used: Optional[str] = None
    ) -> int:
        """Log a node execution trace."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                input_json = json.dumps(input_state, default=str)
                output_json = json.dumps(output_state, default=str)

                cursor.execute("""
                    INSERT INTO traces (
                        run_id, node_id, input_state, output_state, status,
                        cost_tokens, recovery_attempts, saved_cost, token_usage,
                        estimated_cost, diagnosis, duration_ms, error_category, strategy_used
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, node_id, input_json, output_json, status,
                    cost_tokens, recovery_attempts, saved_cost, token_usage,
                    estimated_cost, diagnosis, duration_ms, error_category, strategy_used
                ))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                print(f"Error logging trace: {e}")
                return -1

    def get_run_cost(self, run_id: str) -> float:
        """Calculate total estimated cost for a run so far."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT SUM(estimated_cost) FROM traces WHERE run_id = ?",
                (run_id,)
            )
            result = cursor.fetchone()[0]
            return result or 0.0

    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM global_settings WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO global_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """, (key, value))
            conn.commit()

    def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all traces for a specific run_id, sorted by time."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM traces WHERE run_id = ? ORDER BY id ASC
            """, (run_id,))

            rows = cursor.fetchall()
            history = []

            for row in rows:
                history.append({
                    "id": row["id"],
                    "node_id": row["node_id"],
                    "input_state": json.loads(row["input_state"]) if row["input_state"] else {},
                    "output_state": json.loads(row["output_state"]) if row["output_state"] else {},
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "diagnosis": row["diagnosis"],
                    "error_category": row["error_category"] if "error_category" in row.keys() else None,
                    "strategy_used": row["strategy_used"] if "strategy_used" in row.keys() else None,
                    "duration_ms": row["duration_ms"] if "duration_ms" in row.keys() else 0,
                    "recovery_attempts": row["recovery_attempts"],
                })

            return history

    def prune_old_traces(self, days: int = 30) -> int:
        """Delete traces older than N days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now() - timedelta(days=days)
            cursor.execute(
                "DELETE FROM traces WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def get_traces(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        node_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get traces with filtering and pagination.

        Args:
            limit: Maximum number of traces to return
            offset: Number of traces to skip
            status: Filter by status (success, failed, repaired, etc.)
            node_id: Filter by node ID
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of trace dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM traces WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)

            if node_id:
                query += " AND node_id = ?"
                params.append(node_id)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total counts
            cursor.execute("SELECT COUNT(*) FROM traces")
            stats["total_traces"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT run_id) FROM traces")
            stats["total_runs"] = cursor.fetchone()[0]

            # Status breakdown
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM traces
                GROUP BY status
            """)
            stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Cost stats
            cursor.execute("SELECT SUM(estimated_cost), SUM(saved_cost) FROM traces")
            row = cursor.fetchone()
            stats["total_cost"] = row[0] or 0
            stats["total_saved"] = row[1] or 0

            # Average recovery time
            cursor.execute("""
                SELECT AVG(duration_ms)
                FROM traces
                WHERE status = 'repaired' AND duration_ms > 0
            """)
            stats["avg_repair_time_ms"] = cursor.fetchone()[0] or 0

            # Error category breakdown
            cursor.execute("""
                SELECT error_category, COUNT(*) as count
                FROM traces
                WHERE error_category IS NOT NULL
                GROUP BY error_category
            """)
            stats["by_error_category"] = {
                row["error_category"]: row["count"]
                for row in cursor.fetchall()
            }

            return stats

    def get_node_stats(self, node_id: str) -> Dict[str, Any]:
        """Get statistics for a specific node."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                    SUM(CASE WHEN status = 'repaired' THEN 1 ELSE 0 END) as repaired,
                    SUM(CASE WHEN status LIKE 'failed%' THEN 1 ELSE 0 END) as failed,
                    AVG(duration_ms) as avg_duration,
                    SUM(estimated_cost) as total_cost
                FROM traces
                WHERE node_id = ?
            """, (node_id,))

            row = cursor.fetchone()
            return {
                "node_id": node_id,
                "total": row["total"],
                "success": row["success"],
                "repaired": row["repaired"],
                "failed": row["failed"],
                "success_rate": (row["success"] + row["repaired"]) / row["total"] if row["total"] > 0 else 0,
                "avg_duration_ms": row["avg_duration"] or 0,
                "total_cost": row["total_cost"] or 0
            }

    def vacuum(self) -> None:
        """Optimize the database."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")


class PostgresStorage(BaseStorage):
    """
    PostgreSQL storage backend for team/production use.

    Requires: pip install psycopg2-binary

    Features:
    - Connection pooling
    - Full-text search
    - Better concurrency
    - JSONB for efficient state storage
    """

    def __init__(self, connection_string: str, config: Optional[StorageConfig] = None):
        self.connection_string = connection_string
        self.config = config or StorageConfig(
            backend=StorageBackend.POSTGRESQL,
            connection_string=connection_string
        )
        self._pool = None
        self._init_db()

    def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import psycopg2
                from psycopg2 import pool
            except ImportError:
                raise ImportError(
                    "psycopg2 is required for PostgreSQL support. "
                    "Run: pip install psycopg2-binary"
                )

            self._pool = pool.ThreadedConnectionPool(
                1, self.config.pool_size,
                self.connection_string
            )
        return self._pool

    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            yield conn
        finally:
            pool.putconn(conn)

    def _init_db(self):
        """Initialize PostgreSQL tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    input_state JSONB,
                    output_state JSONB,
                    status TEXT NOT NULL,
                    cost_tokens INTEGER DEFAULT 0,
                    recovery_attempts INTEGER DEFAULT 0,
                    saved_cost REAL DEFAULT 0.0,
                    token_usage INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    diagnosis TEXT,
                    duration_ms REAL DEFAULT 0.0,
                    error_category TEXT,
                    strategy_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_run_id ON traces(run_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_node_id ON traces(node_id)
            """)

            # Settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Default settings
            cursor.execute("""
                INSERT INTO global_settings (key, value)
                VALUES ('cost_per_token', '0.000005')
                ON CONFLICT (key) DO NOTHING
            """)

            conn.commit()

    def log_trace(
        self,
        run_id: str,
        node_id: str,
        input_state: Any,
        output_state: Any,
        status: str,
        cost_tokens: int = 0,
        recovery_attempts: int = 0,
        saved_cost: float = 0.0,
        token_usage: int = 0,
        estimated_cost: float = 0.0,
        diagnosis: Optional[str] = None,
        duration_ms: float = 0.0,
        error_category: Optional[str] = None,
        strategy_used: Optional[str] = None
    ) -> int:
        """Log a trace to PostgreSQL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO traces (
                    run_id, node_id, input_state, output_state, status,
                    cost_tokens, recovery_attempts, saved_cost, token_usage,
                    estimated_cost, diagnosis, duration_ms, error_category, strategy_used
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                run_id, node_id,
                json.dumps(input_state, default=str),
                json.dumps(output_state, default=str),
                status, cost_tokens, recovery_attempts, saved_cost,
                token_usage, estimated_cost, diagnosis, duration_ms,
                error_category, strategy_used
            ))
            trace_id = cursor.fetchone()[0]
            conn.commit()
            return trace_id

    def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        """Get traces for a run from PostgreSQL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM traces WHERE run_id = %s ORDER BY id ASC
            """, (run_id,))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            history = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # JSONB columns are already parsed
                history.append({
                    "id": row_dict["id"],
                    "node_id": row_dict["node_id"],
                    "input_state": row_dict["input_state"] or {},
                    "output_state": row_dict["output_state"] or {},
                    "status": row_dict["status"],
                    "timestamp": str(row_dict["timestamp"]),
                    "diagnosis": row_dict["diagnosis"],
                    "error_category": row_dict.get("error_category"),
                    "strategy_used": row_dict.get("strategy_used"),
                    "duration_ms": row_dict.get("duration_ms", 0),
                    "recovery_attempts": row_dict["recovery_attempts"],
                })

            return history

    def get_run_cost(self, run_id: str) -> float:
        """Get total cost for a run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT SUM(estimated_cost) FROM traces WHERE run_id = %s",
                (run_id,)
            )
            result = cursor.fetchone()[0]
            return result or 0.0

    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM global_settings WHERE key = %s",
                (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO global_settings (key, value, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
            """, (key, value))
            conn.commit()

    def prune_old_traces(self, days: int = 30) -> int:
        """Delete traces older than N days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM traces WHERE timestamp < NOW() - INTERVAL '%s days'",
                (days,)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted


def create_storage(
    backend: Union[str, StorageBackend] = StorageBackend.MEMORY,
    **kwargs
) -> BaseStorage:
    """
    Factory function to create a storage backend.

    Args:
        backend: Storage backend type ("memory", "sqlite", or "postgresql")
        **kwargs: Backend-specific configuration

    Returns:
        Configured storage instance
    """
    if isinstance(backend, str):
        backend = StorageBackend(backend.lower())

    if backend == StorageBackend.MEMORY:
        return InMemoryStorage()
    elif backend == StorageBackend.SQLITE:
        return Storage(
            db_path=kwargs.get("db_path", DB_PATH),
            config=kwargs.get("config")
        )
    elif backend == StorageBackend.POSTGRESQL:
        connection_string = kwargs.get("connection_string")
        if not connection_string:
            raise ValueError("PostgreSQL requires a connection_string")
        return PostgresStorage(
            connection_string=connection_string,
            config=kwargs.get("config")
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend}")


# Module-level default storage instance (lazy singleton)
_default_storage: Optional[BaseStorage] = None


def get_default_storage() -> BaseStorage:
    """Get the default storage instance (in-memory by default)."""
    global _default_storage
    if _default_storage is None:
        _default_storage = InMemoryStorage()
    return _default_storage


def set_default_storage(storage: BaseStorage) -> None:
    """Set the default storage instance (e.g. to switch to SQLite)."""
    global _default_storage
    _default_storage = storage
