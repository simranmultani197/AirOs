import sqlite3
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

DB_PATH = ".air_os/traces.db"

class Storage:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                node_id TEXT,
                input_state TEXT,
                output_state TEXT,
                status TEXT,
                cost_tokens INTEGER,
                recovery_attempts INTEGER,
                saved_cost REAL DEFAULT 0.0,
                token_usage INTEGER DEFAULT 0,
                estimated_cost REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Migration for traces
        cursor.execute("PRAGMA table_info(traces)")
        columns = [info[1] for info in cursor.fetchall()]
        if "saved_cost" not in columns:
            cursor.execute("ALTER TABLE traces ADD COLUMN saved_cost REAL DEFAULT 0.0")
        if "token_usage" not in columns:
            cursor.execute("ALTER TABLE traces ADD COLUMN token_usage INTEGER DEFAULT 0")
        if "estimated_cost" not in columns:
            cursor.execute("ALTER TABLE traces ADD COLUMN estimated_cost REAL DEFAULT 0.0")
        if "diagnosis" not in columns:
            cursor.execute("ALTER TABLE traces ADD COLUMN diagnosis TEXT")

        # Global Settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Default pricing (approx GPT-4o blend $5/1M tokens -> 0.000005 per token)
        # Storing as standard USD cost per token
        cursor.execute("INSERT OR IGNORE INTO global_settings (key, value) VALUES ('cost_per_token', '0.000005')")

        conn.commit()
        conn.close()

    def log_trace(self, 
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
                  diagnosis: Optional[str] = None):
        """Log a node execution trace."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            input_json = json.dumps(input_state, default=str)
            output_json = json.dumps(output_state, default=str)
            
            cursor.execute("""
                INSERT INTO traces (run_id, node_id, input_state, output_state, status, cost_tokens, recovery_attempts, saved_cost, token_usage, estimated_cost, diagnosis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, node_id, input_json, output_json, status, cost_tokens, recovery_attempts, saved_cost, token_usage, estimated_cost, diagnosis))
            conn.commit()
        except Exception as e:
            print(f"Error logging trace: {e}")
        finally:
            conn.close()

    def get_run_cost(self, run_id: str) -> float:
        """Calculate total estimated cost for a run so far."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(estimated_cost) FROM traces WHERE run_id = ?", (run_id,))
        result = cursor.fetchone()[0]
        conn.close()
        return result or 0.0

    def get_setting(self, key: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM global_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def set_setting(self, key: str, value: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO global_settings (key, value) 
            VALUES (?, ?) 
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        conn.commit()
        conn.close()

    def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all traces for a specific run_id, sorted by time."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM traces WHERE run_id = ? ORDER BY id ASC
        """, (run_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            # Check if diagnosis column exists in row (it might be None for old rows)
            # Row factory allows access by name.
            # If migration ran, the column exists but might be NULL.
            diagnosis_val = row["diagnosis"] if "diagnosis" in row.keys() else None
            
            history.append({
                "id": row["id"],
                "node_id": row["node_id"],
                "input_state": json.loads(row["input_state"]),
                "output_state": json.loads(row["output_state"]),
                "status": row["status"],
                "timestamp": row["timestamp"],
                "diagnosis": diagnosis_val
            })
        return history
