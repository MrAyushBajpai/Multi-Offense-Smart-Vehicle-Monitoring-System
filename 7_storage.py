"""
storage.py — Pluggable storage module

Provides two backends that share a common interface:
  - JSONFileStorage  (default) — reads/writes a pretty-printed JSON file
  - DatabaseStorage            — stores records in a SQLite database

Both backends expose:
  save(record: dict)          → None
  get_all()                   → list[dict]
  delete(record_id: int)      → bool
  clear()                     → None
"""

import sqlite3
import json
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base — defines the shared interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseStorage(ABC):
    """Common interface every storage backend must implement."""

    @abstractmethod
    def save(self, record: dict) -> None:
        """Persist a single record (dict)."""

    @abstractmethod
    def get_all(self) -> list[dict]:
        """Return every stored record as a list of dicts."""

    @abstractmethod
    def delete(self, record_id: int) -> bool:
        """Delete the record with the given id. Returns True if found."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all records."""


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1 — JSON file (default)
# ─────────────────────────────────────────────────────────────────────────────

class JSONFileStorage(BaseStorage):
    """
    Stores records as a pretty-printed JSON array in a .json file.

    File structure:
        [
          {
            "id": 1,
            "timestamp": "2026-05-07 12:00:00",
            "data": { ... }
          },
          ...
        ]
    """

    def __init__(self, filepath: str = "record.json") -> None:
        self.filepath = Path(filepath)
        self._ensure_file()

    # ── private ──────────────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self._write([])

    def _read(self) -> list[dict]:
        text = self.filepath.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return json.loads(text)

    def _write(self, records: list[dict]) -> None:
        self.filepath.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _next_id(self, records: list[dict]) -> int:
        if not records:
            return 1
        return max(r.get("id", 0) for r in records) + 1

    # ── public ───────────────────────────────────────────────────────────────

    def save(self, record: dict) -> None:
        """Append a new record and rewrite the JSON file."""
        records = self._read()
        records.append({
            "id":        self._next_id(records),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data":      record,
        })
        self._write(records)

    def get_all(self) -> list[dict]:
        """Return all records from the JSON file."""
        return self._read()

    def delete(self, record_id: int) -> bool:
        """Remove the record with the given id and rewrite the file."""
        records  = self._read()
        filtered = [r for r in records if r.get("id") != record_id]
        if len(filtered) == len(records):
            return False
        self._write(filtered)
        return True

    def clear(self) -> None:
        """Reset the JSON file to an empty array."""
        self._write([])


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2 — SQLite database
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseStorage(BaseStorage):
    """
    Stores records in a SQLite database.

    Schema:
        records(id INTEGER PRIMARY KEY, timestamp TEXT, data TEXT)
        'data' column holds a JSON-serialised dict.
    """

    def __init__(self, db_path: str = "record.db") -> None:
        self.db_path = db_path
        self._init_db()

    # ── private ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT    NOT NULL,
                    data      TEXT    NOT NULL
                )
            """)

    # ── public ───────────────────────────────────────────────────────────────

    def save(self, record: dict) -> None:
        """Insert a new record into the database."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO records (timestamp, data) VALUES (?, ?)",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    json.dumps(record),
                ),
            )

    def get_all(self) -> list[dict]:
        """Return every row as a dict with id, timestamp, and data."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, data FROM records ORDER BY id"
            ).fetchall()
        return [
            {
                "id":        row["id"],
                "timestamp": row["timestamp"],
                "data":      json.loads(row["data"]),
            }
            for row in rows
        ]

    def delete(self, record_id: int) -> bool:
        """Delete the row with the given id."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM records WHERE id = ?", (record_id,)
            )
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all rows from the records table."""
        with self._connect() as conn:
            conn.execute("DELETE FROM records")


# ─────────────────────────────────────────────────────────────────────────────
# Factory — convenience function used by callers
# ─────────────────────────────────────────────────────────────────────────────

def get_storage(backend: str = "json", **kwargs) -> BaseStorage:
    """
    Return a storage instance for the requested backend.

    Parameters
    ----------
    backend : "json" (default) | "db"
    **kwargs : forwarded to the chosen storage class constructor
               e.g. filepath="detections.json"  or  db_path="app.db"

    Examples
    --------
    store = get_storage()                            # JSON file -> record.json
    store = get_storage("json", filepath="x.json")  # JSON file -> x.json
    store = get_storage("db")                        # SQLite    -> record.db
    store = get_storage("db", db_path="app.db")      # SQLite    -> app.db
    """
    if backend == "json":
        return JSONFileStorage(**kwargs)
    elif backend == "db":
        return DatabaseStorage(**kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'json' or 'db'.")


# ─────────────────────────────────────────────────────────────────────────────
# Run directly -> write examples to record.json
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    OUTPUT_FILE = "record.json"

    print(f"Running storage.py directly — writing examples to '{OUTPUT_FILE}'\n")

    store = JSONFileStorage(filepath=OUTPUT_FILE)

    # Start fresh so the demo is repeatable
    store.clear()

    # 1. Save some example vehicle detection records
    examples = [
        {"video": "highway_cam.mp4",    "frame": 10,  "vehicle": "Car",        "confidence": 0.92},
        {"video": "highway_cam.mp4",    "frame": 10,  "vehicle": "Truck",      "confidence": 0.88},
        {"video": "intersection_n.mp4", "frame": 55,  "vehicle": "Motorcycle", "confidence": 0.75},
        {"video": "intersection_n.mp4", "frame": 120, "vehicle": "Bus",        "confidence": 0.95},
        {"video": "parking_lot_b.mp4",  "frame": 5,   "vehicle": "Car",        "confidence": 0.81},
    ]

    print("Saving example records...")
    for ex in examples:
        store.save(ex)
        print(
            f"  + {ex['vehicle']:12s} | conf={ex['confidence']:.2f} "
            f"| {ex['video']} frame {ex['frame']}"
        )

    # 2. Read back and display
    print(f"\nAll records in '{OUTPUT_FILE}':")
    for r in store.get_all():
        d = r["data"]
        print(
            f"  ID {r['id']:>2}  [{r['timestamp']}]  "
            f"{d['vehicle']:12s} conf={d['confidence']:.2f}  "
            f"{d['video']}  frame {d['frame']}"
        )

    # 3. Delete one record
    print("\nDeleting record ID 3...")
    store.delete(3)
    print(f"Records remaining: {len(store.get_all())}")

    print(f"\nDone. Open '{OUTPUT_FILE}' to inspect the formatted JSON.")