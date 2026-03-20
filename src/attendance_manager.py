from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class AttendanceManager:
    """Manages attendance persistence, duplicate checks, and reporting."""

    def __init__(self, db_path: Path) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize_database()

    def _initialize_database(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, date)
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_attendance_date
                ON attendance(date)
                """
            )

        self.logger.info("Database initialized at %s", self.db_path)

    def mark_attendance(
        self,
        student_name: str,
        status: str = "Present",
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """Mark attendance for a student once per day.

        Returns:
            Tuple[bool, str]: (was_marked, message)
        """
        if not student_name or student_name == "Unknown":
            return False, "Unknown face not recorded"

        ts = timestamp or datetime.now()
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M:%S")

        try:
            with self.connection:
                self.connection.execute(
                    "INSERT OR IGNORE INTO students(name) VALUES (?)",
                    (student_name,),
                )
                self.connection.execute(
                    """
                    INSERT INTO attendance(name, date, time, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (student_name, date_str, time_str, status),
                )
            message = f"Attendance marked for {student_name}"
            self.logger.info(message)
            return True, message
        except sqlite3.IntegrityError:
            message = f"Attendance already marked today for {student_name}"
            self.logger.info(message)
            return False, message
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.exception("Failed to mark attendance: %s", exc)
            return False, f"Failed to mark attendance for {student_name}"

    def fetch_attendance(self, date: Optional[str] = None) -> List[sqlite3.Row]:
        query = "SELECT name, date, time, status FROM attendance"
        params = ()
        if date:
            query += " WHERE date = ?"
            params = (date,)
        query += " ORDER BY date, time"

        cursor = self.connection.execute(query, params)
        return cursor.fetchall()

    def generate_report(self, output_csv_path: Path, date: Optional[str] = None) -> Path:
        rows = self.fetch_attendance(date=date)
        data = [dict(row) for row in rows]

        df = pd.DataFrame(data, columns=["name", "date", "time", "status"])
        df.rename(
            columns={"name": "Name", "date": "Date", "time": "Time", "status": "Status"},
            inplace=True,
        )

        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        self.logger.info("Attendance report generated at %s", output_path)
        return output_path

    def close(self) -> None:
        self.connection.close()
        self.logger.info("Database connection closed")

    def __enter__(self) -> "AttendanceManager":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()
