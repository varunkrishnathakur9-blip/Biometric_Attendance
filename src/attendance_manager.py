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
            # Add session and duration columns if not present
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session TEXT,
                    duration REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session, id)
                )
                """
            )
            # Try to add columns if missing (for upgrades)
            try:
                self.connection.execute("ALTER TABLE attendance ADD COLUMN session TEXT")
            except Exception:
                pass
            try:
                self.connection.execute("ALTER TABLE attendance ADD COLUMN duration REAL")
            except Exception:
                pass
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
        session_name: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Mark attendance for a student for a session (session_name as string, or fallback to date).

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
                    INSERT OR REPLACE INTO attendance(name, date, time, status, session, duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (student_name, date_str, time_str, status, session_name, duration),
                )
                def generate_report(self, report_path: Path) -> None:
                    # Export all attendance records with new columns
                    df = pd.read_sql_query(
                        "SELECT name, date, time, status, session, duration FROM attendance ORDER BY date DESC, time DESC",
                        self.connection,
                    )
                    df.rename(columns={"date": "Date", "time": "Time", "name": "Name", "status": "Status", "session": "Session", "duration": "Duration"}, inplace=True)
                    df.to_csv(report_path, index=False)
                    self.logger.info(f"Attendance report generated at {report_path}")
            message = f"Attendance marked for {student_name} ({status}) in session '{date_str}'"
            self.logger.info(message)
            return True, message
        except sqlite3.IntegrityError:
            message = f"Attendance already marked for {student_name} in session '{date_str}'"
            self.logger.info(message)
            return False, message
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.exception("Failed to mark attendance: %s", exc)
            return False, f"Failed to mark attendance for {student_name}"

    def get_all_students(self) -> list:
        cur = self.connection.cursor()
        cur.execute("SELECT name FROM students")
        # Return a list of names (str), not sqlite3.Row objects
        return [row[0] for row in cur.fetchall()]

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
        # Export all attendance records with new columns
        query = "SELECT name, date, time, status, session, duration FROM attendance"
        params = ()
        if date:
            query += " WHERE date = ?"
            params = (date,)
        query += " ORDER BY date DESC, time DESC"

        df = pd.read_sql_query(query, self.connection, params=params)
        df.rename(columns={
            "name": "Name",
            "date": "Date",
            "time": "Time",
            "status": "Status",
            "session": "Session",
            "duration": "Duration"
        }, inplace=True)

        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        self.logger.info(f"Attendance report generated at {output_path}")
        return output_path

    def close(self) -> None:
        self.connection.close()
        self.logger.info("Database connection closed")

    def __enter__(self) -> "AttendanceManager":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()
