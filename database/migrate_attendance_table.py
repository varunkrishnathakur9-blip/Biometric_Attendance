import sqlite3

DB_PATH = "c:/repos/Biometric_Attendance/database/attendance.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Backup old table
def backup_table():
    cursor.execute("DROP TABLE IF EXISTS attendance_backup")
    cursor.execute("CREATE TABLE attendance_backup AS SELECT * FROM attendance")
    print("Backup of attendance table created as attendance_backup.")

def drop_and_recreate_attendance():
    cursor.execute("DROP TABLE IF EXISTS attendance")
    cursor.execute('''
        CREATE TABLE attendance (
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
    ''')
    print("attendance table dropped and recreated with UNIQUE(session, id).")

if __name__ == "__main__":
    backup_table()
    drop_and_recreate_attendance()
    conn.commit()
    conn.close()
    print("Done.")
