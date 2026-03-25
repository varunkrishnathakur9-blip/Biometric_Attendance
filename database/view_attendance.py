import sqlite3

# Path to your database file
DB_PATH = r"C:\\repos\\Biometric_Attendance\\database\\attendance.db"

def print_attendance():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM attendance")
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        print(column_names)
        for row in rows:
            print(row)
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print_attendance()
