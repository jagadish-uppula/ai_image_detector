import sqlite3
from django.conf import settings
import os
from contextlib import contextmanager
import threading

# Thread-local storage for database connections
_local = threading.local()

@contextmanager
def get_db():
    """
    Get a database connection with context manager for proper cleanup
    """
    if not hasattr(_local, "conn"):
        db_path = os.path.join(settings.BASE_DIR, 'db.sqlite3')
        _local.conn = sqlite3.connect(db_path)
        _local.conn.row_factory = sqlite3.Row
    
    try:
        yield _local.conn
    except Exception as e:
        print(f"Database error: {e}")
        raise
    finally:
        # Don't close connection to maintain thread-local connection
        pass

def close_db(e=None):
    """
    Close the database connection for the current thread
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        del _local.conn

def init_db():
    """
    Initialize the database with required tables and indexes
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Create users table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS auth_user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            password TEXT NOT NULL,
            last_login TIMESTAMP,
            is_superuser BOOLEAN NOT NULL,
            username TEXT NOT NULL UNIQUE,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            is_staff BOOLEAN NOT NULL,
            is_active BOOLEAN NOT NULL,
            date_joined TIMESTAMP NOT NULL,
            face_embedding BLOB,
            is_admin BOOLEAN DEFAULT FALSE
        )
        """)
        
        # Create analysis_history table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detector_analysishistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT NOT NULL,
            is_ai_generated BOOLEAN NOT NULL,
            confidence REAL NOT NULL,
            similarity_score REAL,
            face_embedding BLOB,
            created_at TIMESTAMP NOT NULL,
            user_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES auth_user (id)
        )
        """)
        
        # Create indexes
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysishistory_user 
        ON detector_analysishistory(user_id)
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysishistory_created 
        ON detector_analysishistory(created_at)
        """)
        
        conn.commit()