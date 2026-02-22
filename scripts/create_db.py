
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database():
    # Connect to 'postgres' database to create new db
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'note_agent'")
        exists = cur.fetchone()
        
        if not exists:
            print("Creating database 'note_agent'...")
            cur.execute("CREATE DATABASE note_agent")
            print("✅ Database created.")
        else:
            print("ℹ️ Database 'note_agent' already exists.")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Failed to create database: {e}")

if __name__ == "__main__":
    create_database()
