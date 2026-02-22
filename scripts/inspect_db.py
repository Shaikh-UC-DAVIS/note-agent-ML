
import os
import sys
import psycopg2
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DB_CONFIG = "dbname=note_agent user=postgres password=postgres host=localhost"

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def connect():
    try:
        return psycopg2.connect(DB_CONFIG)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

def list_tables(cur):
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    return [row[0] for row in cur.fetchall()]

def inspect_table(table_name, limit=10):
    conn = connect()
    cur = conn.cursor()

    # Verify table exists to prevent SQL injection
    tables = list_tables(cur)
    if table_name not in tables:
        print(f"❌ Table '{table_name}' does not exist.")
        print(f"Available tables: {', '.join(tables)}")
        return

    print(f"\n🔍 Inspecting table: {table_name} (Limit {limit})\n")
    
    # Get values
    cur.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
    rows = cur.fetchall()
    
    # Get column names
    colnames = [desc[0] for desc in cur.description]
    
    if not rows:
        print("   (Table is empty)")
    else:
        # Print JSON-like structure for readability
        for i, row in enumerate(rows):
            row_dict = dict(zip(colnames, row))
            print(f"--- Row {i+1} ---")
            print(json.dumps(row_dict, indent=2, cls=DateTimeEncoder))
            print("")

    cur.close()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Inspect Database Tables")
    parser.add_argument("table", nargs="?", help="Name of the table to inspect")
    parser.add_argument("--list", action="store_true", help="List all tables")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to show")
    
    args = parser.parse_args()
    
    conn = connect()
    cur = conn.cursor()
    tables = list_tables(cur)
    cur.close()
    conn.close()

    if args.list or not args.table:
        print("📂 Available Tables:")
        for t in tables:
            print(f" - {t}")
        if not args.list:
            print("\nUsage: python scripts/inspect_db.py <table_name>")
    
    if args.table:
        inspect_table(args.table, args.limit)

if __name__ == "__main__":
    main()
