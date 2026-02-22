
import os
import sys
import psycopg2
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DB_CONFIG = "dbname=note_agent user=postgres password=postgres host=localhost"

def connect():
    try:
        return psycopg2.connect(DB_CONFIG)
    except psycopg2.OperationalError:
        print("❌ Could not connect to Database.")
        print("   Please ensure PostgreSQL is running on localhost:5432")
        print("   and the database 'note_agent' exists.")
        sys.exit(1)

def check_data():
    conn = connect()
    cur = conn.cursor()

    print("\n🔍 Checking Database Contents...\n")

    # 1. Check Notes
    cur.execute("SELECT count(*) FROM notes;")
    note_count = cur.fetchone()[0]
    print(f"📄 Notes: {note_count}")

    # 2. Check Spans
    cur.execute("SELECT count(*), avg(token_count) FROM spans;")
    span_stats = cur.fetchone()
    print(f"🧩 Spans: {span_stats[0]} (Avg Tokens: {float(span_stats[1] or 0):.1f})")

    # 3. Check Objects by Type
    cur.execute("SELECT type, count(*) FROM objects GROUP BY type;")
    print("📦 Objects:")
    for type_, count in cur.fetchall():
        print(f"   - {type_}: {count}")

    # 4. Check Links
    cur.execute("SELECT type, count(*) FROM links GROUP BY type;")
    print("🔗 Links:")
    for type_, count in cur.fetchall():
        print(f"   - {type_}: {count}")

    # 5. Check Insights
    cur.execute("SELECT type, severity, payload FROM insights;")
    print("💡 Insights:")
    for row in cur.fetchall():
        payload = row[2]
        explanation = payload.get('explanation', 'No explanation') if payload else 'No payload'
        print(f"   - [{row[1].upper()}] {row[0]}: \"{explanation}\"")

    cur.close()
    conn.close()
    print("\n✅ Data verification complete.")

if __name__ == "__main__":
    check_data()
