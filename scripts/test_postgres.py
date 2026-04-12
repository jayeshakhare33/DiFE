#!/usr/bin/env python3
"""
Test PostgreSQL Connection
"""
import psycopg2
import sys
import os

def test_postgres():
    """Test PostgreSQL connection and basic operations"""
    try:
        # Get connection parameters from environment or use defaults
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        database = os.getenv('POSTGRES_DB', 'fraud_detection')
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', 'postgres123')
        
        print(f"Connecting to PostgreSQL at {host}:{port}...")
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        print("✅ PostgreSQL connection successful!")
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL version: {version[0][:50]}...")
        
        # Check if tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print(f"✅ Found {len(tables)} tables:")
        for table in tables:
            print(f"   - {table[0]}")
        
        # Count records
        if tables:
            cursor.execute("SELECT COUNT(*) FROM transactions;")
            count = cursor.fetchone()[0]
            print(f"✅ Transactions count: {count}")
        
        cursor.close()
        conn.close()
        print("\n✅ All PostgreSQL tests passed!")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is running")
        print("2. Check connection parameters")
        print("3. Verify database 'fraud_detection' exists")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = test_postgres()
    sys.exit(0 if success else 1)

