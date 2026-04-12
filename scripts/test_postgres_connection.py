#!/usr/bin/env python3
"""
Test PostgreSQL Connection and Get Credentials
"""
import sys
import os
import psycopg2
import yaml
from pathlib import Path

def test_connection(host, port, database, user, password):
    """Test PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return True, version
    except psycopg2.OperationalError as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("PostgreSQL Connection Tester")
    print("=" * 60)
    print()
    
    # Try Docker PostgreSQL first
    print("Testing Docker PostgreSQL (default credentials)...")
    success, result = test_connection('localhost', '5432', 'fraud_detection', 'postgres', 'postgres123')
    if success:
        print("✅ Docker PostgreSQL connection successful!")
        print(f"   Version: {result[:50]}...")
        print()
        print("You can use:")
        print("  python scripts/sync_postgres_to_neo4j.py")
        return 0
    else:
        print(f"❌ Docker PostgreSQL failed: {result}")
        print()
    
    # Try config.yaml
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        print("Testing PostgreSQL from config.yaml...")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        db_config = config.get('database', {}).get('postgres', {})
        host = db_config.get('host', 'localhost')
        port = str(db_config.get('port', '5432'))
        database = db_config.get('database', 'fraud_detection')
        user = db_config.get('user', 'postgres')
        password = db_config.get('password', 'postgres123')
        
        success, result = test_connection(host, port, database, user, password)
        if success:
            print("✅ PostgreSQL connection from config.yaml successful!")
            print(f"   Host: {host}:{port}")
            print(f"   Database: {database}")
            print(f"   User: {user}")
            print()
            print("You can use:")
            print("  python scripts/sync_postgres_to_neo4j.py")
            return 0
        else:
            print(f"❌ Config PostgreSQL failed: {result}")
            print()
    
    # Try environment variables
    print("Testing PostgreSQL from environment variables...")
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'fraud_detection')
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'postgres123')
    
    success, result = test_connection(host, port, database, user, password)
    if success:
        print("✅ PostgreSQL connection from environment variables successful!")
        return 0
    else:
        print(f"❌ Environment PostgreSQL failed: {result}")
        print()
    
    # All failed
    print("=" * 60)
    print("❌ All connection attempts failed!")
    print("=" * 60)
    print()
    print("Please provide correct PostgreSQL credentials:")
    print()
    print("Option 1: Update config.yaml")
    print("  database:")
    print("    postgres:")
    print("      host: localhost")
    print("      port: 5432")
    print("      database: fraud_detection")
    print("      user: postgres")
    print("      password: YOUR_PASSWORD")
    print()
    print("Option 2: Set environment variables")
    print("  $env:POSTGRES_PASSWORD='your_password'")
    print()
    print("Option 3: Use command line arguments")
    print("  python scripts/sync_postgres_to_neo4j.py --pg-password YOUR_PASSWORD")
    print()
    print("Option 4: If using Docker PostgreSQL, check if it's running:")
    print("  docker-compose ps postgres")
    print()
    
    return 1

if __name__ == '__main__':
    sys.exit(main())

