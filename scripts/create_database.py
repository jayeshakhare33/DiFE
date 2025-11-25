#!/usr/bin/env python3
"""
Create fraud_detection database in PostgreSQL
"""
import sys
import os
import psycopg2
import yaml
from pathlib import Path

def create_database():
    """Create fraud_detection database if it doesn't exist"""
    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    db_config = config.get('database', {}).get('postgres', {})
    host = db_config.get('host', 'localhost')
    port = db_config.get('port', '5432')
    database = db_config.get('database', 'fraud_detection')
    user = db_config.get('user', 'postgres')
    password = db_config.get('password', 'postgres123')
    
    print(f"Connecting to PostgreSQL at {host}:{port}...")
    print(f"User: {user}")
    print(f"Database to create: {database}")
    print()
    
    try:
        # Connect to default 'postgres' database to create new database
        conn = psycopg2.connect(
            host=host,
            port=port,
            database='postgres',  # Connect to default database
            user=user,
            password=password
        )
        conn.autocommit = True  # Required for CREATE DATABASE
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database,)
        )
        exists = cursor.fetchone()
        
        if exists:
            print(f"✅ Database '{database}' already exists")
        else:
            # Create database
            cursor.execute(f'CREATE DATABASE {database}')
            print(f"✅ Created database '{database}'")
        
        cursor.close()
        conn.close()
        
        # Test connection to new database
        print(f"\nTesting connection to '{database}'...")
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
        print(f"✅ Successfully connected to '{database}'")
        print(f"   PostgreSQL version: {version[:50]}...")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if PostgreSQL is running")
        print("2. Verify password in config.yaml")
        print("3. Check if port 5432 is accessible")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = create_database()
    sys.exit(0 if success else 1)

