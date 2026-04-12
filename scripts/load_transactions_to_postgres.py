#!/usr/bin/env python3
"""
Load Transaction Data from CSV to PostgreSQL
"""
import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_transactions_from_csv(csv_path, pg_conn):
    """Load transactions from CSV file"""
    logger.info(f"Loading transactions from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Read {len(df)} rows from CSV")
    
    # Map columns (adjust based on your CSV structure)
    column_mapping = {
        'sender_id': 'sender_id',
        'receiver_id': 'receiver_id',
        'amount': 'amount',
        'timestamp': 'timestamp',
        'location_id': 'location_id',
        'device_id': 'device_id',
        'card_id': 'card_id',
        'is_fraud': 'is_fraud',
        'is_fraud_txn': 'is_fraud',  # Alternative column name
    }
    
    # Prepare data
    records = []
    for _, row in df.iterrows():
        record = {
            'sender_id': str(row.get('sender_id', '')),
            'receiver_id': str(row.get('receiver_id', '')),
            'amount': float(row.get('amount', 0.0)),
            'timestamp': pd.to_datetime(row.get('timestamp', pd.Timestamp.now())),
            'location_id': str(row.get('location_id', '')) if pd.notna(row.get('location_id')) else None,
            'device_id': str(row.get('device_id', '')) if pd.notna(row.get('device_id')) else None,
            'card_id': str(row.get('card_id', '')) if pd.notna(row.get('card_id')) else None,
            'is_fraud': bool(row.get('is_fraud', False) or row.get('is_fraud_txn', False)),
        }
        records.append(record)
    
    # Insert into PostgreSQL
    cursor = pg_conn.cursor()
    
    insert_query = """
    INSERT INTO transactions 
    (sender_id, receiver_id, amount, timestamp, location_id, device_id, card_id, is_fraud)
    VALUES %s
    ON CONFLICT DO NOTHING
    """
    
    values = [
        (
            r['sender_id'],
            r['receiver_id'],
            r['amount'],
            r['timestamp'],
            r['location_id'],
            r['device_id'],
            r['card_id'],
            r['is_fraud']
        )
        for r in records
    ]
    
    execute_values(cursor, insert_query, values)
    pg_conn.commit()
    
    logger.info(f"Inserted {len(records)} transactions into PostgreSQL")
    cursor.close()


def main():
    parser = argparse.ArgumentParser(description='Load transactions from CSV to PostgreSQL')
    parser.add_argument('--file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--host', type=str, default='localhost', help='PostgreSQL host')
    parser.add_argument('--port', type=str, default='5432', help='PostgreSQL port')
    parser.add_argument('--database', type=str, default='fraud_detection', help='Database name')
    parser.add_argument('--user', type=str, default='postgres', help='PostgreSQL user')
    parser.add_argument('--password', type=str, default='postgres123', help='PostgreSQL password')
    
    args = parser.parse_args()
    
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            database=args.database,
            user=args.user,
            password=args.password
        )
        logger.info(f"Connected to PostgreSQL")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return 1
    
    # Load transactions
    try:
        load_transactions_from_csv(args.file, conn)
        logger.info("✅ Successfully loaded transactions!")
        return 0
    except Exception as e:
        logger.error(f"Error loading transactions: {e}", exc_info=True)
        return 1
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())

