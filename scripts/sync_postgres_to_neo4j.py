#!/usr/bin/env python3
"""
Sync PostgreSQL Transaction Data to Neo4j Graph Database
Loads transactions from PostgreSQL and creates graph structure in Neo4j
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from neo4j import GraphDatabase
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgresToNeo4jSync:
    """Sync transaction data from PostgreSQL to Neo4j"""
    
    def __init__(self, config_path=None):
        """Initialize connections"""
        # Load config if available
        config = {}
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
        # PostgreSQL connection - try config first, then env, then defaults
        db_config = config.get('database', {}).get('postgres', {})
        pg_host = os.getenv('POSTGRES_HOST') or db_config.get('host', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT') or str(db_config.get('port', '5432'))
        pg_db = os.getenv('POSTGRES_DB') or db_config.get('database', 'fraud_detection')
        pg_user = os.getenv('POSTGRES_USER') or db_config.get('user', 'postgres')
        pg_password = os.getenv('POSTGRES_PASSWORD') or db_config.get('password', 'postgres123')
        
        # Try to connect to PostgreSQL
        try:
            self.pg_conn = psycopg2.connect(
                host=pg_host,
                port=pg_port,
                database=pg_db,
                user=pg_user,
                password=pg_password
            )
            logger.info(f"✅ Connected to PostgreSQL at {pg_host}:{pg_port}")
        except psycopg2.OperationalError as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.error("\nTroubleshooting:")
            logger.error("1. Check if PostgreSQL is running")
            logger.error("2. Verify credentials:")
            logger.error(f"   Host: {pg_host}")
            logger.error(f"   Port: {pg_port}")
            logger.error(f"   Database: {pg_db}")
            logger.error(f"   User: {pg_user}")
            logger.error("3. If using Docker PostgreSQL, password is: postgres123")
            logger.error("4. If using external PostgreSQL, check your password")
            logger.error("\nTo use Docker PostgreSQL:")
            logger.error("   docker exec -it fraud-detection-postgres psql -U postgres")
            logger.error("\nTo use external PostgreSQL, update config.yaml or set environment variables:")
            logger.error("   POSTGRES_PASSWORD=your_password")
            raise
        
        # Neo4j connection
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'neo4j123')
        
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    
    def load_transactions_from_postgres(self):
        """Load all transactions from PostgreSQL"""
        logger.info("Loading transactions from PostgreSQL...")
        
        query = """
        SELECT 
            t.transaction_id,
            t.sender_id,
            t.receiver_id,
            t.amount,
            t.timestamp,
            t.location_id,
            t.device_id,
            t.card_id,
            t.is_fraud
        FROM transactions t
        ORDER BY t.timestamp
        """
        
        df = pd.read_sql_query(query, self.pg_conn)
        logger.info(f"Loaded {len(df)} transactions from PostgreSQL")
        return df
    
    def load_users_from_postgres(self):
        """Load users from PostgreSQL"""
        logger.info("Loading users from PostgreSQL...")
        
        query = "SELECT user_id, name, email FROM users"
        
        try:
            df = pd.read_sql_query(query, self.pg_conn)
            logger.info(f"Loaded {len(df)} users from PostgreSQL")
            return df
        except Exception as e:
            logger.warning(f"Could not load users: {e}")
            return pd.DataFrame()
    
    def load_locations_from_postgres(self):
        """Load locations from PostgreSQL"""
        logger.info("Loading locations from PostgreSQL...")
        
        query = "SELECT location_id, country, city, latitude, longitude FROM locations"
        
        try:
            df = pd.read_sql_query(query, self.pg_conn)
            logger.info(f"Loaded {len(df)} locations from PostgreSQL")
            return df
        except Exception as e:
            logger.warning(f"Could not load locations: {e}")
            return pd.DataFrame()
    
    def create_indexes(self):
        """Create indexes in Neo4j"""
        logger.info("Creating indexes in Neo4j...")
        
        indexes = [
            "CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.user_id)",
            "CREATE INDEX transaction_id_index IF NOT EXISTS FOR (t:Transaction) ON (t.transaction_id)",
            "CREATE INDEX transaction_timestamp_index IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
            "CREATE INDEX location_id_index IF NOT EXISTS FOR (l:Location) ON (l.location_id)",
            "CREATE INDEX device_id_index IF NOT EXISTS FOR (d:Device) ON (d.device_id)",
            "CREATE INDEX card_id_index IF NOT EXISTS FOR (c:Card) ON (c.card_id)",
        ]
        
        with self.neo4j_driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.info(f"Created index: {index_query.split('FOR')[0].strip()}")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
    
    def sync_users(self, users_df):
        """Sync users to Neo4j"""
        if users_df.empty:
            logger.info("No users to sync")
            return
        
        logger.info(f"Syncing {len(users_df)} users to Neo4j...")
        
        with self.neo4j_driver.session() as session:
            for _, row in users_df.iterrows():
                query = """
                MERGE (u:User {user_id: $user_id})
                SET u.name = $name,
                    u.email = $email,
                    u.created_at = datetime()
                """
                session.run(query, 
                           user_id=str(row['user_id']),
                           name=row.get('name'),
                           email=row.get('email'))
        
        logger.info("Users synced successfully")
    
    def sync_locations(self, locations_df):
        """Sync locations to Neo4j"""
        if locations_df.empty:
            logger.info("No locations to sync")
            return
        
        logger.info(f"Syncing {len(locations_df)} locations to Neo4j...")
        
        with self.neo4j_driver.session() as session:
            for _, row in locations_df.iterrows():
                query = """
                MERGE (l:Location {location_id: $location_id})
                SET l.country = $country,
                    l.city = $city,
                    l.latitude = $latitude,
                    l.longitude = $longitude
                """
                session.run(query,
                           location_id=str(row['location_id']),
                           country=row.get('country'),
                           city=row.get('city'),
                           latitude=float(row.get('latitude', 0)) if pd.notna(row.get('latitude')) else None,
                           longitude=float(row.get('longitude', 0)) if pd.notna(row.get('longitude')) else None)
        
        logger.info("Locations synced successfully")
    
    def sync_transactions(self, transactions_df):
        """Sync transactions to Neo4j"""
        if transactions_df.empty:
            logger.warning("No transactions to sync!")
            return
        
        logger.info(f"Syncing {len(transactions_df)} transactions to Neo4j...")
        
        batch_size = 100
        total_batches = (len(transactions_df) + batch_size - 1) // batch_size
        
        with self.neo4j_driver.session() as session:
            for batch_idx in range(0, len(transactions_df), batch_size):
                batch = transactions_df.iloc[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} transactions)...")
                
                for _, row in batch.iterrows():
                    try:
                        # Convert timestamp to datetime if needed
                        timestamp = row['timestamp']
                        if isinstance(timestamp, str):
                            timestamp = pd.to_datetime(timestamp)
                        elif pd.isna(timestamp):
                            timestamp = datetime.now()
                        
                        # Create transaction node and relationships
                        # Split into main query and optional relationship queries
                        main_query = """
                        // Create or merge sender
                        MERGE (sender:User {user_id: $sender_id})
                        
                        // Create or merge receiver
                        MERGE (receiver:User {user_id: $receiver_id})
                        
                        // Create transaction node
                        MERGE (txn:Transaction {transaction_id: $transaction_id})
                        SET txn.amount = $amount,
                            txn.timestamp = datetime($timestamp),
                            txn.is_fraud = $is_fraud
                        
                        // Create SENT relationship
                        MERGE (sender)-[s:SENT]->(txn)
                        SET s.amount = $amount,
                            s.timestamp = datetime($timestamp)
                        
                        // Create TO relationship
                        MERGE (txn)-[t:TO]->(receiver)
                        SET t.amount = $amount,
                            t.timestamp = datetime($timestamp)
                        """
                        
                        # Run main query
                        session.run(main_query,
                                   transaction_id=str(row['transaction_id']),
                                   sender_id=str(row['sender_id']),
                                   receiver_id=str(row['receiver_id']),
                                   amount=float(row['amount']),
                                   timestamp=timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                                   is_fraud=bool(row.get('is_fraud', False)))
                        
                        # Optional: Create location relationship
                        location_id = str(row.get('location_id', '')) if pd.notna(row.get('location_id')) else None
                        if location_id and location_id != '':
                            location_query = """
                            MATCH (sender:User {user_id: $sender_id})
                            MERGE (loc:Location {location_id: $location_id})
                            MERGE (sender)-[l:LOCATED_AT]->(loc)
                            SET l.timestamp = datetime($timestamp)
                            """
                            session.run(location_query,
                                     sender_id=str(row['sender_id']),
                                     location_id=location_id,
                                     timestamp=timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp))
                        
                        # Optional: Create device relationship
                        device_id = str(row.get('device_id', '')) if pd.notna(row.get('device_id')) else None
                        if device_id and device_id != '':
                            device_query = """
                            MATCH (txn:Transaction {transaction_id: $transaction_id})
                            MERGE (dev:Device {device_id: $device_id})
                            MERGE (txn)-[:USED_DEVICE]->(dev)
                            """
                            session.run(device_query,
                                      transaction_id=str(row['transaction_id']),
                                      device_id=device_id)
                        
                        # Optional: Create card relationship
                        card_id = str(row.get('card_id', '')) if pd.notna(row.get('card_id')) else None
                        if card_id and card_id != '':
                            card_query = """
                            MATCH (sender:User {user_id: $sender_id})
                            MERGE (card:Card {card_id: $card_id})
                            MERGE (sender)-[:HAS_CARD]->(card)
                            """
                            session.run(card_query,
                                      sender_id=str(row['sender_id']),
                                      card_id=card_id)
                        
                    
                    except Exception as e:
                        logger.error(f"Error processing transaction {row.get('transaction_id', 'unknown')}: {e}")
                        continue
        
        logger.info("Transactions synced successfully")
    
    def verify_sync(self):
        """Verify data was synced correctly"""
        logger.info("Verifying sync...")
        
        with self.neo4j_driver.session() as session:
            # Count nodes
            result = session.run("MATCH (u:User) RETURN count(u) as count")
            user_count = result.single()['count']
            
            result = session.run("MATCH (t:Transaction) RETURN count(t) as count")
            txn_count = result.single()['count']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            
            logger.info(f"Neo4j now contains:")
            logger.info(f"  - {user_count} User nodes")
            logger.info(f"  - {txn_count} Transaction nodes")
            logger.info(f"  - {rel_count} Relationships")
            
            return user_count > 0 and txn_count > 0
    
    def run(self):
        """Run complete sync"""
        try:
            logger.info("=" * 60)
            logger.info("PostgreSQL to Neo4j Sync")
            logger.info("=" * 60)
            
            # Step 1: Create indexes
            self.create_indexes()
            
            # Step 2: Load data from PostgreSQL
            users_df = self.load_users_from_postgres()
            locations_df = self.load_locations_from_postgres()
            transactions_df = self.load_transactions_from_postgres()
            
            if transactions_df.empty:
                logger.error("No transactions found in PostgreSQL!")
                logger.error("Please load transaction data first.")
                return False
            
            # Step 3: Sync to Neo4j
            self.sync_users(users_df)
            self.sync_locations(locations_df)
            self.sync_transactions(transactions_df)
            
            # Step 4: Verify
            success = self.verify_sync()
            
            logger.info("=" * 60)
            if success:
                logger.info("✅ Sync completed successfully!")
            else:
                logger.warning("⚠️  Sync completed but verification failed")
            logger.info("=" * 60)
            
            return success
            
        except Exception as e:
            logger.error(f"Error during sync: {e}", exc_info=True)
            return False
        finally:
            self.pg_conn.close()
            self.neo4j_driver.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync PostgreSQL data to Neo4j')
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--pg-host', type=str, help='PostgreSQL host')
    parser.add_argument('--pg-port', type=str, help='PostgreSQL port')
    parser.add_argument('--pg-db', type=str, help='PostgreSQL database')
    parser.add_argument('--pg-user', type=str, help='PostgreSQL user')
    parser.add_argument('--pg-password', type=str, help='PostgreSQL password')
    
    args = parser.parse_args()
    
    # Override with command line args if provided
    if args.pg_host:
        os.environ['POSTGRES_HOST'] = args.pg_host
    if args.pg_port:
        os.environ['POSTGRES_PORT'] = args.pg_port
    if args.pg_db:
        os.environ['POSTGRES_DB'] = args.pg_db
    if args.pg_user:
        os.environ['POSTGRES_USER'] = args.pg_user
    if args.pg_password:
        os.environ['POSTGRES_PASSWORD'] = args.pg_password
    
    try:
        syncer = PostgresToNeo4jSync(config_path=args.config)
        success = syncer.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Failed to sync: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

