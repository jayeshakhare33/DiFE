#!/usr/bin/env python3
"""
Test Neo4j Connection
"""
from neo4j import GraphDatabase
import sys
import os

def test_neo4j():
    """Test Neo4j connection and basic operations"""
    try:
        # Get connection parameters
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'neo4j123')
        
        print(f"Connecting to Neo4j at {uri}...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            print("✅ Neo4j connection successful!")
            print(f"✅ Test result: {record['test']}")
            
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            record = result.single()
            node_count = record['count']
            print(f"✅ Total nodes in graph: {node_count}")
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = result.single()
            rel_count = record['count']
            print(f"✅ Total relationships: {rel_count}")
            
            # Check for User nodes
            result = session.run("MATCH (u:User) RETURN count(u) as count")
            record = result.single()
            user_count = record['count']
            print(f"✅ User nodes: {user_count}")
            
            # Check for Transaction nodes
            result = session.run("MATCH (t:Transaction) RETURN count(t) as count")
            record = result.single()
            txn_count = record['count']
            print(f"✅ Transaction nodes: {txn_count}")
            
            # Check for fraud transactions
            result = session.run("MATCH (t:Transaction {is_fraud: true}) RETURN count(t) as count")
            record = result.single()
            fraud_count = record['count']
            print(f"✅ Fraud transactions: {fraud_count}")
        
        driver.close()
        print("\n✅ All Neo4j tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Neo4j is running (check http://localhost:7474)")
        print("2. Verify credentials (default: neo4j/neo4j123)")
        print("3. Check if Neo4j is accessible at bolt://localhost:7687")
        return False

if __name__ == '__main__':
    success = test_neo4j()
    sys.exit(0 if success else 1)

