// Neo4j Graph Database Initialization Script
// Run this in Neo4j Browser (http://localhost:7474)

// Create indexes for better performance
CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.user_id);
CREATE INDEX transaction_id_index IF NOT EXISTS FOR (t:Transaction) ON (t.transaction_id);
CREATE INDEX transaction_timestamp_index IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp);
CREATE INDEX location_id_index IF NOT EXISTS FOR (l:Location) ON (l.location_id);
CREATE INDEX device_id_index IF NOT EXISTS FOR (d:Device) ON (d.device_id);
CREATE INDEX card_id_index IF NOT EXISTS FOR (c:Card) ON (c.card_id);

// Create sample User nodes
MERGE (u1:User {user_id: 'user_001'})
SET u1.name = 'John Doe', u1.email = 'john@example.com', u1.created_at = datetime()

MERGE (u2:User {user_id: 'user_002'})
SET u2.name = 'Jane Smith', u2.email = 'jane@example.com', u2.created_at = datetime()

MERGE (u3:User {user_id: 'user_003'})
SET u3.name = 'Bob Johnson', u3.email = 'bob@example.com', u3.created_at = datetime()

// Create Location nodes
MERGE (l1:Location {location_id: 'loc_001'})
SET l1.country = 'USA', l1.city = 'New York', l1.latitude = 40.7128, l1.longitude = -74.0060

MERGE (l2:Location {location_id: 'loc_002'})
SET l2.country = 'USA', l2.city = 'Los Angeles', l2.latitude = 34.0522, l2.longitude = -118.2437

MERGE (l3:Location {location_id: 'loc_003'})
SET l3.country = 'USA', l3.city = 'Chicago', l3.latitude = 41.8781, l3.longitude = -87.6298

// Create Device nodes
MERGE (d1:Device {device_id: 'dev_001'})
SET d1.device_type = 'mobile', d1.device_info = 'iPhone 13'

MERGE (d2:Device {device_id: 'dev_002'})
SET d2.device_type = 'desktop', d2.device_info = 'Windows PC'

MERGE (d3:Device {device_id: 'dev_003'})
SET d3.device_type = 'mobile', d3.device_info = 'Android Phone'

// Create Card nodes
MERGE (c1:Card {card_id: 'card_001'})
SET c1.card_type = 'credit'

MERGE (c2:Card {card_id: 'card_002'})
SET c2.card_type = 'debit'

MERGE (c3:Card {card_id: 'card_003'})
SET c3.card_type = 'credit'

// Create Transaction nodes and relationships
// Transaction 1
MERGE (t1:Transaction {transaction_id: 'txn_001'})
SET t1.amount = 1500.50, t1.timestamp = datetime() - duration({days: 1}), t1.is_fraud = false

MERGE (u1)-[:SENT {amount: 1500.50, timestamp: datetime() - duration({days: 1})}]->(t1)
MERGE (t1)-[:TO {amount: 1500.50, timestamp: datetime() - duration({days: 1})}]->(u2)
MERGE (u1)-[:LOCATED_AT {timestamp: datetime() - duration({days: 1})}]->(l1)
MERGE (t1)-[:USED_DEVICE]->(d1)
MERGE (u1)-[:HAS_CARD]->(c1)

// Transaction 2
MERGE (t2:Transaction {transaction_id: 'txn_002'})
SET t2.amount = 750.25, t2.timestamp = datetime() - duration({hours: 12}), t2.is_fraud = false

MERGE (u2)-[:SENT {amount: 750.25, timestamp: datetime() - duration({hours: 12})}]->(t2)
MERGE (t2)-[:TO {amount: 750.25, timestamp: datetime() - duration({hours: 12})}]->(u1)
MERGE (u2)-[:LOCATED_AT {timestamp: datetime() - duration({hours: 12})}]->(l2)
MERGE (t2)-[:USED_DEVICE]->(d2)
MERGE (u2)-[:HAS_CARD]->(c2)

// Transaction 3
MERGE (t3:Transaction {transaction_id: 'txn_003'})
SET t3.amount = 2000.00, t3.timestamp = datetime() - duration({hours: 6}), t3.is_fraud = false

MERGE (u1)-[:SENT {amount: 2000.00, timestamp: datetime() - duration({hours: 6})}]->(t3)
MERGE (t3)-[:TO {amount: 2000.00, timestamp: datetime() - duration({hours: 6})}]->(u3)
MERGE (u1)-[:LOCATED_AT {timestamp: datetime() - duration({hours: 6})}]->(l1)
MERGE (t3)-[:USED_DEVICE]->(d1)
MERGE (u1)-[:HAS_CARD]->(c1)

// Transaction 4 (Fraud)
MERGE (t4:Transaction {transaction_id: 'txn_004'})
SET t4.amount = 500.00, t4.timestamp = datetime() - duration({hours: 3}), t4.is_fraud = true

MERGE (u3)-[:SENT {amount: 500.00, timestamp: datetime() - duration({hours: 3})}]->(t4)
MERGE (t4)-[:TO {amount: 500.00, timestamp: datetime() - duration({hours: 3})}]->(u2)
MERGE (u3)-[:LOCATED_AT {timestamp: datetime() - duration({hours: 3})}]->(l3)
MERGE (t4)-[:USED_DEVICE]->(d3)
MERGE (u3)-[:HAS_CARD]->(c3)

// Transaction 5
MERGE (t5:Transaction {transaction_id: 'txn_005'})
SET t5.amount = 1200.75, t5.timestamp = datetime() - duration({hours: 1}), t5.is_fraud = false

MERGE (u2)-[:SENT {amount: 1200.75, timestamp: datetime() - duration({hours: 1})}]->(t5)
MERGE (t5)-[:TO {amount: 1200.75, timestamp: datetime() - duration({hours: 1})}]->(u3)
MERGE (u2)-[:LOCATED_AT {timestamp: datetime() - duration({hours: 1})}]->(l2)
MERGE (t5)-[:USED_DEVICE]->(d2)
MERGE (u2)-[:HAS_CARD]->(c2)

// Return summary
MATCH (u:User) RETURN count(u) as users_count
UNION ALL
MATCH (t:Transaction) RETURN count(t) as transactions_count
UNION ALL
MATCH (t:Transaction {is_fraud: true}) RETURN count(t) as fraud_count
UNION ALL
MATCH ()-[r]->() RETURN count(r) as relationships_count;

