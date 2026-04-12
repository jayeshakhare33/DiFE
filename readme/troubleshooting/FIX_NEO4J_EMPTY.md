# Fix: Neo4j is Empty - No Data Found

## Problem
The feature extraction script is failing because Neo4j has no data. You need to load data into Neo4j first.

## Solution: Two Options

### Option 1: Sync from PostgreSQL (Recommended)

If you have data in PostgreSQL:

```bash
# Step 1: Sync PostgreSQL data to Neo4j
python scripts/sync_postgres_to_neo4j.py
```

This will:
- Load transactions from PostgreSQL
- Create User, Transaction, Location, Device, Card nodes in Neo4j
- Create relationships (SENT, TO, LOCATED_AT, etc.)

### Option 2: Manual Neo4j Initialization

If PostgreSQL is also empty, initialize Neo4j manually:

1. **Open Neo4j Browser**: http://localhost:7474
2. **Login**: username `neo4j`, password `neo4j123`
3. **Copy and paste** the entire contents of `scripts/init_neo4j.cypher`
4. **Click "Run"** button

This creates sample data for testing.

---

## Quick Fix Steps

### Step 1: Check if PostgreSQL has data

```bash
# Connect to PostgreSQL
docker exec -it fraud-detection-postgres psql -U postgres -d fraud_detection

# Check transactions
SELECT COUNT(*) FROM transactions;

# Exit
\q
```

### Step 2A: If PostgreSQL has data → Sync to Neo4j

```bash
python scripts/sync_postgres_to_neo4j.py
```

### Step 2B: If PostgreSQL is empty → Initialize both

**Initialize PostgreSQL:**
```bash
docker exec -i fraud-detection-postgres psql -U postgres -d fraud_detection < scripts/init_postgres.sql
```

**Then sync to Neo4j:**
```bash
python scripts/sync_postgres_to_neo4j.py
```

**OR manually initialize Neo4j:**
- Open http://localhost:7474
- Run `scripts/init_neo4j.cypher`

### Step 3: Verify Neo4j has data

```bash
python scripts/test_neo4j.py
```

Should show:
- ✅ User nodes: X
- ✅ Transaction nodes: X
- ✅ Relationships: X

### Step 4: Extract features

```bash
python scripts/extract_all_features.py
```

---

## If You Have Your Own Transaction Data

If you have a CSV file with transactions:

```bash
# Step 1: Load CSV to PostgreSQL
python scripts/load_transactions_to_postgres.py --file your_transactions.csv

# Step 2: Sync to Neo4j
python scripts/sync_postgres_to_neo4j.py

# Step 3: Extract features
python scripts/extract_all_features.py
```

---

## Expected CSV Format

Your CSV should have these columns:
- `sender_id` (required)
- `receiver_id` (required)
- `amount` (required)
- `timestamp` (required)
- `location_id` (optional)
- `device_id` (optional)
- `card_id` (optional)
- `is_fraud` or `is_fraud_txn` (optional)

---

## Quick Test

After syncing, verify in Neo4j Browser:

```cypher
// Count nodes
MATCH (u:User) RETURN count(u) as users
UNION ALL
MATCH (t:Transaction) RETURN count(t) as transactions

// View sample data
MATCH (u1:User)-[:SENT]->(t:Transaction)-[:TO]->(u2:User)
RETURN u1.user_id, u2.user_id, t.amount, t.is_fraud
LIMIT 10
```

If you see data, you're ready to extract features!

