# PostgreSQL Setup: Docker vs External

## Why Two PostgreSQL Options?

You have **two PostgreSQL instances** available:

1. **Docker PostgreSQL** (in docker-compose.yml)
   - Password: `postgres123`
   - Database: `fraud_detection` (auto-created)
   - Port: 5432
   - **Purpose**: Easy setup, isolated, consistent environment

2. **External PostgreSQL** (your existing installation)
   - Password: `M4RC0` (from your config.yaml)
   - Database: May not exist yet
   - Port: 5432 (same port - this causes conflicts!)
   - **Purpose**: Your existing database with your data

## The Problem

Both are trying to use port **5432**, which causes conflicts. You can only use **ONE** at a time.

---

## Which Should You Use?

### ✅ **Use External PostgreSQL** (Recommended for you)

**Why:**
- You already have it installed
- You already have data in it (5 transactions found)
- You control the credentials
- No Docker overhead
- Better for production

**What to do:**
1. **Stop Docker PostgreSQL** (free up port 5432)
2. **Use your external PostgreSQL** with password `M4RC0`
3. **Create database** `fraud_detection` if it doesn't exist
4. **Update config.yaml** to use external PostgreSQL

### ❌ **Use Docker PostgreSQL** (Not recommended for you)

**Why not:**
- You already have external PostgreSQL
- Port conflict (both use 5432)
- Extra Docker overhead
- Your data is in external PostgreSQL

**When to use:**
- If you want isolated development environment
- If you don't have PostgreSQL installed
- If you want everything in Docker

---

## Solution: Use Your External PostgreSQL

### Step 1: Stop Docker PostgreSQL

```bash
docker-compose stop postgres
```

This frees up port 5432 for your external PostgreSQL.

### Step 2: Create Database (if needed)

Connect to your external PostgreSQL:

```bash
# Using psql command line
psql -U postgres

# Or if you have pgAdmin or another tool, use that
```

Then create the database:

```sql
CREATE DATABASE fraud_detection;
```

### Step 3: Verify config.yaml

Your `config.yaml` already has the correct password:
```yaml
database:
  postgres:
    host: localhost
    port: 5432
    database: fraud_detection
    user: postgres
    password: M4RC0  # ✅ This is correct
```

### Step 4: Test Connection

```bash
python scripts/test_postgres_connection.py
```

Should now work!

### Step 5: Sync to Neo4j

```bash
python scripts/sync_postgres_to_neo4j.py
```

---

## Alternative: Use Docker PostgreSQL

If you prefer Docker PostgreSQL:

### Step 1: Stop External PostgreSQL

Stop your external PostgreSQL service:
```bash
# Windows Services
services.msc
# Find "PostgreSQL" and stop it
```

### Step 2: Update config.yaml

Change password to Docker PostgreSQL password:
```yaml
database:
  postgres:
    password: postgres123  # Docker PostgreSQL password
```

### Step 3: Start Docker PostgreSQL

```bash
docker-compose up -d postgres
```

---

## Recommendation

**Use your External PostgreSQL** because:
1. ✅ You already have it
2. ✅ You already have data (5 transactions)
3. ✅ You control it
4. ✅ No Docker overhead
5. ✅ Better for production

**Just need to:**
1. Stop Docker PostgreSQL
2. Create `fraud_detection` database (if needed)
3. Use your existing password `M4RC0`

---

## Quick Fix Commands

**To use External PostgreSQL:**

```bash
# 1. Stop Docker PostgreSQL
docker-compose stop postgres

# 2. Create database (if needed)
psql -U postgres -c "CREATE DATABASE fraud_detection;"

# 3. Test connection
python scripts/test_postgres_connection.py

# 4. Sync to Neo4j
python scripts/sync_postgres_to_neo4j.py
```

**To use Docker PostgreSQL:**

```bash
# 1. Stop external PostgreSQL (Windows Services)
# 2. Update config.yaml password to: postgres123
# 3. Start Docker PostgreSQL
docker-compose up -d postgres

# 4. Test connection
python scripts/test_postgres_connection.py
```

---

## Summary

- **Docker PostgreSQL**: Included for easy setup, but you don't need it
- **External PostgreSQL**: Your existing one - **USE THIS**
- **Action**: Stop Docker PostgreSQL, use external with password `M4RC0`

