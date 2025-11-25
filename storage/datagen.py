#!/usr/bin/env python3
"""
Synthetic India-Focused Fraud Dataset Generator (Explainable Version)
---------------------------------------------------------------------
Generates and inserts into PostgreSQL:
  - users table
  - transactions table
  - locations table
  - devices table (derived from transactions)
  - cards table (derived from transactions)

Also optionally generates CSV files.

Features:
  ✓ 95% domestic (India), 5% international
  ✓ Realistic UPI / NEFT / IMPS modes
  ✓ Fraud logic based on behavior & transaction anomalies
  ✓ Cross-border fraud propagation
  ✓ Explainable fraud reasons
"""

import os
import sys
import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------- CONFIGURATION ----------------
SEED = 42
OUT_DIR = "./india_fraud_data_explainable"

N_USERS = 50  # Increased to support more transactions
TARGET_TRANSACTIONS = 500  # Target 500 transactions

FRAUD_RATE = 0.035  # initial guess, replaced by logical tagging
USE_EPOCH_MS = True

YEARS_BACK_USERS = 3
YEARS_TX_COVERAGE = 2

RECIPROCAL_RATIO = 0.10
FAILED_REVERSED_RATIO = 0.025
ROUND_AMOUNT_RATIO = 0.08
THRESHOLD_AMOUNT_RATIO = 0.03

# Country setup
COUNTRIES = ["IN", "US", "AE", "SG", "GB"]
COUNTRY_WEIGHTS = [0.95, 0.015, 0.015, 0.01, 0.01]
CURRENCIES = {"IN": "INR", "US": "USD", "AE": "AED", "SG": "SGD", "GB": "GBP"}
COUNTRY_LATLON = {
    "IN": (20.5937, 78.9629),
    "US": (37.0902, -95.7129),
    "AE": (23.4241, 53.8478),
    "SG": (1.3521, 103.8198),
    "GB": (55.3781, -3.4360)
}

DEVICE_TYPES = ["android", "ios", "desktop", "feature_phone"]
ACCOUNT_TYPES = ["savings", "current", "merchant", "salary", "wallet"]
TX_MODES = ["upi", "wallet", "card", "net_banking", "neft", "imps"]
TX_STATUSES = ["success", "failed", "reversed"]

# ---------------- POSTGRESQL CONFIGURATION ----------------
SAVE_TO_POSTGRES = True  # Set to False to only generate CSV files
SAVE_TO_CSV = True  # Set to False to only save to PostgreSQL

def load_config():
    """Load configuration from config.yaml"""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_postgres_connection():
    """Get PostgreSQL connection from config"""
    config = load_config()
    pg_config = config['database']['postgres']
    return psycopg2.connect(
        host=pg_config['host'],
        port=pg_config['port'],
        database=pg_config['database'],
        user=pg_config['user'],
        password=pg_config['password']
    )

def create_tables_if_not_exists(conn):
    """Create PostgreSQL tables if they don't exist"""
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Locations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            location_id VARCHAR(50) PRIMARY KEY,
            country VARCHAR(100),
            city VARCHAR(100),
            latitude DECIMAL(10, 8),
            longitude DECIMAL(11, 8)
        );
    """)
    
    # Devices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id VARCHAR(50) PRIMARY KEY,
            device_type VARCHAR(100),
            device_info VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Cards table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            card_id VARCHAR(50) PRIMARY KEY,
            card_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Check if transactions table exists and what type transaction_id is
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'transactions' AND column_name = 'transaction_id';
    """)
    result = cursor.fetchone()
    
    if result:
        # Table exists, check if transaction_id is VARCHAR
        if result[1] not in ('character varying', 'varchar', 'text'):
            # Need to alter the column type
            print(f"[i] Converting transaction_id from {result[1]} to VARCHAR(50)...")
            try:
                # Drop the primary key constraint first
                cursor.execute("ALTER TABLE transactions DROP CONSTRAINT IF EXISTS transactions_pkey;")
                # Alter the column type
                cursor.execute("ALTER TABLE transactions ALTER COLUMN transaction_id TYPE VARCHAR(50);")
                # Recreate primary key
                cursor.execute("ALTER TABLE transactions ADD PRIMARY KEY (transaction_id);")
                conn.commit()
                print("[✓] transaction_id column type updated to VARCHAR(50)")
            except Exception as e:
                print(f"[⚠] Could not alter column type: {e}")
                print("[i] Dropping and recreating transactions table...")
                cursor.execute("DROP TABLE IF EXISTS transactions CASCADE;")
                conn.commit()
                # Recreate table below
                result = None  # Force table creation
    
    if not result:
        # Transactions table doesn't exist or was dropped, create it
        cursor.execute("""
            CREATE TABLE transactions (
                transaction_id VARCHAR(50) PRIMARY KEY,
                sender_id VARCHAR(50) NOT NULL,
                receiver_id VARCHAR(50) NOT NULL,
                amount DECIMAL(15, 2) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                location_id VARCHAR(50),
                device_id VARCHAR(50),
                card_id VARCHAR(50),
                is_fraud BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
    
    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender_id);",
        "CREATE INDEX IF NOT EXISTS idx_transactions_receiver ON transactions(receiver_id);",
        "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);",
    ]
    
    for idx_sql in indexes:
        cursor.execute(idx_sql)
    
    conn.commit()
    cursor.close()
    print("[✓] PostgreSQL tables created/verified")

def epoch_ms_to_datetime(epoch_val):
    """Convert epoch milliseconds or datetime string to datetime object"""
    if isinstance(epoch_val, datetime):
        return epoch_val
    elif isinstance(epoch_val, (int, float)):
        # Epoch milliseconds
        return datetime.fromtimestamp(epoch_val / 1000.0, tz=timezone.utc)
    elif isinstance(epoch_val, str):
        # Try to parse as datetime string
        try:
            # Try ISO format first
            if 'T' in epoch_val or ' ' in epoch_val:
                return datetime.fromisoformat(epoch_val.replace('Z', '+00:00'))
            # Otherwise try as epoch string
            return datetime.fromtimestamp(float(epoch_val) / 1000.0, tz=timezone.utc)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)

# ---------------- UTILS ----------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def rand_dt_between(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def dt_to_epoch_ms(dt):
    return int(dt.timestamp() * 1000)

def epoch_or_str(dt):
    return dt_to_epoch_ms(dt) if USE_EPOCH_MS else dt.strftime("%Y-%m-%d %H:%M:%S")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# ---------------- USER GENERATION ----------------
def generate_users(n_users, start_dt, end_dt):
    """Generate user metadata."""
    user_ids = [f"U{i:07d}" for i in range(n_users)]
    countries = np.random.choice(COUNTRIES, size=n_users, p=COUNTRY_WEIGHTS)
    account_times = [rand_dt_between(start_dt, end_dt) for _ in range(n_users)]
    device_types = np.random.choice(DEVICE_TYPES, size=n_users)
    account_types = np.random.choice(ACCOUNT_TYPES, size=n_users)
    kyc_verified = np.random.choice([0, 1], size=n_users, p=[0.2, 0.8])
    avg_balance = np.clip(np.random.normal(3500, 2500, n_users), 0, None)
    risk_score = np.random.beta(2, 8, n_users)

    df = pd.DataFrame({
        "user_id": user_ids,
        "account_creation_time": [epoch_or_str(t) for t in account_times],
        "country": countries,
        "risk_score": np.round(risk_score, 4),
        "device_type": device_types,
        "kyc_verified": kyc_verified,
        "account_type": account_types,
        "avg_balance": np.round(avg_balance, 2)
    })
    return df

# ---------------- TRANSACTION GENERATION ----------------
def gen_amount():
    val = np.random.lognormal(mean=5.3, sigma=1.2)
    if random.random() < ROUND_AMOUNT_RATIO:
        val = random.choice([100, 200, 500, 1000, 2000, 5000, 10000])
    elif random.random() < THRESHOLD_AMOUNT_RATIO:
        val = random.choice([9999, 1999, 4999]) - random.choice([0.01, 0.05, 0.10])
    return round(min(val, 200000.0), 2)

def generate_transactions(users_df, tx_start, tx_end):
    user_ids = users_df["user_id"].tolist()
    countries = dict(zip(users_df["user_id"], users_df["country"]))
    tx_records = []
    tx_id = 0
    reciprocal_buffer = []

    for uid in user_ids:
        # Limit transactions per user to reach target of 500
        remaining = TARGET_TRANSACTIONS - len(tx_records)
        if remaining <= 0:
            break
        # More transactions per user for 500 total
        n_tx = min(np.random.randint(5, 25), remaining)
        for _ in range(n_tx):
            if len(tx_records) >= TARGET_TRANSACTIONS:
                break
            receiver = random.choice(user_ids)
            if receiver == uid:
                continue

            sender_country = countries[uid]
            receiver_country = random.choices(COUNTRIES, COUNTRY_WEIGHTS, k=1)[0]
            is_cross = sender_country != receiver_country

            lat1, lon1 = COUNTRY_LATLON[sender_country]
            lat2, lon2 = COUNTRY_LATLON[receiver_country]
            distance = round(haversine_km(lat1, lon1, lat2, lon2), 2)

            t = rand_dt_between(tx_start, tx_end)
            amount = gen_amount()
            mode = random.choice(TX_MODES)
            status = np.random.choice(TX_STATUSES,
                                      p=[1 - FAILED_REVERSED_RATIO,
                                         FAILED_REVERSED_RATIO / 2,
                                         FAILED_REVERSED_RATIO / 2])

            tx_records.append({
                "transaction_id": f"T{tx_id:012d}",
                "sender_id": uid,
                "receiver_id": receiver,
                "amount": amount,
                "timestamp": epoch_or_str(t),
                "mode": mode,
                "status": status,
                "currency": CURRENCIES[sender_country],
                "sender_country": sender_country,
                "receiver_country": receiver_country,
                "is_cross_border": bool(is_cross),
                "device_id": uuid.uuid4().hex[:16],
                "geo_distance_km": distance
            })
            tx_id += 1

            if random.random() < RECIPROCAL_RATIO:
                reciprocal_buffer.append((receiver, uid, amount, t))

    # add reciprocal transactions (respecting target limit)
    for (s, r, amt, t) in reciprocal_buffer:
        if len(tx_records) >= TARGET_TRANSACTIONS:
            break
        t2 = t + timedelta(minutes=random.randint(1, 120))
        lat1, lon1 = COUNTRY_LATLON[countries[s]]
        lat2, lon2 = COUNTRY_LATLON[countries[r]]
        tx_records.append({
            "transaction_id": f"T{tx_id:012d}",
            "sender_id": s,
            "receiver_id": r,
            "amount": gen_amount(),
            "timestamp": epoch_or_str(t2),
            "mode": random.choice(TX_MODES),
            "status": "success",
            "currency": CURRENCIES[countries[s]],
            "sender_country": countries[s],
            "receiver_country": countries[r],
            "is_cross_border": countries[s] != countries[r],
            "device_id": uuid.uuid4().hex[:16],
            "geo_distance_km": round(haversine_km(lat1, lon1, lat2, lon2), 2)
        })
        tx_id += 1

    return pd.DataFrame(tx_records)

# ---------------- FRAUD LABELLING ----------------
def label_fraudulent_users(users_df, tx_df):
    """Enhanced user fraud detection with sophisticated patterns."""
    stats = tx_df.groupby("sender_id").agg(
        sent_count=("transaction_id", "count"),
        sent_sum=("amount", "sum"),
        sent_mean=("amount", "mean"),
        sent_max=("amount", "max"),
        sent_std=("amount", "std")
    ).reset_index()
    users_df = users_df.merge(stats, left_on="user_id", right_on="sender_id", how="left").fillna(0)

    # Calculate additional metrics
    user_cross_border = tx_df.groupby("sender_id")["is_cross_border"].sum().to_dict()
    user_failed_txs = tx_df[tx_df["status"] != "success"].groupby("sender_id").size().to_dict()
    user_devices = tx_df.groupby("sender_id")["device_id"].nunique().to_dict()
    
    fraud_flags = []
    fraud_reasons = []
    for _, row in users_df.iterrows():
        risk = 0.0
        reasons = []
        user_id = row["user_id"]

        # Pattern 1: Unverified high volume
        if row["kyc_verified"] == 0 and row["sent_sum"] > 20000:
            risk += 0.5
            reasons.append("unverified_high_volume")
        elif row["kyc_verified"] == 0 and row["sent_sum"] > 10000:
            risk += 0.3
            reasons.append("unverified_moderate_volume")

        # Pattern 2: Low balance, high spending
        if row["avg_balance"] < 500 and row["sent_sum"] > 50000:
            risk += 0.5
            reasons.append("low_balance_high_spend")
        elif row["avg_balance"] < 1000 and row["sent_sum"] > 30000:
            risk += 0.3
            reasons.append("low_balance_moderate_spend")

        # Pattern 3: Unusual transaction frequency (adjusted for 500 transactions)
        if row["sent_count"] > 30:  # More than 30 transactions
            risk += 0.4
            reasons.append("high_tx_frequency")
        elif row["sent_count"] > 20:
            risk += 0.2
            reasons.append("moderate_tx_frequency")

        # Pattern 4: High risk score
        if row["risk_score"] > 0.8:
            risk += 0.4
            reasons.append("high_risk_score")
        elif row["risk_score"] > 0.6:
            risk += 0.2
            reasons.append("moderate_risk_score")

        # Pattern 5: Fake merchant pattern
        if row["account_type"] == "merchant" and row["sent_mean"] < 50:
            risk += 0.3
            reasons.append("fake_merchant")
        elif row["account_type"] == "merchant" and row["sent_count"] < 5:
            risk += 0.2
            reasons.append("inactive_merchant")

        # Pattern 6: High amount variance (money laundering pattern)
        if row["sent_std"] > 0 and row["sent_mean"] > 0:
            cv = row["sent_std"] / row["sent_mean"]  # Coefficient of variation
            if cv > 2.0:  # High variance
                risk += 0.3
                reasons.append("high_amount_variance")

        # Pattern 7: Excessive cross-border transactions
        if user_id in user_cross_border:
            cross_border_ratio = user_cross_border[user_id] / max(row["sent_count"], 1)
            if cross_border_ratio > 0.5 and row["sent_count"] > 10:
                risk += 0.4
                reasons.append("excessive_cross_border")
            elif cross_border_ratio > 0.3:
                risk += 0.2
                reasons.append("frequent_cross_border")

        # Pattern 8: High failure rate
        if user_id in user_failed_txs:
            failure_rate = user_failed_txs[user_id] / max(row["sent_count"], 1)
            if failure_rate > 0.3 and row["sent_count"] > 5:
                risk += 0.4
                reasons.append("high_failure_rate")
            elif failure_rate > 0.2:
                risk += 0.2
                reasons.append("moderate_failure_rate")

        # Pattern 9: Device switching
        if user_id in user_devices and user_devices[user_id] > 5:
            risk += 0.3
            reasons.append("excessive_device_switching")
        elif user_id in user_devices and user_devices[user_id] > 3:
            risk += 0.2
            reasons.append("frequent_device_switching")

        # Pattern 10: Extremely high single transaction
        if row["sent_max"] > 100000:
            risk += 0.4
            reasons.append("extremely_high_single_tx")
        elif row["sent_max"] > 50000:
            risk += 0.2
            reasons.append("very_high_single_tx")

        # Pattern 11: Baseline random risk
        if random.random() < 0.05:
            risk += 0.3
            reasons.append("baseline_risk")

        is_fraud = 1 if risk > 0.5 else 0
        fraud_flags.append(is_fraud)
        fraud_reasons.append(",".join(reasons) if is_fraud else "")

    users_df["is_fraud"] = fraud_flags
    users_df["fraud_reason"] = fraud_reasons
    return users_df

def label_fraudulent_transactions(tx_df, users_df):
    """Enhanced logical transaction fraud tagging with sophisticated patterns."""
    fraud_users = set(users_df.loc[users_df["is_fraud"] == 1, "user_id"])
    tx_df = tx_df.copy()
    tx_df["is_fraud_txn"] = 0
    tx_df["fraud_reason"] = ""
    
    # Convert timestamps to datetime for analysis
    def convert_timestamp(ts):
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        elif isinstance(ts, str):
            try:
                return pd.to_datetime(ts)
            except:
                return datetime.now(timezone.utc)
        return ts
    
    tx_df['timestamp_dt'] = tx_df['timestamp'].apply(convert_timestamp)
    
    # Calculate transaction velocity (transactions per hour per sender)
    tx_df_sorted = tx_df.sort_values('timestamp_dt').copy()
    sender_velocity = {}
    for sender_id in tx_df['sender_id'].unique():
        sender_txs = tx_df_sorted[tx_df_sorted['sender_id'] == sender_id].copy()
        if len(sender_txs) > 1:
            sender_txs = sender_txs.sort_values('timestamp_dt')
            time_diffs = sender_txs['timestamp_dt'].diff().dt.total_seconds() / 3600  # hours
            rapid_txs = (time_diffs < 0.1).sum() if len(time_diffs) > 0 else 0  # transactions within 6 minutes
            sender_velocity[sender_id] = {
                'rapid_count': int(rapid_txs),
                'total_txs': len(sender_txs),
                'avg_time_diff': float(time_diffs.mean()) if len(time_diffs) > 0 and not time_diffs.isna().all() else 24.0
            }
        else:
            sender_velocity[sender_id] = {'rapid_count': 0, 'total_txs': 1, 'avg_time_diff': 24.0}
    
    # Calculate amount patterns
    sender_amounts = {}
    for sender_id in tx_df['sender_id'].unique():
        sender_tx_amounts = tx_df[tx_df['sender_id'] == sender_id]['amount']
        sender_amounts[sender_id] = {
            'mean': float(sender_tx_amounts.mean()),
            'std': float(sender_tx_amounts.std()) if len(sender_tx_amounts) > 1 else 0.0,
            'count': len(sender_tx_amounts)
        }
    
    # Device switching patterns
    sender_devices = tx_df.groupby('sender_id')['device_id'].nunique().to_dict()
    
    # Time-based patterns (unusual hours)
    tx_df['hour'] = tx_df['timestamp_dt'].dt.hour
    unusual_hours_mask = (tx_df['hour'] < 3) | (tx_df['hour'] > 22)  # Late night/early morning
    
    fraud_count = 0
    MAX_FRAUD_TXN = int(len(tx_df) * 0.15)  # 15% fraud rate for 500 transactions = ~75 frauds
    
    # Score each transaction
    for idx, row in tx_df.iterrows():
        risk_score = 0.0
        reasons = []
        
        sender_id = row['sender_id']
        receiver_id = row['receiver_id']
        amount = row['amount']
        
        # Pattern 1: Fraud user network
        if sender_id in fraud_users and receiver_id in fraud_users:
            risk_score += 0.6
            reasons.append("fraud_network")
        
        # Pattern 2: Transaction velocity (rapid successive transactions)
        if sender_id in sender_velocity:
            vel = sender_velocity[sender_id]
            if vel['rapid_count'] > 3:
                risk_score += 0.5
                reasons.append("high_velocity")
            if vel['total_txs'] > 30 and vel['avg_time_diff'] < 1:  # More than 30 txs with avg < 1 hour apart
                risk_score += 0.4
                reasons.append("unusual_frequency")
        
        # Pattern 3: Amount anomalies
        if 9990 <= amount <= 10010:  # Threshold avoidance
            risk_score += 0.5
            reasons.append("threshold_avoidance")
        elif amount % 1000 == 0 and amount > 5000:  # Large round numbers
            risk_score += 0.3
            reasons.append("round_amount_large")
        elif sender_id in sender_amounts:
            sender_stats = sender_amounts[sender_id]
            if sender_stats['std'] > 0:
                z_score = abs((amount - sender_stats['mean']) / sender_stats['std'])
                if z_score > 3:  # Statistical outlier
                    risk_score += 0.4
                    reasons.append("amount_outlier")
        
        # Pattern 4: Cross-border anomalies
        if row['is_cross_border']:
            if amount > 10000:
                risk_score += 0.5
                reasons.append("cross_border_high")
            elif amount > 5000:
                risk_score += 0.3
                reasons.append("cross_border_moderate")
            # Rapid cross-border transactions
            sender_cross = tx_df[(tx_df['sender_id'] == sender_id) & 
                                 (tx_df['is_cross_border'] == True)]
            if len(sender_cross) > 5:
                risk_score += 0.3
                reasons.append("frequent_cross_border")
        
        # Pattern 5: Failed/reversed high-value transactions
        if row['status'] != 'success':
            if amount > 5000:
                risk_score += 0.4
                reasons.append("failed_high_value")
            elif amount > 2000:
                risk_score += 0.2
                reasons.append("failed_moderate")
        
        # Pattern 6: Device switching (multiple devices in short time)
        if sender_id in sender_devices and sender_devices[sender_id] > 3:
            risk_score += 0.3
            reasons.append("device_switching")
        
        # Pattern 7: Unusual time patterns
        if unusual_hours_mask.loc[idx] and amount > 2000:
            risk_score += 0.3
            reasons.append("unusual_hours")
        
        # Pattern 8: Geographic distance anomalies
        if row['geo_distance_km'] > 5000 and amount > 3000:  # Long distance, high amount
            risk_score += 0.4
            reasons.append("long_distance_high_amount")
        
        # Pattern 9: Transaction mode anomalies
        if row['mode'] in ['net_banking', 'card'] and amount < 100:  # Expensive mode for small amount
            risk_score += 0.2
            reasons.append("mode_mismatch")
        
        # Pattern 10: Reciprocity patterns (money laundering)
        reciprocal_txs = tx_df[(tx_df['sender_id'] == receiver_id) & 
                               (tx_df['receiver_id'] == sender_id)]
        if len(reciprocal_txs) > 2:
            risk_score += 0.4
            reasons.append("reciprocal_pattern")
        
        # Pattern 11: Very high amounts
        if amount > 50000:
            risk_score += 0.5
            reasons.append("extremely_high_amount")
        elif amount > 20000:
            risk_score += 0.3
            reasons.append("very_high_amount")
        
        # Pattern 12: Random noise (baseline fraud)
        if random.random() < 0.05:
            risk_score += 0.3
            reasons.append("baseline_risk")
        
        # Flag as fraud if risk score exceeds threshold
        is_fraud = 1 if risk_score >= 0.5 else 0
        
        if is_fraud:
            fraud_count += 1
            tx_df.at[idx, "is_fraud_txn"] = 1
            tx_df.at[idx, "fraud_reason"] = ",".join(reasons) if reasons else "multiple_indicators"
        else:
            tx_df.at[idx, "is_fraud_txn"] = 0
            tx_df.at[idx, "fraud_reason"] = ""
    
    # Clean up temporary columns
    tx_df.drop(['timestamp_dt', 'hour'], axis=1, errors='ignore')
    
    print(f"[i] Flagged {fraud_count} fraudulent transactions ({fraud_count/len(tx_df)*100:.1f}%)")
    return tx_df

# ---------------- POSTGRESQL INSERTION ----------------
def insert_users_to_postgres(conn, users_df):
    """Insert users into PostgreSQL"""
    cursor = conn.cursor()
    
    # Prepare user data for PostgreSQL
    user_records = []
    for _, row in users_df.iterrows():
        # Convert epoch to datetime if needed
        created_at = epoch_ms_to_datetime(row['account_creation_time'])
        
        # Generate name and email from user_id
        name = f"User {row['user_id']}"
        email = f"{row['user_id'].lower()}@example.com"
        
        user_records.append((
            row['user_id'],
            name,
            email,
            created_at
        ))
    
    execute_batch(
        cursor,
        "INSERT INTO users (user_id, name, email, created_at) VALUES (%s, %s, %s, %s) ON CONFLICT (user_id) DO UPDATE SET name = EXCLUDED.name, email = EXCLUDED.email",
        user_records
    )
    conn.commit()
    cursor.close()
    print(f"[✓] Inserted {len(user_records)} users into PostgreSQL")

def insert_locations_to_postgres(conn, loc_df):
    """Insert locations into PostgreSQL"""
    cursor = conn.cursor()
    
    location_records = []
    city_map = {
        "IN": "Mumbai",
        "US": "New York",
        "AE": "Dubai",
        "SG": "Singapore",
        "GB": "London"
    }
    
    for _, row in loc_df.iterrows():
        location_id = f"loc_{row['country']}"
        location_records.append((
            location_id,
            row['country'],
            city_map.get(row['country'], 'Unknown'),
            float(row['lat']),
            float(row['lon'])
        ))
    
    execute_batch(
        cursor,
        "INSERT INTO locations (location_id, country, city, latitude, longitude) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (location_id) DO UPDATE SET country = EXCLUDED.country, city = EXCLUDED.city, latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude",
        location_records
    )
    conn.commit()
    cursor.close()
    print(f"[✓] Inserted {len(location_records)} locations into PostgreSQL")

def insert_devices_to_postgres(conn, tx_df):
    """Insert unique devices into PostgreSQL"""
    cursor = conn.cursor()
    
    # Get unique devices from transactions
    unique_devices = tx_df[['device_id']].drop_duplicates()
    device_types = ["android", "ios", "desktop", "feature_phone"]
    
    device_records = []
    for _, row in unique_devices.iterrows():
        device_type = random.choice(device_types)
        device_info = f"{device_type} device"
        device_records.append((
            row['device_id'],
            device_type,
            device_info
        ))
    
    if device_records:
        execute_batch(
            cursor,
            "INSERT INTO devices (device_id, device_type, device_info) VALUES (%s, %s, %s) ON CONFLICT (device_id) DO NOTHING",
            device_records
        )
        conn.commit()
        print(f"[✓] Inserted {len(device_records)} devices into PostgreSQL")
    
    cursor.close()

def insert_cards_to_postgres(conn, tx_df):
    """Insert cards for transactions into PostgreSQL"""
    cursor = conn.cursor()
    
    # Generate card_id for each transaction based on sender
    card_types = ["credit", "debit", "prepaid"]
    card_map = {}
    
    for sender_id in tx_df['sender_id'].unique():
        if sender_id not in card_map:
            card_map[sender_id] = f"card_{sender_id}_{uuid.uuid4().hex[:8]}"
    
    card_records = []
    for sender_id, card_id in card_map.items():
        card_type = random.choice(card_types)
        card_records.append((card_id, card_type))
    
    if card_records:
        execute_batch(
            cursor,
            "INSERT INTO cards (card_id, card_type) VALUES (%s, %s) ON CONFLICT (card_id) DO NOTHING",
            card_records
        )
        conn.commit()
        print(f"[✓] Inserted {len(card_records)} cards into PostgreSQL")
    
    cursor.close()
    return card_map

def insert_transactions_to_postgres(conn, tx_df, card_map):
    """Insert transactions into PostgreSQL"""
    cursor = conn.cursor()
    
    transaction_records = []
    for _, row in tx_df.iterrows():
        # Convert timestamp (handles both epoch_ms and string formats)
        timestamp = epoch_ms_to_datetime(row['timestamp'])
        
        # Get location_id from sender_country
        location_id = f"loc_{row['sender_country']}"
        
        # Get card_id
        card_id = card_map.get(row['sender_id'], None)
        
        transaction_records.append((
            row['transaction_id'],
            row['sender_id'],
            row['receiver_id'],
            float(row['amount']),
            timestamp,
            location_id,
            row['device_id'],
            card_id,
            bool(row['is_fraud_txn'])
        ))
    
    execute_batch(
        cursor,
        """INSERT INTO transactions (transaction_id, sender_id, receiver_id, amount, timestamp, location_id, device_id, card_id, is_fraud) 
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) 
           ON CONFLICT (transaction_id) DO UPDATE SET 
           sender_id = EXCLUDED.sender_id, receiver_id = EXCLUDED.receiver_id, 
           amount = EXCLUDED.amount, timestamp = EXCLUDED.timestamp, 
           location_id = EXCLUDED.location_id, device_id = EXCLUDED.device_id, 
           card_id = EXCLUDED.card_id, is_fraud = EXCLUDED.is_fraud, updated_at = NOW()""",
        transaction_records,
        page_size=100
    )
    conn.commit()
    cursor.close()
    print(f"[✓] Inserted {len(transaction_records)} transactions into PostgreSQL")

# ---------------- MAIN ----------------
def main():
    set_seed(SEED)
    
    if SAVE_TO_CSV:
        ensure_outdir(OUT_DIR)

    now = datetime.now(timezone.utc)
    users_start = now - timedelta(days=365 * YEARS_BACK_USERS)
    tx_start = now - timedelta(days=365 * YEARS_TX_COVERAGE)

    print("[i] Generating users...")
    users_df = generate_users(N_USERS, users_start, now)
    print(f"[✓] Generated {len(users_df)} users")

    print("[i] Generating transactions (this may take some time)...")
    tx_df = generate_transactions(users_df, tx_start, now)
    print(f"[✓] Generated {len(tx_df)} transactions")

    print("[i] Labelling fraud logically...")
    users_df = label_fraudulent_users(users_df, tx_df)
    tx_df = label_fraudulent_transactions(tx_df, users_df)

    # Generate locations dataframe
    loc_df = pd.DataFrame([
        {"country": c, "currency": CURRENCIES[c], "lat": COUNTRY_LATLON[c][0], "lon": COUNTRY_LATLON[c][1]}
        for c in COUNTRIES
    ])

    # Save to PostgreSQL
    if SAVE_TO_POSTGRES:
        try:
            print("\n[i] Connecting to PostgreSQL...")
            conn = get_postgres_connection()
            print("[✓] Connected to PostgreSQL")
            
            print("[i] Creating tables if they don't exist...")
            create_tables_if_not_exists(conn)
            
            print("[i] Inserting data into PostgreSQL...")
            insert_users_to_postgres(conn, users_df)
            insert_locations_to_postgres(conn, loc_df)
            insert_devices_to_postgres(conn, tx_df)
            card_map = insert_cards_to_postgres(conn, tx_df)
            insert_transactions_to_postgres(conn, tx_df, card_map)
            
            conn.close()
            print("[✓] All data inserted into PostgreSQL successfully!")
        except Exception as e:
            print(f"[✗] Error inserting into PostgreSQL: {e}")
            import traceback
            traceback.print_exc()
            if SAVE_TO_CSV:
                print("[i] Continuing with CSV generation...")

    # Save to CSV files (optional)
    if SAVE_TO_CSV:
        print("\n[i] Writing CSV files...")
        users_df.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)
        tx_df.to_csv(os.path.join(OUT_DIR, "transactions.csv"), index=False)
        loc_df.to_csv(os.path.join(OUT_DIR, "locations.csv"), index=False)
        print(f"[✓] CSV files written to {OUT_DIR}")

    # Summary
    print("\n" + "="*60)
    print("[✓] Dataset Generation Complete!")
    print("="*60)
    print(f"  → Users: {len(users_df)} ({users_df['is_fraud'].sum()} fraudulent)")
    print(f"  → Transactions: {len(tx_df)} ({tx_df['is_fraud_txn'].sum()} fraudulent)")
    print(f"  → Locations: {len(loc_df)} countries")
    if SAVE_TO_POSTGRES:
        print(f"  → PostgreSQL: Data inserted successfully")
    if SAVE_TO_CSV:
        print(f"  → CSV Files: {OUT_DIR}")

if __name__ == "__main__":
    main()
