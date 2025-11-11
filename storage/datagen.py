#!/usr/bin/env python3
"""
Synthetic India-Focused Fraud Dataset Generator (Explainable Version)
---------------------------------------------------------------------
Generates:
  - users.csv
  - transactions.csv
  - locations.csv

Features:
  ✓ 95% domestic (India), 5% international
  ✓ Realistic UPI / NEFT / IMPS modes
  ✓ Fraud logic based on behavior & transaction anomalies
  ✓ Cross-border fraud propagation
  ✓ Explainable fraud reasons
"""

import os
import math
import random
import uuid
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# ---------------- CONFIGURATION ----------------
SEED = 42
OUT_DIR = "./india_fraud_data_explainable"

N_USERS = 20  # Reduced for small dataset
TARGET_TRANSACTIONS = 250  # Target 200-300 rows

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
        # Limit transactions per user to keep total around 200-300
        remaining = TARGET_TRANSACTIONS - len(tx_records)
        if remaining <= 0:
            break
        n_tx = min(np.random.randint(1, 15), remaining)  # Reduced from 10-300 to 1-15
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
    """Mark users as fraudulent logically."""
    stats = tx_df.groupby("sender_id").agg(
        sent_count=("transaction_id", "count"),
        sent_sum=("amount", "sum"),
        sent_mean=("amount", "mean")
    ).reset_index()
    users_df = users_df.merge(stats, left_on="user_id", right_on="sender_id", how="left").fillna(0)

    fraud_flags = []
    fraud_reasons = []
    for _, row in users_df.iterrows():
        risk = 0.0
        reasons = []

        if row["kyc_verified"] == 0 and row["sent_sum"] > 10000:
            risk += 0.4; reasons.append("unverified_high_volume")
        if row["avg_balance"] < 500 and row["sent_sum"] > 50000:
            risk += 0.4; reasons.append("low_balance_high_spend")
        if row["sent_count"] > 500:
            risk += 0.3; reasons.append("unusual_tx_frequency")
        if row["risk_score"] > 0.8:
            risk += 0.3; reasons.append("high_risk_score")
        if row["account_type"] == "merchant" and row["sent_mean"] < 50:
            risk += 0.2; reasons.append("fake_merchant")
        if random.random() < 0.02:
            risk += 0.5; reasons.append("random_noise")

        is_fraud = 1 if risk > 0.5 else 0
        fraud_flags.append(is_fraud)
        fraud_reasons.append(",".join(reasons) if is_fraud else "")

    users_df["is_fraud"] = fraud_flags
    users_df["fraud_reason"] = fraud_reasons
    return users_df

def label_fraudulent_transactions(tx_df, users_df):
    """Logical transaction fraud tagging."""
    fraud_users = set(users_df.loc[users_df["is_fraud"] == 1, "user_id"])
    tx_df["is_fraud_txn"] = 0
    tx_df["fraud_reason"] = ""
    
    MAX_FRAUD_TXN = 10  # Target maximum fraudulent transactions
    fraud_count = 0

    # Shuffle indices to randomize which transactions get flagged
    indices = list(tx_df.index)
    random.shuffle(indices)

    for i in indices:
        row = tx_df.loc[i]
        flag = 0
        reason = ""
        
        # Only flag as fraud if we haven't reached the max yet
        if fraud_count < MAX_FRAUD_TXN:
            # More lenient fraud detection conditions
            if row["sender_id"] in fraud_users and row["receiver_id"] in fraud_users:
                flag = 1; reason = "fraud_to_fraud"
            elif row["is_cross_border"] and row["amount"] > 5000:  # Lowered from 10000
                flag = 1; reason = "cross_border_high_amount"
            elif 9990 <= row["amount"] <= 10010:  # Wider threshold range
                flag = 1; reason = "threshold_pattern"
            elif row["status"] != "success" and row["amount"] > 10000:  # Lowered from 50000
                flag = 1; reason = "failed_high_amount"
            elif row["amount"] > 15000:  # New: very high amount
                flag = 1; reason = "unusually_high_amount"
            elif row["is_cross_border"] and row["amount"] > 2000:  # New: moderate cross-border
                flag = 1; reason = "suspicious_cross_border"
            elif random.random() < 0.08:  # Increased probability to ensure we get enough frauds
                flag = 1; reason = "random_noise"

        if flag == 1:
            fraud_count += 1
        tx_df.at[i, "is_fraud_txn"] = flag
        tx_df.at[i, "fraud_reason"] = reason

    return tx_df

# ---------------- MAIN ----------------
def main():
    set_seed(SEED)
    ensure_outdir(OUT_DIR)

    now = datetime.now(timezone.utc)
    users_start = now - timedelta(days=365 * YEARS_BACK_USERS)
    tx_start = now - timedelta(days=365 * YEARS_TX_COVERAGE)

    print("[i] Generating users...")
    users_df = generate_users(N_USERS, users_start, now)
    print(f"[✓] users.csv base ready ({len(users_df)} users)")

    print("[i] Generating transactions (this may take some time)...")
    tx_df = generate_transactions(users_df, tx_start, now)
    print(f"[✓] transactions.csv base ready ({len(tx_df)} txns)")

    print("[i] Labelling fraud logically...")
    users_df = label_fraudulent_users(users_df, tx_df)
    tx_df = label_fraudulent_transactions(tx_df, users_df)

    print("[i] Writing CSV files...")
    users_df.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)
    tx_df.to_csv(os.path.join(OUT_DIR, "transactions.csv"), index=False)

    loc_df = pd.DataFrame([
        {"country": c, "currency": CURRENCIES[c], "lat": COUNTRY_LATLON[c][0], "lon": COUNTRY_LATLON[c][1]}
        for c in COUNTRIES
    ])
    loc_df.to_csv(os.path.join(OUT_DIR, "locations.csv"), index=False)

    # Summary
    print("\n[✓] Dataset Generated Successfully!")
    print(f"  → users.csv: {len(users_df)} users ({users_df['is_fraud'].sum()} frauds)")
    print(f"  → transactions.csv: {len(tx_df)} transactions ({tx_df['is_fraud_txn'].sum()} fraudulent)")
    print(f"  → locations.csv: {len(loc_df)} countries")
    print(f"  Output Folder: {OUT_DIR}")

if __name__ == "__main__":
    main()
