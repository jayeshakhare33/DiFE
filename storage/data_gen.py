#!/usr/bin/env python3
"""
India-focused synthetic fraud dataset generator for model training.

Why this generator exists:
- preserve the schema used by the current project
- create transactions first and labels second in a logically consistent way
- inject explicit fraud scenarios rather than relying on random noise

Fraud scenarios are based on publicly documented patterns such as:
- account takeover bursts with unusual timing / geography / sudden outbound transfers
- money mule activity where funds are received and rapidly forwarded
- structuring with repeated just-below-threshold payments
- funnel / layering flows with rapid cross-border movement

Notes:
- default profile is sized for practical local experimentation, not maximum scale
- a 1M transaction profile is available, but the current graph feature pipeline in this
  repository is likely to need optimization before 1M edges is comfortable end-to-end
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


SEED = 42
USE_EPOCH_MS = True
YEARS_BACK_USERS = 4
YEARS_TX_COVERAGE = 2

COUNTRY_CURRENCY = {
    "IN": "INR",
    "US": "USD",
    "AE": "AED",
    "SG": "SGD",
    "GB": "GBP",
}

CITY_CATALOG = {
    "IN": [
        ("Mumbai", 19.0760, 72.8777),
        ("Delhi", 28.6139, 77.2090),
        ("Bengaluru", 12.9716, 77.5946),
        ("Hyderabad", 17.3850, 78.4867),
        ("Chennai", 13.0827, 80.2707),
        ("Pune", 18.5204, 73.8567),
        ("Kolkata", 22.5726, 88.3639),
        ("Ahmedabad", 23.0225, 72.5714),
    ],
    "US": [("New York", 40.7128, -74.0060), ("San Francisco", 37.7749, -122.4194)],
    "AE": [("Dubai", 25.2048, 55.2708), ("Abu Dhabi", 24.4539, 54.3773)],
    "SG": [("Singapore", 1.3521, 103.8198)],
    "GB": [("London", 51.5072, -0.1276), ("Manchester", 53.4808, -2.2426)],
}

COUNTRY_WEIGHTS = {
    "IN": 0.965,
    "US": 0.010,
    "AE": 0.010,
    "SG": 0.008,
    "GB": 0.007,
}

SEGMENT_SPECS = {
    "salaried": {
        "weight": 0.42,
        "account_type": "salary",
        "kyc_prob": 0.96,
        "income_range": (25000, 180000),
        "balance_range": (5000, 350000),
        "risk_beta": (2.0, 9.0),
        "devices": (1, 2),
    },
    "student": {
        "weight": 0.08,
        "account_type": "savings",
        "kyc_prob": 0.85,
        "income_range": (4000, 25000),
        "balance_range": (500, 40000),
        "risk_beta": (2.5, 8.5),
        "devices": (1, 2),
    },
    "gig_worker": {
        "weight": 0.10,
        "account_type": "savings",
        "kyc_prob": 0.90,
        "income_range": (10000, 80000),
        "balance_range": (1500, 90000),
        "risk_beta": (2.5, 8.0),
        "devices": (1, 2),
    },
    "family_remitter": {
        "weight": 0.07,
        "account_type": "savings",
        "kyc_prob": 0.94,
        "income_range": (20000, 150000),
        "balance_range": (4000, 200000),
        "risk_beta": (2.0, 8.0),
        "devices": (1, 2),
    },
    "small_merchant": {
        "weight": 0.14,
        "account_type": "merchant",
        "kyc_prob": 0.92,
        "income_range": (40000, 400000),
        "balance_range": (10000, 500000),
        "risk_beta": (2.0, 7.0),
        "devices": (1, 3),
    },
    "sme_business": {
        "weight": 0.08,
        "account_type": "current",
        "kyc_prob": 0.98,
        "income_range": (100000, 1200000),
        "balance_range": (50000, 2500000),
        "risk_beta": (1.8, 6.0),
        "devices": (1, 3),
    },
    "high_value": {
        "weight": 0.03,
        "account_type": "savings",
        "kyc_prob": 0.99,
        "income_range": (150000, 1500000),
        "balance_range": (150000, 5000000),
        "risk_beta": (1.5, 7.0),
        "devices": (1, 3),
    },
    "mule_candidate": {
        "weight": 0.08,
        "account_type": "wallet",
        "kyc_prob": 0.55,
        "income_range": (5000, 50000),
        "balance_range": (500, 25000),
        "risk_beta": (3.2, 4.8),
        "devices": (1, 2),
    },
}

MODE_SET = ["upi", "wallet", "card", "net_banking", "neft", "imps"]
PROFILE_SIZES = {
    "basic": {"n_users": 8000, "n_transactions": 100000, "fraud_rate": 0.022},
    "medium": {"n_users": 16000, "n_transactions": 250000, "fraud_rate": 0.020},
    "million": {"n_users": 60000, "n_transactions": 1000000, "fraud_rate": 0.018},
}


@dataclass
class GeneratorConfig:
    output_dir: Path
    n_users: int
    n_transactions: int
    fraud_rate: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate logical synthetic fraud dataset")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SIZES.keys()),
        default="basic",
        help="Dataset size profile. 'basic' is the default for practical local training.",
    )
    parser.add_argument("--users", type=int, default=None, help="Override number of users")
    parser.add_argument("--transactions", type=int, default=None, help="Override number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=None, help="Override target fraud transaction ratio")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "india_fraud_data_explainable"),
        help="Directory where users.csv, transactions.csv and locations.csv will be written",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def dt_to_epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def epoch_or_dt(dt: datetime):
    return dt_to_epoch_ms(dt) if USE_EPOCH_MS else dt.isoformat()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def choose_weighted_key(mapping: Dict[str, float], rng: random.Random) -> str:
    keys = list(mapping.keys())
    weights = list(mapping.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def pick_city(country: str, rng: random.Random) -> Tuple[str, float, float]:
    return rng.choice(CITY_CATALOG[country])


def make_device_ids(user_id: str, device_count: int) -> List[str]:
    return [f"d_{user_id}_{idx}_{uuid.uuid4().hex[:10]}" for idx in range(device_count)]


def normal_hours(segment: str, rng: random.Random) -> int:
    if segment in {"salaried", "sme_business", "small_merchant"}:
        weights = [0.02, 0.05, 0.23, 0.30, 0.25, 0.15]
        ranges = [(0, 5), (6, 8), (9, 12), (13, 17), (18, 21), (22, 23)]
    else:
        weights = [0.03, 0.09, 0.18, 0.26, 0.27, 0.17]
        ranges = [(0, 5), (6, 8), (9, 12), (13, 17), (18, 21), (22, 23)]

    hour_range = rng.choices(ranges, weights=weights, k=1)[0]
    return rng.randint(hour_range[0], hour_range[1])


def off_hours(rng: random.Random) -> int:
    return rng.choice([0, 1, 2, 3, 4, 5, 23])


def sample_timestamp(start: datetime, end: datetime, rng: random.Random, hour_sampler) -> datetime:
    day_span = (end.date() - start.date()).days
    chosen_day = start + timedelta(days=rng.randint(0, max(day_span, 1)))
    hour = hour_sampler()
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return chosen_day.replace(hour=hour, minute=minute, second=second, microsecond=rng.randint(0, 999999))


def users_dataframe(config: GeneratorConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    rng = random.Random(config.seed)
    now = datetime.now(timezone.utc)
    user_start = now - timedelta(days=365 * YEARS_BACK_USERS)

    rows: List[Dict] = []
    devices: Dict[str, List[str]] = {}

    segment_weights = {key: spec["weight"] for key, spec in SEGMENT_SPECS.items()}

    for idx in range(config.n_users):
        user_id = f"U{idx:07d}"
        country = choose_weighted_key(COUNTRY_WEIGHTS, rng)
        city, lat, lon = pick_city(country, rng)
        segment = choose_weighted_key(segment_weights, rng)
        spec = SEGMENT_SPECS[segment]

        created = user_start + timedelta(
            seconds=rng.randint(0, int((now - user_start).total_seconds()))
        )
        monthly_income = round(rng.uniform(*spec["income_range"]), 2)
        avg_balance = round(rng.uniform(*spec["balance_range"]), 2)
        device_count = rng.randint(*spec["devices"])
        kyc_verified = 1 if rng.random() < spec["kyc_prob"] else 0
        risk_beta_a, risk_beta_b = spec["risk_beta"]
        risk_score = min(max(np.random.beta(risk_beta_a, risk_beta_b), 0.01), 0.99)
        device_type = rng.choices(
            ["android", "ios", "desktop", "feature_phone"],
            weights=[0.47, 0.23, 0.22, 0.08],
            k=1,
        )[0]

        devices[user_id] = make_device_ids(user_id, device_count)

        rows.append(
            {
                "user_id": user_id,
                "account_creation_time": epoch_or_dt(created),
                "country": country,
                "home_city": city,
                "home_lat": round(lat, 6),
                "home_lon": round(lon, 6),
                "risk_score": round(float(risk_score), 4),
                "device_type": device_type,
                "kyc_verified": kyc_verified,
                "account_type": spec["account_type"],
                "segment": segment,
                "monthly_income": monthly_income,
                "avg_balance": avg_balance,
                "is_fraud": 0,
                "fraud_reason": "",
                "fraud_role": "none",
            }
        )

    return pd.DataFrame(rows), devices


def build_contacts(users_df: pd.DataFrame, rng: random.Random) -> Dict[str, List[str]]:
    users_by_country: Dict[str, List[str]] = defaultdict(list)
    merchants_by_country: Dict[str, List[str]] = defaultdict(list)
    businesses_by_country: Dict[str, List[str]] = defaultdict(list)

    for row in users_df.itertuples(index=False):
        users_by_country[row.country].append(row.user_id)
        if row.account_type == "merchant":
            merchants_by_country[row.country].append(row.user_id)
        if row.account_type == "current":
            businesses_by_country[row.country].append(row.user_id)

    contacts: Dict[str, List[str]] = {}
    for row in users_df.itertuples(index=False):
        pool = [uid for uid in users_by_country[row.country] if uid != row.user_id]
        if row.account_type == "merchant":
            merchant_pool = [uid for uid in businesses_by_country[row.country] if uid != row.user_id]
            peer_pool = merchant_pool + pool
        elif row.account_type == "current":
            peer_pool = merchants_by_country[row.country] + pool
        else:
            peer_pool = merchants_by_country[row.country] + pool

        if not peer_pool:
            peer_pool = [uid for uid in users_df["user_id"].tolist() if uid != row.user_id]

        sample_size = min(len(peer_pool), rng.randint(8, 24))
        contacts[row.user_id] = rng.sample(peer_pool, sample_size) if sample_size else []

    return contacts


def choose_legit_tx_kind(segment: str, rng: random.Random) -> str:
    if segment == "salaried":
        return rng.choices(
            ["p2p", "merchant_payment", "bill_pay", "family_support"],
            weights=[0.46, 0.34, 0.12, 0.08],
            k=1,
        )[0]
    if segment == "student":
        return rng.choices(
            ["p2p", "merchant_payment", "wallet_topup", "bill_pay"],
            weights=[0.44, 0.26, 0.18, 0.12],
            k=1,
        )[0]
    if segment == "gig_worker":
        return rng.choices(
            ["p2p", "merchant_payment", "bill_pay", "family_support"],
            weights=[0.38, 0.28, 0.14, 0.20],
            k=1,
        )[0]
    if segment == "family_remitter":
        return rng.choices(
            ["p2p", "merchant_payment", "bill_pay", "remittance"],
            weights=[0.34, 0.22, 0.08, 0.36],
            k=1,
        )[0]
    if segment == "small_merchant":
        return rng.choices(
            ["supplier_payment", "refund", "business_expense"],
            weights=[0.58, 0.18, 0.24],
            k=1,
        )[0]
    if segment == "sme_business":
        return rng.choices(
            ["supplier_payment", "payroll", "business_expense"],
            weights=[0.46, 0.32, 0.22],
            k=1,
        )[0]
    if segment == "high_value":
        return rng.choices(
            ["p2p", "merchant_payment", "remittance", "business_expense"],
            weights=[0.28, 0.22, 0.30, 0.20],
            k=1,
        )[0]
    return rng.choices(
        ["p2p", "merchant_payment", "wallet_topup", "family_support"],
        weights=[0.44, 0.24, 0.14, 0.18],
        k=1,
    )[0]


def choose_legit_receiver(
    sender: pd.Series,
    tx_kind: str,
    contacts: Dict[str, List[str]],
    users_df: pd.DataFrame,
    rng: random.Random,
) -> pd.Series:
    same_country = users_df[users_df["country"] == sender["country"]]
    foreign = users_df[users_df["country"] != sender["country"]]
    merchants = same_country[same_country["account_type"] == "merchant"]
    businesses = same_country[same_country["account_type"] == "current"]
    consumers = same_country[same_country["account_type"].isin(["salary", "savings", "wallet"])]

    preferred_ids = contacts.get(sender["user_id"], [])
    preferred = users_df[users_df["user_id"].isin(preferred_ids)]

    def pick_from(df: pd.DataFrame, fallback: pd.DataFrame) -> pd.Series:
        if not df.empty:
            return df.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        if not fallback.empty:
            return fallback.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        return users_df[users_df["user_id"] != sender["user_id"]].sample(
            1, random_state=rng.randint(0, 10**9)
        ).iloc[0]

    if tx_kind in {"merchant_payment", "bill_pay"}:
        return pick_from(merchants[merchants["user_id"] != sender["user_id"]], preferred)
    if tx_kind in {"supplier_payment", "business_expense"}:
        supplier_pool = pd.concat([merchants, businesses]).drop_duplicates("user_id")
        supplier_pool = supplier_pool[supplier_pool["user_id"] != sender["user_id"]]
        return pick_from(supplier_pool, preferred)
    if tx_kind == "payroll":
        payroll_pool = consumers[consumers["user_id"] != sender["user_id"]]
        return pick_from(payroll_pool, preferred)
    if tx_kind == "refund":
        refund_pool = consumers[consumers["user_id"] != sender["user_id"]]
        return pick_from(refund_pool, preferred)
    if tx_kind == "remittance":
        return pick_from(foreign[foreign["user_id"] != sender["user_id"]], preferred)

    p2p_pool = preferred[preferred["user_id"] != sender["user_id"]]
    return pick_from(p2p_pool, same_country[same_country["user_id"] != sender["user_id"]])


def sample_legit_amount(sender: pd.Series, receiver: pd.Series, tx_kind: str, rng: random.Random) -> float:
    monthly_income = float(sender["monthly_income"])
    avg_balance = float(sender["avg_balance"])

    if tx_kind == "merchant_payment":
        value = np.random.lognormal(mean=6.4, sigma=0.7)
        return round(min(max(value, 50.0), min(avg_balance * 0.25 + 10000.0, 45000.0)), 2)
    if tx_kind == "bill_pay":
        value = np.random.lognormal(mean=7.0, sigma=0.5)
        return round(min(max(value, 200.0), 25000.0), 2)
    if tx_kind == "wallet_topup":
        return round(min(max(np.random.lognormal(mean=6.0, sigma=0.55), 100.0), 8000.0), 2)
    if tx_kind == "family_support":
        return round(min(max(np.random.lognormal(mean=7.2, sigma=0.8), 500.0), max(10000.0, monthly_income * 0.8)), 2)
    if tx_kind == "supplier_payment":
        return round(min(max(np.random.lognormal(mean=9.1, sigma=0.9), 3000.0), max(25000.0, avg_balance * 0.9)), 2)
    if tx_kind == "payroll":
        base = float(receiver.get("monthly_income", monthly_income * 0.55))
        return round(max(base * rng.uniform(0.75, 1.10), 12000.0), 2)
    if tx_kind == "refund":
        return round(min(max(np.random.lognormal(mean=6.3, sigma=0.65), 50.0), 20000.0), 2)
    if tx_kind == "remittance":
        return round(min(max(np.random.lognormal(mean=8.8, sigma=0.85), 5000.0), max(40000.0, avg_balance * 0.8)), 2)
    if tx_kind == "business_expense":
        return round(min(max(np.random.lognormal(mean=8.2, sigma=0.9), 1500.0), max(30000.0, avg_balance * 0.6)), 2)

    return round(min(max(np.random.lognormal(mean=6.6, sigma=0.75), 50.0), max(12000.0, avg_balance * 0.3)), 2)


def choose_legit_mode(tx_kind: str, rng: random.Random) -> str:
    if tx_kind in {"merchant_payment", "wallet_topup"}:
        return rng.choices(["upi", "card", "wallet", "imps"], weights=[0.46, 0.26, 0.16, 0.12], k=1)[0]
    if tx_kind in {"bill_pay", "family_support", "p2p"}:
        return rng.choices(["upi", "imps", "wallet", "net_banking"], weights=[0.55, 0.22, 0.11, 0.12], k=1)[0]
    if tx_kind in {"supplier_payment", "payroll", "business_expense", "remittance"}:
        return rng.choices(["neft", "net_banking", "imps", "card"], weights=[0.43, 0.31, 0.20, 0.06], k=1)[0]
    return rng.choice(MODE_SET)


def choose_legit_status(tx_kind: str, rng: random.Random) -> str:
    if tx_kind in {"merchant_payment", "wallet_topup"}:
        return rng.choices(["success", "failed", "reversed"], weights=[0.975, 0.020, 0.005], k=1)[0]
    if tx_kind in {"supplier_payment", "payroll"}:
        return rng.choices(["success", "failed", "reversed"], weights=[0.985, 0.010, 0.005], k=1)[0]
    return rng.choices(["success", "failed", "reversed"], weights=[0.978, 0.017, 0.005], k=1)[0]


def geo_distance(sender: pd.Series, receiver: pd.Series) -> float:
    return round(
        haversine_km(
            float(sender["home_lat"]),
            float(sender["home_lon"]),
            float(receiver["home_lat"]),
            float(receiver["home_lon"]),
        ),
        2,
    )


def make_record(
    tx_id: int,
    sender: pd.Series,
    receiver: pd.Series,
    amount: float,
    tx_time: datetime,
    mode: str,
    status: str,
    device_id: str,
    is_fraud: int,
    fraud_reason: Sequence[str],
) -> Dict:
    fraud_reason_text = ",".join(sorted(set(fraud_reason))) if is_fraud else ""
    return {
        "transaction_id": f"T{tx_id:012d}",
        "sender_id": sender["user_id"],
        "receiver_id": receiver["user_id"],
        "amount": round(float(amount), 2),
        "timestamp": epoch_or_dt(tx_time),
        "mode": mode,
        "status": status,
        "currency": COUNTRY_CURRENCY[str(sender["country"])],
        "sender_country": sender["country"],
        "receiver_country": receiver["country"],
        "is_cross_border": bool(sender["country"] != receiver["country"]),
        "device_id": device_id,
        "geo_distance_km": geo_distance(sender, receiver),
        "is_fraud_txn": int(is_fraud),
        "fraud_reason": fraud_reason_text,
    }


def legitimate_transactions(
    users_df: pd.DataFrame,
    contacts: Dict[str, List[str]],
    target_count: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    tx_start = now - timedelta(days=365 * YEARS_TX_COVERAGE)

    rows = users_df.to_dict("records")
    user_lookup = {row["user_id"]: row for row in rows}
    senders = rows

    activity_weights = []
    for row in rows:
        segment = row["segment"]
        base = {
            "salaried": 1.2,
            "student": 0.8,
            "gig_worker": 1.1,
            "family_remitter": 1.0,
            "small_merchant": 2.1,
            "sme_business": 2.6,
            "high_value": 1.0,
            "mule_candidate": 0.7,
        }[segment]
        activity_weights.append(base)

    transactions: List[Dict] = []
    tx_id = 0

    while len(transactions) < target_count:
        sender = rng.choices(senders, weights=activity_weights, k=1)[0]
        tx_kind = choose_legit_tx_kind(sender["segment"], rng)
        receiver = choose_legit_receiver(pd.Series(sender), tx_kind, contacts, users_df, rng)

        if receiver["user_id"] == sender["user_id"]:
            continue

        amount = sample_legit_amount(pd.Series(sender), receiver, tx_kind, rng)
        status = choose_legit_status(tx_kind, rng)
        mode = choose_legit_mode(tx_kind, rng)
        device_id = rng.choice(sender["devices"])
        tx_time = sample_timestamp(
            tx_start,
            now,
            rng,
            lambda seg=sender["segment"]: normal_hours(seg, rng),
        )

        transactions.append(
            make_record(
                tx_id=tx_id,
                sender=pd.Series(sender),
                receiver=receiver,
                amount=amount,
                tx_time=tx_time,
                mode=mode,
                status=status,
                device_id=device_id,
                is_fraud=0,
                fraud_reason=[],
            )
        )
        tx_id += 1

    return transactions


def select_role_pools(users_df: pd.DataFrame, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed + 99)

    new_low_kyc = users_df[
        (users_df["segment"].isin(["mule_candidate", "student", "gig_worker"]))
        & (users_df["kyc_verified"] == 0)
    ]["user_id"].tolist()
    if len(new_low_kyc) < 20:
        new_low_kyc = users_df[users_df["segment"] == "mule_candidate"]["user_id"].tolist()

    foreign_business = users_df[
        (users_df["country"] != "IN") & (users_df["account_type"].isin(["merchant", "current"]))
    ]["user_id"].tolist()
    if len(foreign_business) < 8:
        foreign_business = users_df[users_df["country"] != "IN"]["user_id"].tolist()

    mature_high_balance = users_df[
        (users_df["kyc_verified"] == 1)
        & (users_df["avg_balance"] >= users_df["avg_balance"].quantile(0.65))
        & (users_df["segment"].isin(["salaried", "high_value", "family_remitter"]))
    ]["user_id"].tolist()

    merchant_candidates = users_df[
        users_df["account_type"].isin(["merchant", "current"])
    ]["user_id"].tolist()

    return {
        "mules": rng.sample(new_low_kyc, min(len(new_low_kyc), max(20, len(users_df) // 250))),
        "foreign_beneficiaries": rng.sample(
            foreign_business, min(len(foreign_business), max(10, len(users_df) // 400))
        ),
        "victims": rng.sample(
            mature_high_balance, min(len(mature_high_balance), max(40, len(users_df) // 120))
        ),
        "fraud_merchants": rng.sample(
            merchant_candidates, min(len(merchant_candidates), max(8, len(users_df) // 700))
        ),
    }


def quick_user_lookup(users_df: pd.DataFrame) -> Dict[str, Dict]:
    return {row["user_id"]: row for row in users_df.to_dict("records")}


def next_tx_id(records: List[Dict]) -> int:
    return len(records)


def add_account_takeover(
    records: List[Dict],
    users: Dict[str, Dict],
    role_pools: Dict[str, List[str]],
    target_count: int,
    seed: int,
) -> int:
    rng = random.Random(seed + 201)
    now = datetime.now(timezone.utc)
    tx_start = now - timedelta(days=365 * YEARS_TX_COVERAGE)
    generated = 0

    if not role_pools["victims"] or not role_pools["mules"]:
        return 0

    while generated < target_count:
        victim = users[rng.choice(role_pools["victims"])]
        new_beneficiaries = rng.sample(role_pools["mules"], k=min(len(role_pools["mules"]), rng.randint(2, 4)))
        base_time = sample_timestamp(tx_start, now, rng, lambda: off_hours(rng))
        fraud_device = f"compromised_{victim['user_id']}_{uuid.uuid4().hex[:10]}"
        available = max(float(victim["avg_balance"]) * rng.uniform(0.35, 0.70), 15000.0)

        for beneficiary_id in new_beneficiaries:
            beneficiary = users[beneficiary_id]
            amount = min(available / len(new_beneficiaries) * rng.uniform(0.75, 1.20), available)
            attempt_count = rng.randint(2, 5)
            for attempt_idx in range(attempt_count):
                tx_time = base_time + timedelta(minutes=attempt_idx * rng.randint(1, 6))
                mode = rng.choices(["imps", "upi", "net_banking"], weights=[0.45, 0.40, 0.15], k=1)[0]
                status = "failed" if attempt_idx == 0 and rng.random() < 0.55 else "success"
                reasons = ["account_takeover", "new_device", "off_hours", "burst_transfer"]
                if beneficiary["country"] != victim["country"]:
                    reasons.extend(["cross_border", "geo_mismatch"])
                if amount > victim["avg_balance"] * 0.20:
                    reasons.append("high_value_outlier")

                records.append(
                    make_record(
                        tx_id=next_tx_id(records),
                        sender=pd.Series(victim),
                        receiver=pd.Series(beneficiary),
                        amount=round(amount * rng.uniform(0.25, 0.45), 2),
                        tx_time=tx_time,
                        mode=mode,
                        status=status,
                        device_id=fraud_device,
                        is_fraud=1,
                        fraud_reason=reasons,
                    )
                )
                generated += 1
                if generated >= target_count:
                    return generated

    return generated


def add_money_mule_layering(
    records: List[Dict],
    users: Dict[str, Dict],
    role_pools: Dict[str, List[str]],
    target_count: int,
    seed: int,
) -> int:
    rng = random.Random(seed + 301)
    now = datetime.now(timezone.utc)
    tx_start = now - timedelta(days=365 * YEARS_TX_COVERAGE)
    generated = 0
    victims = role_pools["victims"]
    mules = role_pools["mules"]
    beneficiaries = role_pools["foreign_beneficiaries"] or role_pools["mules"]

    if not victims or not mules or not beneficiaries:
        return 0

    while generated < target_count:
        mule = users[rng.choice(mules)]
        incoming_sources = rng.sample(victims, k=min(len(victims), rng.randint(2, 4)))
        outgoing_beneficiaries = rng.sample(beneficiaries, k=min(len(beneficiaries), rng.randint(2, 4)))
        base_time = sample_timestamp(tx_start, now, rng, lambda: normal_hours("mule_candidate", rng))
        shared_device = f"mule_{mule['user_id']}_{uuid.uuid4().hex[:10]}"

        pooled_amount = 0.0
        for source_id in incoming_sources:
            source = users[source_id]
            amount = max(np.random.lognormal(mean=8.5, sigma=0.75), 3000.0)
            pooled_amount += amount
            reasons = ["money_mule", "fan_in", "unexpected_inflow"]
            records.append(
                make_record(
                    tx_id=next_tx_id(records),
                    sender=pd.Series(source),
                    receiver=pd.Series(mule),
                    amount=amount,
                    tx_time=base_time + timedelta(minutes=rng.randint(0, 25)),
                    mode=rng.choices(["imps", "upi", "wallet"], weights=[0.45, 0.45, 0.10], k=1)[0],
                    status="success",
                    device_id=rng.choice(source["devices"]),
                    is_fraud=1,
                    fraud_reason=reasons,
                )
            )
            generated += 1
            if generated >= target_count:
                return generated

        retained = pooled_amount * rng.uniform(0.02, 0.06)
        distributable = max(pooled_amount - retained, pooled_amount * 0.85)
        split_weights = np.random.dirichlet(np.ones(len(outgoing_beneficiaries)))

        for idx, beneficiary_id in enumerate(outgoing_beneficiaries):
            beneficiary = users[beneficiary_id]
            reasons = ["money_mule", "rapid_turnover", "fan_out", "commission_skimming"]
            if beneficiary["country"] != mule["country"]:
                reasons.extend(["cross_border", "layering"])
            records.append(
                make_record(
                    tx_id=next_tx_id(records),
                    sender=pd.Series(mule),
                    receiver=pd.Series(beneficiary),
                    amount=round(distributable * float(split_weights[idx]), 2),
                    tx_time=base_time + timedelta(minutes=20 + idx * rng.randint(4, 18)),
                    mode=rng.choices(["imps", "neft", "net_banking"], weights=[0.35, 0.35, 0.30], k=1)[0],
                    status="success",
                    device_id=shared_device,
                    is_fraud=1,
                    fraud_reason=reasons,
                )
            )
            generated += 1
            if generated >= target_count:
                return generated

    return generated


def add_structuring(
    records: List[Dict],
    users: Dict[str, Dict],
    role_pools: Dict[str, List[str]],
    target_count: int,
    seed: int,
) -> int:
    rng = random.Random(seed + 401)
    now = datetime.now(timezone.utc)
    tx_start = now - timedelta(days=365 * YEARS_TX_COVERAGE)
    generated = 0

    senders = role_pools["mules"] + role_pools["fraud_merchants"]
    receivers = role_pools["foreign_beneficiaries"] + role_pools["mules"]
    if not senders or not receivers:
        return 0

    thresholds = [9999.0, 19999.0, 49999.0]
    while generated < target_count:
        sender = users[rng.choice(senders)]
        receiver = users[rng.choice(receivers)]
        if sender["user_id"] == receiver["user_id"]:
            continue

        series_count = rng.randint(3, 6)
        start_time = sample_timestamp(tx_start, now, rng, lambda: normal_hours(sender["segment"], rng))
        for offset in range(series_count):
            threshold = rng.choice(thresholds)
            amount = round(threshold - rng.uniform(0.01, 199.99), 2)
            reasons = ["structuring", "near_threshold", "repeat_pattern"]
            if receiver["country"] != sender["country"]:
                reasons.append("cross_border")
            records.append(
                make_record(
                    tx_id=next_tx_id(records),
                    sender=pd.Series(sender),
                    receiver=pd.Series(receiver),
                    amount=amount,
                    tx_time=start_time + timedelta(hours=offset * rng.randint(3, 12)),
                    mode=rng.choices(["imps", "neft", "net_banking"], weights=[0.30, 0.40, 0.30], k=1)[0],
                    status="success",
                    device_id=rng.choice(sender["devices"]),
                    is_fraud=1,
                    fraud_reason=reasons,
                )
            )
            generated += 1
            if generated >= target_count:
                return generated

    return generated


def add_funnel_cross_border(
    records: List[Dict],
    users: Dict[str, Dict],
    role_pools: Dict[str, List[str]],
    target_count: int,
    seed: int,
) -> int:
    rng = random.Random(seed + 501)
    now = datetime.now(timezone.utc)
    tx_start = now - timedelta(days=365 * YEARS_TX_COVERAGE)
    generated = 0

    victims = role_pools["victims"]
    mules = role_pools["mules"]
    foreign_beneficiaries = role_pools["foreign_beneficiaries"]
    if not victims or not mules or not foreign_beneficiaries:
        return 0

    while generated < target_count:
        aggregator = users[rng.choice(mules)]
        beneficiary = users[rng.choice(foreign_beneficiaries)]
        inflow_sources = rng.sample(victims, k=min(len(victims), rng.randint(3, 5)))
        base_time = sample_timestamp(tx_start, now, rng, lambda: normal_hours("mule_candidate", rng))

        total_inflow = 0.0
        for source_id in inflow_sources:
            source = users[source_id]
            amount = max(np.random.lognormal(mean=8.9, sigma=0.7), 5000.0)
            total_inflow += amount
            records.append(
                make_record(
                    tx_id=next_tx_id(records),
                    sender=pd.Series(source),
                    receiver=pd.Series(aggregator),
                    amount=amount,
                    tx_time=base_time + timedelta(minutes=rng.randint(0, 40)),
                    mode=rng.choices(["imps", "upi", "neft"], weights=[0.40, 0.35, 0.25], k=1)[0],
                    status="success",
                    device_id=rng.choice(source["devices"]),
                    is_fraud=1,
                    fraud_reason=["funnel_account", "fan_in", "geo_dispersion"],
                )
            )
            generated += 1
            if generated >= target_count:
                return generated

        records.append(
            make_record(
                tx_id=next_tx_id(records),
                sender=pd.Series(aggregator),
                receiver=pd.Series(beneficiary),
                amount=round(total_inflow * rng.uniform(0.90, 0.97), 2),
                tx_time=base_time + timedelta(minutes=rng.randint(45, 180)),
                mode=rng.choices(["neft", "net_banking", "imps"], weights=[0.40, 0.40, 0.20], k=1)[0],
                status="success",
                device_id=f"funnel_{aggregator['user_id']}_{uuid.uuid4().hex[:10]}",
                is_fraud=1,
                fraud_reason=["funnel_account", "rapid_turnover", "cross_border", "geo_dispersion", "layering"],
            )
        )
        generated += 1

    return generated


def inject_fraud(records: List[Dict], users_df: pd.DataFrame, fraud_target: int, seed: int) -> pd.DataFrame:
    role_pools = select_role_pools(users_df, seed)
    users = quick_user_lookup(users_df)

    allocation = {
        "account_takeover": int(fraud_target * 0.28),
        "money_mule": int(fraud_target * 0.32),
        "structuring": int(fraud_target * 0.18),
        "funnel": int(fraud_target * 0.22),
    }
    allocation["funnel"] += fraud_target - sum(allocation.values())

    add_account_takeover(records, users, role_pools, allocation["account_takeover"], seed)
    add_money_mule_layering(records, users, role_pools, allocation["money_mule"], seed)
    add_structuring(records, users, role_pools, allocation["structuring"], seed)
    add_funnel_cross_border(records, users, role_pools, allocation["funnel"], seed)

    fraud_users = set(role_pools["mules"]) | set(role_pools["foreign_beneficiaries"]) | set(role_pools["fraud_merchants"])
    compromised = set(role_pools["victims"])
    users_df = users_df.copy()
    users_df.loc[users_df["user_id"].isin(role_pools["mules"]), "fraud_role"] = "money_mule"
    users_df.loc[users_df["user_id"].isin(role_pools["foreign_beneficiaries"]), "fraud_role"] = "fraud_beneficiary"
    users_df.loc[users_df["user_id"].isin(role_pools["fraud_merchants"]), "fraud_role"] = "fraud_merchant"
    users_df.loc[users_df["user_id"].isin(compromised - fraud_users), "fraud_role"] = "compromised_victim"
    users_df.loc[users_df["user_id"].isin(fraud_users), "is_fraud"] = 1
    users_df.loc[users_df["user_id"].isin(fraud_users), "fraud_reason"] = users_df.loc[
        users_df["user_id"].isin(fraud_users), "fraud_role"
    ]

    tx_df = pd.DataFrame(records)
    if len(tx_df) > 0:
        tx_df = tx_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        tx_df["transaction_id"] = [f"T{i:012d}" for i in range(len(tx_df))]
        tx_df["timestamp_dt"] = pd.to_datetime(tx_df["timestamp"], unit="ms", utc=True)
        tx_df["hour"] = tx_df["timestamp_dt"].dt.hour
    return users_df, tx_df


def finalize_users(users_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
    users_df = users_df.drop(columns=["devices"], errors="ignore").copy()
    sender_stats = tx_df.groupby("sender_id").agg(
        sent_count=("transaction_id", "count"),
        sent_sum=("amount", "sum"),
        sent_mean=("amount", "mean"),
        sent_max=("amount", "max"),
        sent_std=("amount", "std"),
        sent_fraud_count=("is_fraud_txn", "sum"),
    ).reset_index()

    receiver_stats = tx_df.groupby("receiver_id").agg(
        received_count=("transaction_id", "count"),
        received_sum=("amount", "sum"),
        received_fraud_count=("is_fraud_txn", "sum"),
    ).reset_index()

    users_df = users_df.merge(sender_stats, left_on="user_id", right_on="sender_id", how="left")
    users_df = users_df.merge(receiver_stats, left_on="user_id", right_on="receiver_id", how="left")
    users_df = users_df.fillna(
        {
            "sender_id": "",
            "sent_count": 0,
            "sent_sum": 0.0,
            "sent_mean": 0.0,
            "sent_max": 0.0,
            "sent_std": 0.0,
            "sent_fraud_count": 0,
            "received_count": 0,
            "received_sum": 0.0,
            "received_fraud_count": 0,
        }
    )

    fraud_involved = (
        (users_df["sent_fraud_count"] > 0) | (users_df["received_fraud_count"] > 0)
    ) & (users_df["fraud_role"] == "none")
    users_df.loc[fraud_involved, "fraud_role"] = "fraud_adjacent"
    users_df.loc[users_df["sender_id"] == "", "sender_id"] = users_df["user_id"]

    cols = [
        "user_id",
        "account_creation_time",
        "country",
        "home_city",
        "home_lat",
        "home_lon",
        "risk_score",
        "device_type",
        "kyc_verified",
        "account_type",
        "segment",
        "monthly_income",
        "avg_balance",
        "sender_id",
        "sent_count",
        "sent_sum",
        "sent_mean",
        "sent_max",
        "sent_std",
        "received_count",
        "received_sum",
        "is_fraud",
        "fraud_reason",
        "fraud_role",
    ]
    return users_df[cols]


def locations_df() -> pd.DataFrame:
    rows = []
    for country, cities in CITY_CATALOG.items():
        lat = round(sum(city[1] for city in cities) / len(cities), 4)
        lon = round(sum(city[2] for city in cities) / len(cities), 4)
        rows.append(
            {
                "country": country,
                "currency": COUNTRY_CURRENCY[country],
                "lat": lat,
                "lon": lon,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(output_dir: Path, users_df: pd.DataFrame, tx_df: pd.DataFrame, loc_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    users_df.to_csv(output_dir / "users.csv", index=False)
    tx_df.to_csv(output_dir / "transactions.csv", index=False)
    loc_df.to_csv(output_dir / "locations.csv", index=False)

    summary = {
        "users": int(len(users_df)),
        "transactions": int(len(tx_df)),
        "fraud_transactions": int(tx_df["is_fraud_txn"].sum()),
        "fraud_transaction_rate": float(tx_df["is_fraud_txn"].mean()),
        "fraud_users": int(users_df["is_fraud"].sum()),
        "cross_border_transactions": int(tx_df["is_cross_border"].sum()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def attach_devices(users_df: pd.DataFrame, devices: Dict[str, List[str]]) -> pd.DataFrame:
    users_df = users_df.copy()
    users_df["devices"] = users_df["user_id"].map(devices)
    return users_df


def main() -> None:
    args = parse_args()
    profile = PROFILE_SIZES[args.profile]
    config = GeneratorConfig(
        output_dir=Path(args.output_dir),
        n_users=args.users or profile["n_users"],
        n_transactions=args.transactions or profile["n_transactions"],
        fraud_rate=args.fraud_rate or profile["fraud_rate"],
        seed=args.seed,
    )

    set_seed(config.seed)
    print(f"[i] Generating dataset profile={args.profile} users={config.n_users} tx={config.n_transactions}")

    users_df, devices = users_dataframe(config)
    users_df = attach_devices(users_df, devices)
    contacts = build_contacts(users_df.drop(columns=["devices"]), random.Random(config.seed + 11))

    fraud_target = int(round(config.n_transactions * config.fraud_rate))
    legit_target = max(config.n_transactions - fraud_target, 1)
    print(f"[i] Creating {legit_target} legitimate transactions and ~{fraud_target} fraud transactions")

    records = legitimate_transactions(
        users_df=users_df,
        contacts=contacts,
        target_count=legit_target,
        seed=config.seed + 21,
    )

    users_df_labeled, tx_df = inject_fraud(records, users_df, fraud_target, config.seed + 31)
    users_df_final = finalize_users(users_df_labeled, tx_df)
    loc_df = locations_df()

    write_outputs(config.output_dir, users_df_final, tx_df, loc_df)

    fraud_tx = int(tx_df["is_fraud_txn"].sum())
    print("[ok] Dataset generation complete")
    print(f"     output: {config.output_dir}")
    print(f"     users: {len(users_df_final)}")
    print(f"     transactions: {len(tx_df)}")
    print(f"     fraud transactions: {fraud_tx} ({fraud_tx / max(len(tx_df), 1):.2%})")
    print(f"     fraud users: {int(users_df_final['is_fraud'].sum())}")


if __name__ == "__main__":
    main()
