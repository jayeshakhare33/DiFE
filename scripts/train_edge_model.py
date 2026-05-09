#!/usr/bin/env python3
"""
Standalone edge-classification trainer for transaction fraud detection.

This script is intentionally independent from the eventual deployment target.
It produces model artifacts that can later be loaded by an API running on EC2
or any other environment, regardless of where transactions and features live.

Training is CPU-only and expects precomputed parquet features from data/features.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gnn_training.edge_model import GraphSAGEEdgeClassifier
from graph_processing.graph_builder import GraphBuilder


LOGGER = logging.getLogger("train_edge_model")
EDGE_ETYPE = ("user", "transaction", "user")
DEFAULT_LEAKY_NODE_FEATURES = [
    "connected_to_fraud_count",
    "fraud_propagation_score",
    "distance_to_nearest_fraud",
    "common_neighbors_with_frauds",
    "fraud_cluster_membership",
]


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE edge fraud model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--transactions-path", default=None, help="Path to transactions CSV")
    parser.add_argument("--node-features-path", default=None, help="Optional node feature parquet/csv")
    parser.add_argument("--edge-features-path", default=None, help="Optional edge feature parquet/csv")
    parser.add_argument("--output-dir", default="model/edge_fraud", help="Directory for model artifacts")
    parser.add_argument("--fraud-col", default=None, help="Fraud label column name")
    parser.add_argument("--sender-col", default=None, help="Sender id column")
    parser.add_argument("--receiver-col", default=None, help="Receiver id column")
    parser.add_argument("--amount-col", default=None, help="Amount column")
    parser.add_argument("--timestamp-col", default=None, help="Timestamp column")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GraphSAGE hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GraphSAGE layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation edge ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test edge ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include-leaky-features",
        action="store_true",
        help="Keep fraud-label-derived node features. Off by default for safer training.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime_paths(args: argparse.Namespace, config: Dict) -> Dict[str, str]:
    data_cfg = config.get("data", {})
    return {
        "transactions_path": args.transactions_path or data_cfg.get("transaction_path"),
        "node_features_path": args.node_features_path or "data/features/node_features.parquet",
        "edge_features_path": args.edge_features_path or "data/features/edge_features.parquet",
        "sender_col": args.sender_col or data_cfg.get("sender_col", "sender_id"),
        "receiver_col": args.receiver_col or data_cfg.get("receiver_col", "receiver_id"),
        "amount_col": args.amount_col or data_cfg.get("amount_col", "amount"),
        "timestamp_col": args.timestamp_col or data_cfg.get("timestamp_col", "timestamp"),
        "fraud_col": args.fraud_col or data_cfg.get("fraud_col", "is_fraud_txn"),
    }


def load_transactions(path: str, timestamp_col: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Transactions CSV not found: {path}")

    df = pd.read_csv(path)
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    else:
        LOGGER.warning("Timestamp column %s not found. Falling back to input row order.", timestamp_col)
        df = df.reset_index(drop=True)
    return df


def load_feature_frame(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file format: {path}")


def sanitize_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.astype(np.float32)


def align_node_features(
    features_df: pd.DataFrame,
    user_id_to_node: Dict[str, int],
    include_leaky_features: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    if not include_leaky_features:
        features_df = features_df.drop(columns=DEFAULT_LEAKY_NODE_FEATURES, errors="ignore")

    features_df = sanitize_numeric_frame(features_df)
    num_nodes = len(user_id_to_node)
    ordered_ids = sorted(user_id_to_node.items(), key=lambda item: item[1])
    ordered_user_ids = [user_id for user_id, _ in ordered_ids]

    if len(features_df) != num_nodes:
        if set(map(str, features_df.index.tolist())) >= set(ordered_user_ids):
            features_df.index = features_df.index.map(str)
            features_df = features_df.loc[ordered_user_ids]
        else:
            raise ValueError(
                f"Node feature row count {len(features_df)} does not match graph node count {num_nodes}"
            )
    elif pd.api.types.is_numeric_dtype(features_df.index):
        features_df = features_df.sort_index()
    else:
        features_df.index = features_df.index.map(str)
        if set(ordered_user_ids).issubset(set(features_df.index.tolist())):
            features_df = features_df.loc[ordered_user_ids]

    zero_var_cols = [col for col in features_df.columns if float(features_df[col].std()) < 1e-12]
    if zero_var_cols:
        LOGGER.info("Dropping %d zero-variance node features", len(zero_var_cols))
        features_df = features_df.drop(columns=zero_var_cols)

    return features_df, zero_var_cols


def align_edge_features(features_df: pd.DataFrame, expected_rows: int) -> Tuple[pd.DataFrame, List[str]]:
    features_df = sanitize_numeric_frame(features_df)
    if len(features_df) != expected_rows:
        raise ValueError(
            f"Edge feature row count {len(features_df)} does not match graph edge count {expected_rows}"
        )

    zero_var_cols = [col for col in features_df.columns if float(features_df[col].std()) < 1e-12]
    if zero_var_cols:
        LOGGER.info("Dropping %d zero-variance edge features", len(zero_var_cols))
        features_df = features_df.drop(columns=zero_var_cols)

    return features_df.reset_index(drop=True), zero_var_cols


def compute_feature_frames(
    g,
    transactions_df: pd.DataFrame,
    user_id_to_node: Dict[str, int],
    args: argparse.Namespace,
    node_features_path: str,
    edge_features_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(node_features_path):
        raise FileNotFoundError(f"Node features parquet/csv not found: {node_features_path}")
    if not os.path.exists(edge_features_path):
        raise FileNotFoundError(f"Edge features parquet/csv not found: {edge_features_path}")

    LOGGER.info("Loading precomputed features from %s and %s", node_features_path, edge_features_path)
    node_df = load_feature_frame(node_features_path)
    edge_df = load_feature_frame(edge_features_path)

    node_df, _ = align_node_features(node_df, user_id_to_node, args.include_leaky_features)
    edge_df, _ = align_edge_features(edge_df, g.number_of_edges(EDGE_ETYPE))
    return node_df, edge_df


def build_graph(
    transactions_df: pd.DataFrame,
    sender_col: str,
    receiver_col: str,
    amount_col: str,
    timestamp_col: str,
    fraud_col: str,
) -> Tuple[dgl.DGLHeteroGraph, Dict[str, int], pd.DataFrame]:
    builder = GraphBuilder(num_workers=1, chunk_size=max(len(transactions_df), 1))
    graph, user_id_to_node, returned_df = builder.build_user_transaction_graph(
        transactions_df.copy(),
        sender_col=sender_col,
        receiver_col=receiver_col,
        amount_col=amount_col,
        timestamp_col=timestamp_col,
        fraud_col=fraud_col,
    )
    return graph, user_id_to_node, returned_df


def prepare_labels_and_timestamps(
    df: pd.DataFrame,
    sender_col: str,
    receiver_col: str,
    fraud_col: str,
    timestamp_col: str,
    expected_edges: int,
) -> Tuple[np.ndarray, np.ndarray]:
    edge_df = df.copy()
    edge_df = edge_df[edge_df[sender_col].notna() & edge_df[receiver_col].notna()].reset_index(drop=True)

    if len(edge_df) != expected_edges:
        raise ValueError(
            f"Edge count mismatch between dataframe ({len(edge_df)}) and graph ({expected_edges})"
        )

    if fraud_col not in edge_df.columns:
        raise ValueError(f"Fraud column not found in transactions CSV: {fraud_col}")

    labels = edge_df[fraud_col].fillna(0).astype(np.float32).to_numpy()
    if timestamp_col in edge_df.columns:
        timestamps = pd.to_numeric(edge_df[timestamp_col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    else:
        timestamps = np.arange(len(edge_df), dtype=np.float64)
    return labels, timestamps


def temporal_edge_split(
    timestamps: np.ndarray,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")

    order = np.argsort(timestamps, kind="mergesort")
    n_edges = len(order)
    n_test = max(1, int(round(n_edges * test_ratio)))
    n_val = max(1, int(round(n_edges * val_ratio)))
    n_train = n_edges - n_val - n_test

    if n_train < 1:
        raise ValueError("Not enough edges for temporal train/val/test split")

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return train_idx, val_idx, test_idx


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = np.concatenate(([0.0], tps / positives, [1.0]))
    fpr = np.concatenate(([0.0], fps / negatives, [1.0]))
    return float(np.trapz(tpr, fpr))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives

    precision = np.concatenate(([1.0], precision, [y_sorted.mean()]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    return float(np.trapz(precision, recall))


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = compute_binary_metrics(y_true, y_score, float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    return best_threshold


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    metrics["roc_auc"] = compute_roc_auc(y_true, y_score)
    metrics["pr_auc"] = compute_pr_auc(y_true, y_score)
    return metrics


def to_device_tensor(array: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(array, dtype=dtype, device=device)


def train_model(
    model: GraphSAGEEdgeClassifier,
    graph,
    node_features: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_labels = labels[train_idx]
    positives = float(train_labels.sum().item())
    negatives = float(train_labels.shape[0] - positives)
    pos_weight_value = negatives / max(positives, 1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=labels.device))

    best_state = None
    best_pr_auc = -math.inf
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(graph, node_features, edge_src, edge_dst, edge_features)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(graph, node_features, edge_src, edge_dst, edge_features)

        val_scores = torch.sigmoid(eval_logits[val_idx]).detach().cpu().numpy()
        val_true = labels[val_idx].detach().cpu().numpy()
        val_metrics = evaluate_scores(val_true, val_scores, threshold=0.5)

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "val_roc_auc": float(val_metrics["roc_auc"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
        }
        history.append(epoch_result)

        LOGGER.info(
            "Epoch %03d | loss=%.4f | val_pr_auc=%.4f | val_roc_auc=%.4f | val_f1=%.4f",
            epoch,
            epoch_result["train_loss"],
            epoch_result["val_pr_auc"],
            epoch_result["val_roc_auc"],
            epoch_result["val_f1"],
        )

        score = val_metrics["pr_auc"]
        if np.isnan(score):
            score = val_metrics["f1"]

        if score > best_pr_auc:
            best_pr_auc = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    return best_state, history


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    config = load_config(args.config)
    runtime = resolve_runtime_paths(args, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)

    transactions_df = load_transactions(runtime["transactions_path"], runtime["timestamp_col"])
    LOGGER.info("Loaded %d transactions from %s", len(transactions_df), runtime["transactions_path"])

    graph, user_id_to_node, transactions_df = build_graph(
        transactions_df,
        sender_col=runtime["sender_col"],
        receiver_col=runtime["receiver_col"],
        amount_col=runtime["amount_col"],
        timestamp_col=runtime["timestamp_col"],
        fraud_col=runtime["fraud_col"],
    )

    node_df, edge_df = compute_feature_frames(
        graph,
        transactions_df,
        user_id_to_node,
        args,
        runtime["node_features_path"],
        runtime["edge_features_path"],
    )

    labels_np, timestamps_np = prepare_labels_and_timestamps(
        transactions_df,
        sender_col=runtime["sender_col"],
        receiver_col=runtime["receiver_col"],
        fraud_col=runtime["fraud_col"],
        timestamp_col=runtime["timestamp_col"],
        expected_edges=graph.number_of_edges(EDGE_ETYPE),
    )

    train_idx, val_idx, test_idx = temporal_edge_split(
        timestamps_np,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)

    LOGGER.info(
        "Temporal split sizes | train=%d val=%d test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    node_values = node_df.to_numpy(dtype=np.float32)
    edge_values = edge_df.to_numpy(dtype=np.float32)

    node_scaler = StandardScaler.fit(node_values)
    edge_scaler = StandardScaler.fit(edge_values[train_idx])

    node_values = node_scaler.transform(node_values)
    edge_values = edge_scaler.transform(edge_values)

    src_nodes, dst_nodes = graph.edges(etype=EDGE_ETYPE)
    encoder_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=graph.number_of_nodes("user"))
    encoder_graph = dgl.add_self_loop(encoder_graph).to(device)

    node_features = to_device_tensor(node_values, device)
    edge_features = to_device_tensor(edge_values, device)
    labels = to_device_tensor(labels_np, device)
    edge_src = src_nodes.to(device)
    edge_dst = dst_nodes.to(device)

    model = GraphSAGEEdgeClassifier(
        node_in_dim=node_features.shape[1],
        edge_in_dim=edge_features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    best_state, history = train_model(
        model=model,
        graph=encoder_graph,
        node_features=node_features,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_features=edge_features,
        labels=labels,
        train_idx=train_idx_t,
        val_idx=val_idx_t,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(encoder_graph, node_features, edge_src, edge_dst, edge_features)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

    val_threshold = find_best_threshold(labels_np[val_idx], scores[val_idx])
    train_metrics = evaluate_scores(labels_np[train_idx], scores[train_idx], val_threshold)
    val_metrics = evaluate_scores(labels_np[val_idx], scores[val_idx], val_threshold)
    test_metrics = evaluate_scores(labels_np[test_idx], scores[test_idx], val_threshold)

    checkpoint_path = output_dir / "model.pt"
    metadata_path = output_dir / "metadata.json"
    history_path = output_dir / "training_history.json"
    report_path = output_dir / "metrics.json"

    torch.save(best_state, checkpoint_path)

    metadata = {
        "model_type": "graphsage_edge_classifier",
        "edge_type": list(EDGE_ETYPE),
        "node_feature_columns": node_df.columns.tolist(),
        "edge_feature_columns": edge_df.columns.tolist(),
        "node_scaler": node_scaler.to_dict(),
        "edge_scaler": edge_scaler.to_dict(),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "threshold": val_threshold,
        "sender_col": runtime["sender_col"],
        "receiver_col": runtime["receiver_col"],
        "amount_col": runtime["amount_col"],
        "timestamp_col": runtime["timestamp_col"],
        "fraud_col": runtime["fraud_col"],
        "include_leaky_features": args.include_leaky_features,
        "artifacts": {
            "model_state_dict": str(checkpoint_path),
            "metrics": str(report_path),
            "history": str(history_path),
        },
    }

    report = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "split_counts": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "class_balance": {
            "train_fraud_rate": float(labels_np[train_idx].mean()),
            "validation_fraud_rate": float(labels_np[val_idx].mean()),
            "test_fraud_rate": float(labels_np[test_idx].mean()),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    LOGGER.info("Saved model checkpoint to %s", checkpoint_path)
    LOGGER.info("Saved metadata to %s", metadata_path)
    LOGGER.info("Saved metrics to %s", report_path)
    LOGGER.info(
        "Final test metrics | threshold=%.2f precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f roc_auc=%.4f",
        test_metrics["threshold"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["pr_auc"],
        test_metrics["roc_auc"],
    )


if __name__ == "__main__":
    main()
