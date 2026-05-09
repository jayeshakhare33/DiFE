#!/usr/bin/env python3
"""
Score new transaction CSV records with the trained GraphSAGE edge fraud model.

The script can score a CSV by itself or combine it with a historical CSV first.
Using history produces more realistic node and edge context for the new records.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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

from feature_engineering.feature_extractor import EdgeFeatureExtractor, FeatureExtractor
from gnn_training.edge_model import GraphSAGEEdgeClassifier
from graph_processing.graph_builder import GraphBuilder


LOGGER = logging.getLogger("score_transactions_csv")
EDGE_ETYPE = ("user", "transaction", "user")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score transaction CSV with trained edge model")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt and metadata.json")
    parser.add_argument("--new-transactions-path", required=True, help="CSV with transactions to score")
    parser.add_argument(
        "--history-transactions-path",
        default=None,
        help="Optional historical CSV used to build graph context before scoring the new rows",
    )
    parser.add_argument("--config", default="config.yaml", help="Optional config file for column defaults")
    parser.add_argument("--output-path", default="data/predictions/scored_transactions.csv", help="Output CSV path")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--sender-col", default=None, help="Sender id column override")
    parser.add_argument("--receiver-col", default=None, help="Receiver id column override")
    parser.add_argument("--amount-col", default=None, help="Amount column override")
    parser.add_argument("--timestamp-col", default=None, help="Timestamp column override")
    parser.add_argument("--fraud-col", default=None, help="Fraud label column override if present")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_config(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_runtime_paths(args: argparse.Namespace, config: Dict, metadata: Dict) -> Dict[str, str]:
    data_cfg = config.get("data", {})
    return {
        "sender_col": args.sender_col or metadata.get("sender_col") or data_cfg.get("sender_col", "sender_id"),
        "receiver_col": args.receiver_col or metadata.get("receiver_col") or data_cfg.get("receiver_col", "receiver_id"),
        "amount_col": args.amount_col or metadata.get("amount_col") or data_cfg.get("amount_col", "amount"),
        "timestamp_col": args.timestamp_col or metadata.get("timestamp_col") or data_cfg.get("timestamp_col", "timestamp"),
        "fraud_col": args.fraud_col or metadata.get("fraud_col") or data_cfg.get("fraud_col", "is_fraud_txn"),
    }


def load_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def sanitize_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.astype(np.float32)


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
        fraud_col=fraud_col if fraud_col in transactions_df.columns else None,
    )
    return graph, user_id_to_node, returned_df


def compute_feature_frames(
    g: dgl.DGLHeteroGraph,
    transactions_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    node_extractor = FeatureExtractor(transaction_df=transactions_df)
    edge_extractor = EdgeFeatureExtractor()
    node_df = node_extractor.extract_all_features(g, node_type="user", transaction_df=transactions_df)
    edge_features = edge_extractor.extract_all_edge_features(g, edge_type=EDGE_ETYPE, transaction_df=transactions_df)
    edge_df = pd.DataFrame(
        {
            name: tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor
            for name, tensor in edge_features.items()
        }
    )
    return node_df, edge_df


def align_node_features(
    features_df: pd.DataFrame,
    user_id_to_node: Dict[str, int],
    expected_columns: List[str],
) -> pd.DataFrame:
    features_df = sanitize_numeric_frame(features_df)
    ordered_ids = sorted(user_id_to_node.items(), key=lambda item: item[1])
    ordered_user_ids = [user_id for user_id, _ in ordered_ids]

    if len(features_df) == len(ordered_user_ids) and pd.api.types.is_numeric_dtype(features_df.index):
        features_df = features_df.sort_index()
    else:
        features_df.index = features_df.index.map(str)
        features_df = features_df.loc[ordered_user_ids]

    for column in expected_columns:
        if column not in features_df.columns:
            features_df[column] = 0.0

    return features_df[expected_columns].astype(np.float32)


def align_edge_features(features_df: pd.DataFrame, expected_columns: List[str], expected_rows: int) -> pd.DataFrame:
    features_df = sanitize_numeric_frame(features_df).reset_index(drop=True)
    if len(features_df) != expected_rows:
        raise ValueError(
            f"Edge feature row count {len(features_df)} does not match graph edge count {expected_rows}"
        )

    for column in expected_columns:
        if column not in features_df.columns:
            features_df[column] = 0.0

    return features_df[expected_columns].astype(np.float32)


def apply_scaler(values: np.ndarray, scaler_cfg: Dict[str, List[float]]) -> np.ndarray:
    mean = np.asarray(scaler_cfg["mean"], dtype=np.float32)
    std = np.asarray(scaler_cfg["std"], dtype=np.float32)
    std = np.where(std < 1e-12, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def to_device_tensor(array: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(array, dtype=dtype, device=device)


def load_model(model_dir: Path, metadata: Dict, device: torch.device) -> GraphSAGEEdgeClassifier:
    model = GraphSAGEEdgeClassifier(
        node_in_dim=len(metadata["node_feature_columns"]),
        edge_in_dim=len(metadata["edge_feature_columns"]),
        hidden_dim=metadata["hidden_dim"],
        num_layers=metadata["num_layers"],
        dropout=metadata["dropout"],
    ).to(device)
    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    setup_logging()

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    config = load_config(args.config)
    runtime = resolve_runtime_paths(args, config, metadata)
    device = resolve_device(args.device)

    history_df = load_csv(args.history_transactions_path) if args.history_transactions_path else None
    new_df = load_csv(args.new_transactions_path).copy()
    new_df["__is_new_record__"] = 1
    new_df["__source_row_id__"] = np.arange(len(new_df))

    if history_df is not None:
        history_df = history_df.copy()
        history_df["__is_new_record__"] = 0
        history_df["__source_row_id__"] = -1
        combined_df = pd.concat([history_df, new_df], ignore_index=True, sort=False)
    else:
        combined_df = new_df.copy()

    if runtime["timestamp_col"] in combined_df.columns:
        combined_df = combined_df.sort_values(runtime["timestamp_col"], kind="mergesort").reset_index(drop=True)
    else:
        combined_df = combined_df.reset_index(drop=True)

    valid_mask = combined_df[runtime["sender_col"]].notna() & combined_df[runtime["receiver_col"]].notna()
    valid_df = combined_df.loc[valid_mask].reset_index(drop=True)
    if valid_df.empty:
        raise ValueError("No valid transactions found after filtering sender/receiver columns")

    graph, user_id_to_node, valid_df = build_graph(
        valid_df,
        sender_col=runtime["sender_col"],
        receiver_col=runtime["receiver_col"],
        amount_col=runtime["amount_col"],
        timestamp_col=runtime["timestamp_col"],
        fraud_col=runtime["fraud_col"],
    )

    node_df, edge_df = compute_feature_frames(graph, valid_df)
    node_df = align_node_features(node_df, user_id_to_node, metadata["node_feature_columns"])
    edge_df = align_edge_features(edge_df, metadata["edge_feature_columns"], graph.number_of_edges(EDGE_ETYPE))

    node_values = apply_scaler(node_df.to_numpy(dtype=np.float32), metadata["node_scaler"])
    edge_values = apply_scaler(edge_df.to_numpy(dtype=np.float32), metadata["edge_scaler"])

    src_nodes, dst_nodes = graph.edges(etype=EDGE_ETYPE)
    encoder_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=graph.number_of_nodes("user"))
    encoder_graph = dgl.add_self_loop(encoder_graph).to(device)

    node_features = to_device_tensor(node_values, device)
    edge_features = to_device_tensor(edge_values, device)
    edge_src = src_nodes.to(device)
    edge_dst = dst_nodes.to(device)

    model = load_model(model_dir, metadata, device)
    with torch.no_grad():
        logits = model(encoder_graph, node_features, edge_src, edge_dst, edge_features)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

    threshold = float(metadata.get("threshold", 0.5))
    predictions = (scores >= threshold).astype(np.int32)

    scored_df = valid_df.copy()
    scored_df["fraud_probability"] = scores
    scored_df["predicted_is_fraud"] = predictions
    scored_df["model_threshold"] = threshold

    new_scored_df = scored_df[scored_df["__is_new_record__"] == 1].copy()
    new_scored_df = new_scored_df.sort_values("__source_row_id__").reset_index(drop=True)
    new_scored_df = new_scored_df.drop(columns=["__is_new_record__", "__source_row_id__"], errors="ignore")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_scored_df.to_csv(output_path, index=False)

    LOGGER.info("Scored %d new transactions", len(new_scored_df))
    LOGGER.info("Fraud predictions above threshold: %d", int(new_scored_df["predicted_is_fraud"].sum()))
    LOGGER.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
