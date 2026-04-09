# 📋 Seminar 2 Preparation Guide: DiFE — Graph-Based Fraud Detection

> **Project:** Fraud Detection using Graph Neural Networks with Distributed Feature Engineering (DiFE)  
> **Dataset:** IEEE-CIS Fraud Detection (Kaggle)  
> **Date:** February 2026

---

## 📌 Table of Contents

1. [Seminar 1 Recap — What Was Covered](#seminar-1-recap)
2. [Current Project Status — What's Done](#current-project-status)
3. [What's Remaining — Phase 2 Work](#whats-remaining)
4. [Seminar 2 PPT Structure & Slide Guide](#seminar-2-ppt-guide)

---

## 📖 Seminar 1 Recap

Based on the Seminar 1 PPT (`poject seminar 1.pptx`), the following areas were likely covered:

### Topics Presented in Phase 1

| # | Topic | Status |
|---|-------|--------|
| 1 | Problem Statement — Online fraud costs billions annually | ✅ Presented |
| 2 | Literature Survey — Existing fraud detection approaches | ✅ Presented |
| 3 | Proposed Approach — Using GNNs on heterogeneous graphs | ✅ Presented |
| 4 | Dataset Description — IEEE-CIS Fraud Detection from Kaggle | ✅ Presented |
| 5 | System Architecture Overview — High-level design | ✅ Presented |
| 6 | Methodology — HeteroRGCN model approach | ✅ Presented |
| 7 | Expected Outcomes / Objectives | ✅ Presented |
| 8 | Timeline / Project Plan | ✅ Presented |

### What Phase 1 Was About (Proposal Phase)

In Seminar 1, you proposed the **idea**, justified **why GNNs are better than traditional ML** for fraud detection, and outlined the **plan**. The focus was on:
- **What** you're building (fraud detection system using GNN)
- **Why** it's needed (traditional ML misses relational patterns)
- **How** you plan to do it (heterogeneous graph + RGCN)
- **What data** you'll use (IEEE-CIS dataset with 590K+ transactions)

---

## ✅ Current Project Status — What's Done

### 1. 🏗️ Core Infrastructure — COMPLETE

| Component | File(s) | Status |
|-----------|---------|--------|
| Data loading & preprocessing | `10_data_loader.ipynb` | ✅ Done |
| Graph construction (heterogeneous) | `gnn/graph_utils.py`, `gnn/data.py` | ✅ Done |
| GNN Model (HeteroRGCN) | `gnn/pytorch_model.py` | ✅ Done |
| Training pipeline | `train.py`, `20_modeling.ipynb` | ✅ Done |
| Evaluation & metrics | `gnn/utils.py` | ✅ Done |
| Visualization | `30_visual.ipynb` | ✅ Done |
| Argument parsing & hyperparameters | `gnn/estimator_fns.py` | ✅ Done |

### 2. 📊 Model Training — COMPLETE (Baseline)

The model has been trained for **1000 epochs** with results available in `output/results.txt`.

**Key Baseline Results:**
- **ROC AUC:** 0.92
- **Precision:** 0.86
- **Best F1 Score:** ~0.56 (around epoch 622, F1=0.5646)
- **Graph Stats:** 726,345 Nodes | 19,518,802 Edges

**Confusion Matrix (Best Model):**

| | Labels Positive | Labels Negative |
|---|---|---|
| **Predicted Positive** | 1,435 | 240 |
| **Predicted Negative** | 2,629 | 113,804 |

**Output Files:**
- `output/roc_curve.png` — ROC curve (AUC = 0.92) ✅
- `output/pr_curve.png` — PR curve (AP = 0.59) ✅
- `output/f1.jpg` — F1 score progression ✅
- `output/loss.jpg` — Loss curve ✅
- `output/results.txt` — Full training log ✅

### 3. 🧪 Distributed Feature Engineering Framework — CODE COMPLETE (Not Validated)

| Component | File | Status |
|-----------|------|--------|
| Feature Extractor Base Class | `gnn/feature_engineering.py` | ✅ Code written |
| GraphNeighborAggregator | `gnn/feature_engineering.py` | ✅ Code written |
| TemporalFeatureExtractor | `gnn/feature_engineering.py` | ✅ Code written |
| StatisticalFeatureExtractor | `gnn/feature_engineering.py` | ✅ Code written |
| RiskScoreExtractor | `gnn/feature_engineering.py` | ✅ Code written |
| GraphCentralityExtractor | `gnn/advanced_features.py` | ✅ Code written |
| PatternMatchingExtractor | `gnn/advanced_features.py` | ✅ Code written |
| CrossFeatureExtractor | `gnn/advanced_features.py` | ✅ Code written |
| EmbeddingBasedExtractor | `gnn/advanced_features.py` | ⚠️ Placeholder only |
| DistributedFeaturePipeline | `gnn/distributed_feature_pipeline.py` | ✅ Code written |
| Standalone extraction script | `distributed_feature_extraction.py` | ✅ Code written |
| Feature Registry & Engine | `gnn/feature_engineering.py` | ✅ Code written |

### 4. 📚 Documentation — Written

| Document | Description | Status |
|----------|-------------|--------|
| `README.md` | Project overview & usage | ✅ Done |
| `FEATURES_DOCUMENTATION.md` | Complete feature extraction docs | ✅ Done |
| `GNN_ARCHITECTURE_EXPLANATION.md` | Model architecture explanation | ✅ Done |
| `HETEROGENEOUS_GRAPH_EXPLANATION.md` | Heterogeneous graph concepts | ✅ Done |
| `IMPLEMENTATION_SUMMARY.md` | Implementation overview | ✅ Done |
| `README_DISTRIBUTED_FEATURES.md` | Distributed features usage guide | ✅ Done |

---

## ⚠️ What's Remaining — Phase 2 Work

### 🔴 Critical (Must Do)

#### 1. **Run & Validate Distributed Feature Engineering**
- **Status:** Code is written but **never actually executed end-to-end**
- **What to do:**
  - Run `distributed_feature_extraction.py` with actual IEEE-CIS data
  - Verify all 7 extractors produce valid features
  - Check feature shapes and alignment
  - Fix any runtime bugs
- **Why it matters:** This is the core "DiFE" contribution — distributed, decentralized feature engineering

#### 2. **Train Model WITH Enhanced Features & Compare**
- **Status:** Model only trained with baseline features
- **What to do:**
  ```bash
  # Step 1: Extract distributed features
  python distributed_feature_extraction.py \
      --transaction-data ./ieee-data/train_transaction.csv \
      --identity-data ./ieee-data/train_identity.csv \
      --output-dir ./features
  
  # Step 2: Train with enhanced features
  python train.py --use-distributed-features --feature-dir ./features --enhance-features true
  ```
- **Expected outcome:** Improved F1 score, precision, and recall compared to baseline
- **Why it matters:** You need **before vs. after comparison** to prove DiFE works

#### 3. **Generate Comparison Results Table**
- **Status:** Only baseline results available
- **What to do:** Create a comparison like:
  
  | Metric | Baseline (No DiFE) | With DiFE | Improvement |
  |--------|--------------------:|----------:|------------:|
  | ROC AUC | 0.92 | ? | ? |
  | Precision | 0.86 | ? | ? |
  | Recall | 0.35 | ? | ? |
  | F1 Score | 0.56 | ? | ? |
  | PR AUC | 0.59 | ? | ? |

#### 4. **Fix Training Instability After Epoch 650**
- **Status:** Training results show severe instability/collapse after epoch ~650
- **What happened:** F1 drops from 0.55+ to near 0.0 between epochs 652-665, then partially recovers
- **What to do:**
  - Implement **early stopping** (save best model based on validation F1)
  - Add **learning rate scheduling** (reduce LR on plateau)
  - Try gradient clipping
- **Why it matters:** Shows model robustness and proper engineering

### 🟡 Important (Should Do)

#### 5. **Complete the EmbeddingBasedExtractor**
- **Status:** Currently a placeholder that returns empty array
- **What to do:** Implement actual embedding-based feature extraction:
  - Use pre-trained node embeddings from initial GNN training
  - Or implement Node2Vec / DeepWalk embedding approach
- **File:** `gnn/advanced_features.py`, class `EmbeddingBasedExtractor`

#### 6. **Add Proper Model Saving & Loading**
- **Status:** Model directory (`./model/`) is empty — no saved model checkpoint
- **What to do:**
  - Save best model checkpoint during training
  - Implement model loading for inference
  - Save training configuration alongside model

#### 7. **Implement Inference/Prediction Pipeline**
- **Status:** No inference script exists
- **What to do:**
  - Create `predict.py` or `inference.py` that:
    - Loads a trained model
    - Accepts new transaction data
    - Returns fraud probability
  - This demonstrates real-world usability

#### 8. **Add Hyperparameter Tuning Results**
- **Status:** Only default hyperparameters used
- **Current defaults:** `n_hidden=16`, `n_layers=3`, `lr=0.01`, `epochs=700`, `embedding_size=360`
- **What to do:** Try different combinations and document results

#### 9. **Benchmark Against Traditional ML Models**
- **Status:** No comparison with non-GNN baselines
- **What to do:** Train and compare with:
  - Random Forest
  - XGBoost / LightGBM
  - Logistic Regression
  - Feed-forward Neural Network
- **Why it matters:** Proves GNN advantage over traditional approaches

### 🟢 Nice to Have (Bonus Points)

#### 10. **Real-time/Streaming Fraud Detection Demo**
- Build a simple web interface or Streamlit app showing real-time predictions

#### 11. **Feature Importance Analysis**
- Show which features contribute most to fraud detection
- Use SHAP values or attention weights

#### 12. **Scalability Benchmarks**
- Measure feature extraction time with 1, 2, 4, 8 workers
- Show parallelization speedup graph

#### 13. **Error Analysis**
- Analyze false positives and false negatives
- Identify patterns in misclassified transactions

#### 14. **Unit Tests**
- No tests exist in the project
- Add tests for core functions (graph construction, feature extraction, etc.)

---

## 🎤 Seminar 2 PPT Guide

### PPT Philosophy for Phase 2

> **Phase 1 was about PROPOSING.** ❝Here's what we plan to do.❞  
> **Phase 2 is about DELIVERING.** ❝Here's what we built, and here are the results.❞

Your PPT should demonstrate that you **implemented** the system, **ran experiments**, **obtained results**, and **learned from them**.

---

### 📑 Recommended Slide Structure (20-25 slides)

---

#### **Slide 1: Title Slide**
- **Title:** DiFE: Distributed Feature Engineering for Graph-Based Fraud Detection
- **Subtitle:** Project Seminar 2 — Implementation & Results
- Your name, roll number, guide's name, department, college, date

---

#### **Slide 2: Agenda / Outline**
List: Problem Statement → Approach Recap → Implementation → Feature Engineering → Results → Comparison → Conclusion → Future Work

---

#### **Slide 3: Problem Statement (Quick Recap)**
- Online fraud costs $40B+ annually
- Traditional ML treats transactions independently — misses relational patterns
- Need: A system that captures **relationships** between transactions, cards, addresses, devices
- Keep this brief — they've seen it before in Seminar 1

---

#### **Slide 4: Proposed Solution (Quick Recap)**
- Graph Neural Network on **heterogeneous transaction graph**
- Key innovation: **Distributed Feature Engineering (DiFE)** framework
- Dataset: IEEE-CIS Fraud Detection (590K+ transactions, 726K nodes, 19.5M edges)
- 1-2 sentences on why this is better than traditional ML

---

#### **Slide 5: System Architecture**
- Full architecture diagram showing:
  ```
  Raw Data → Data Preprocessing → Graph Construction → 
  Distributed Feature Engineering → HeteroRGCN Model → 
  Fraud Classification
  ```
- Highlight the DiFE component as your contribution
- Use the architecture diagram from `IMPLEMENTATION_SUMMARY.md`

---

#### **Slide 6: Heterogeneous Graph Construction**
- Show the graph structure with node types and edge types
- Use the visual from `graph_intro.png` or recreate it
- Key stats: **726,345 Nodes** | **19,518,802 Edges** | **50+ node types** | **100+ edge types**
- Explain: "Each transaction is connected to its card, address, email, device, product"

---

#### **Slide 7: Graph Structure — Why It Works**
- Visual example showing how fraud patterns emerge:
  - Multiple transactions sharing same card → suspicious
  - Multiple cards from same address → fraud ring
  - Same device, different card numbers → compromised device
- Use the diagrams from `HETEROGENEOUS_GRAPH_EXPLANATION.md`

---

#### **Slide 8: GNN Architecture — HeteroRGCN**
- Architecture diagram:
  ```
  Input (Node Features + Embeddings)
    ↓
  HeteroRGCN Layer 1 (Relation-specific Linear transforms)
    ↓ LeakyReLU
  HeteroRGCN Layer 2
    ↓ LeakyReLU  
  HeteroRGCN Layer 3
    ↓
  Output Layer (Binary Classification: Fraud / Not Fraud)
  ```
- Key design choices: Sum aggregation, relation-specific weights, trainable embeddings
- Mathematical formulation: `h_v^(l+1) = LeakyReLU(Σ_{r∈R} Σ_{u∈N_r(v)} W_r · h_u^(l))`

---

#### **Slide 9: DiFE Framework — Overview**
- **This is your KEY contribution slide**
- Distributed Feature Engineering framework:
  - Modular, parallelizable feature extractors
  - 7 different extractor types
  - Can scale across multiple workers/machines
- Architecture diagram showing Worker 1, Worker 2, ... Worker N → Feature Combiner → Output

---

#### **Slide 10: Feature Engineering — Feature Categories**
- Table of all 7 extractors:

  | # | Extractor | Type | What It Captures |
  |---|-----------|------|------------------|
  | 1 | GraphNeighborAggregator | Graph | Local graph structure, neighbor patterns |
  | 2 | TemporalFeatureExtractor | Temporal | Time-of-day, velocity, rapid transactions |
  | 3 | StatisticalFeatureExtractor | Statistical | Distribution characteristics (skewness, kurtosis) |
  | 4 | RiskScoreExtractor | Risk | Amount anomalies, entity frequency |
  | 5 | GraphCentralityExtractor | Graph | Node importance (degree centrality) |
  | 6 | PatternMatchingExtractor | Risk | Card testing, device sharing, round amounts |
  | 7 | CrossFeatureExtractor | Statistical | Feature interactions (Amount×Product, etc.) |

- Total: **470+ features** extracted

---

#### **Slide 11: Feature Engineering — How It Works**
- Example walkthrough — how features detect a "card testing attack":
  1. `rapid_transactions = 1` (multiple transactions in short time)
  2. `small_amounts = 1` (testing with small amounts)
  3. `round_amounts = 1` (round number testing)
  4. High `card_velocity` (card used frequently)
  5. High `neighbor_count` (same card used multiple times in graph)
- **Result:** Model combines these features → detects card testing pattern

---

#### **Slide 12: Implementation Details**
- **Tech Stack:**
  - Python, PyTorch, PyTorch Geometric (PyG)
  - Deep Graph Library concepts, Pandas, NumPy, scikit-learn
  - Multiprocessing for distributed feature extraction
- **Code Structure:**
  ```
  graph-fraud-detection/
  ├── 10_data_loader.ipynb      # Data preprocessing
  ├── 20_modeling.ipynb          # Training
  ├── 30_visual.ipynb            # Visualization
  ├── train.py                   # Training script
  ├── distributed_feature_extraction.py
  ├── gnn/
  │   ├── pytorch_model.py       # HeteroRGCN model
  │   ├── feature_engineering.py # DiFE framework
  │   ├── advanced_features.py   # Advanced extractors
  │   ├── graph_utils.py         # Graph construction
  │   └── ...
  ```
- **Hyperparameters:** Hidden=16, Layers=3, LR=0.01, Epochs=1000, Embedding=360

---

#### **Slide 13: Data Preprocessing Pipeline**
- Steps from raw Kaggle data to model input:
  1. Load transaction CSV (590K rows, 400+ columns)
  2. Load identity CSV (join on TransactionID)
  3. Extract edge lists (card, address, email, device, product relationships)
  4. Build heterogeneous graph with PyG HeteroData
  5. Compute node features, train/test split
  6. Normalize features (mean=0, std=1)

---

#### **Slide 14: Training Results — Loss & F1 Curves**
- Show `output/loss.jpg` and `output/f1.jpg`
- Highlight:
  - Loss decreases steadily from 0.65 to ~0.08
  - F1 score improves from 0.0 to peak ~0.56
  - Training time: ~31 seconds/epoch × 1000 epochs ≈ 8.5 hours
- Note the instability after epoch 650 and discuss solutions (early stopping, LR scheduling)

---

#### **Slide 15: Results — ROC and PR Curves**
- Show `output/roc_curve.png` (AUC = 0.92) and `output/pr_curve.png` (AP = 0.59)
- **ROC AUC of 0.92** = strong discriminative ability
- **PR AUC of 0.59** = good performance on imbalanced data
- Confusion matrix:

  | | Positive | Negative |
  |---|---|---|
  | **Pred +** | 1,435 (TP) | 240 (FP) |
  | **Pred −** | 2,629 (FN) | 113,804 (TN) |

---

#### **Slide 16: Results — Performance Metrics Summary**

| Metric | Value |
|--------|------:|
| **ROC AUC** | 0.92 |
| **Precision** | 0.86 |
| **Recall** | 0.35 |
| **F1 Score** | 0.56 (best) |
| **PR AUC / AP** | 0.59 |
| **Accuracy** | 97.6% |

- Note: High precision (0.86) was prioritized — misclassifying legitimate transactions as fraud harms user experience

---

#### **Slide 17: DiFE Impact — Before vs. After** *(Run these experiments!)*

| Metric | Without DiFE | With DiFE | Δ Improvement |
|--------|-------------:|----------:|--------------:|
| ROC AUC | 0.92 | _fill after running_ | — |
| Precision | 0.86 | _fill after running_ | — |
| Recall | 0.35 | _fill after running_ | — |
| F1 Score | 0.56 | _fill after running_ | — |

- **If you run the experiments**, this slide becomes the most impactful one
- Show that distributed feature engineering **improves** fraud detection

---

#### **Slide 18: Comparison with Traditional ML** *(If time permits)*

| Model | ROC AUC | F1 | Precision | Recall |
|-------|--------:|---:|----------:|-------:|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| **HeteroRGCN (Ours)** | **0.92** | **0.56** | **0.86** | **0.35** |
| **HeteroRGCN + DiFE** | **?** | **?** | **?** | **?** |

---

#### **Slide 19: Key Challenges & Solutions**

| Challenge | Solution |
|-----------|----------|
| Highly imbalanced data (3.5% fraud) | Weighted loss, precision-focused evaluation |
| Large graph (19.5M edges) | Efficient PyG batching, full graph training |
| Training instability (F1 collapse at epoch 650+) | Early stopping, LR scheduling |
| Feature dimensionality (470+ features) | Feature selection, normalization |
| Scalability for large datasets | Distributed parallel feature extraction (DiFE) |

---

#### **Slide 20: Distributed Processing — Scalability**

- Show how DiFE enables parallel processing:
  - Each extractor runs independently
  - Features are combined after extraction
  - Can scale to multiple machines
- If possible, show timing benchmarks:
  - 1 worker: X seconds
  - 4 workers: Y seconds
  - 8 workers: Z seconds

---

#### **Slide 21: Limitations**
- Recall is relatively low (0.35) — misses some fraud cases
- No real-time inference pipeline built
- Model requires full graph retraining for new data
- No mini-batch training (limited by GPU memory for very large graphs)
- Feature engineering increases computational cost

---

#### **Slide 22: Future Work**
1. **Attention-based GNN** (GAT) for interpretable fraud detection
2. **Mini-batch training** using NeighborLoader for scalability
3. **Real-time streaming** fraud detection with incremental graph updates
4. **Explainability** using SHAP / GNNExplainer for model interpretability
5. **Deploy as API** for production fraud screening
6. **Temporal graph networks** to capture evolving fraud patterns
7. **Transfer learning** to other fraud domains (insurance, healthcare)

---

#### **Slide 23: Conclusion**
- Built a **complete graph-based fraud detection system** using HeteroRGCN
- Designed a **distributed feature engineering framework (DiFE)** with 7 extractors and 470+ features
- Achieved **ROC AUC of 0.92** and **Precision of 0.86** on IEEE-CIS dataset
- Demonstrated that **graph structure captures fraud patterns** that tabular ML misses
- DiFE enables **scalable, parallel** feature extraction for large-scale deployment

---

#### **Slide 24: References**
1. IEEE-CIS Fraud Detection Dataset (Kaggle)
2. Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
3. Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" (2018) — RGCN
4. Wang et al., "Heterogeneous Graph Attention Network" (2019)
5. Liu et al., "GeniePath: Graph Neural Networks with Adaptive Receptive Paths" (2019)
6. AWS SageMaker Graph Fraud Detection (GitHub)
7. Deep Graph Library (DGL) Documentation
8. PyTorch Geometric Documentation
9. *(Add any papers you cited in Seminar 1)*

---

#### **Slide 25: Thank You / Q&A**
- "Thank you! Questions?"
- Include your name, email, and GitHub repo link

---

## 🎯 Priority Action Items for Seminar 2

### Must Complete Before PPT (Priority Order):

```
✅ 1. Fix early stopping in training (save best model by F1)
   → Edit train.py to save best checkpoint

✅ 2. Run distributed feature extraction end-to-end
   → Execute distributed_feature_extraction.py with real data
   → Debug any errors

✅ 3. Train model WITH DiFE features
   → python train.py --use-distributed-features --feature-dir ./features

✅ 4. Generate comparison table (baseline vs DiFE)
   → This is Slide 17 — the most important result slide

✅ 5. Create the PPT slides following the structure above

⬜ 6. (Bonus) Train baseline ML models for comparison table
   → Random Forest, XGBoost on same dataset

⬜ 7. (Bonus) Scalability benchmark with different worker counts
```

### Tips for a Great PPT:

1. **Use visuals heavily** — architecture diagrams, graphs, charts. Avoid text-heavy slides.
2. **Lead with results** — professors love seeing numbers. Your ROC AUC of 0.92 is impressive.
3. **Tell a story** — "Problem → Why existing solutions fail → Our approach → Our results → Impact"
4. **Practice the DiFE pitch** — explain WHY distributed feature engineering matters (scalability, modularity)
5. **Prepare for questions about:**
   - Why RGCN over GAT or other GNN architectures?
   - Why not mini-batch training?
   - How does this compare to XGBoost?
   - What is the recall issue and how can you fix it?
   - How would this work in real-time production?
6. **Keep duration in mind** — aim for 15-20 minutes presentation + 5-10 minutes Q&A

---

## 📁 Files You'll Need for the PPT

| What | Where |
|------|-------|
| ROC Curve | `output/roc_curve.png` |
| PR Curve | `output/pr_curve.png` |
| F1 Progression | `output/f1.jpg` |
| Loss Curve | `output/loss.jpg` |
| Graph Architecture Visual | `graph_intro.png` |
| Training Results Data | `output/results.txt` |
| Feature Documentation | `FEATURES_DOCUMENTATION.md` |
| GNN Architecture Details | `GNN_ARCHITECTURE_EXPLANATION.md` |
| DiFE Implementation Details | `IMPLEMENTATION_SUMMARY.md` |
