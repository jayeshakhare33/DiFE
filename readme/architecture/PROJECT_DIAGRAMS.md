# Project Diagrams - Fraud Detection GNN System

This document contains all architectural and workflow diagrams for the Fraud Detection System using Graph Neural Networks.

---

## 1. Project Structure

```
GNN-project/
│
├── api/                          # FastAPI REST API
│   ├── app.py                   # Main API application
│   └── inference.py             # Inference service
│
├── feature_engineering/         # Feature extraction
│   ├── feature_extractor.py     # Node feature extraction (50 features)
│   └── graph_embeddings.py      # Graph embedding generation
│
├── gnn_training/                # Model training
│   ├── gnn_model.py             # GNN model architecture
│   └── distributed_trainer.py  # Distributed training logic
│
├── graph_processing/            # Graph construction
│   ├── graph_builder.py         # Build graph from transactions
│   └── graph_partitioner.py     # Graph partitioning for distributed processing
│
├── storage/                      # Feature storage backends
│   ├── storage_backend.py       # Abstract storage interface
│   ├── feature_store.py         # Feature store implementation
│   └── datagen.py               # Data generation utilities
│
├── scripts/                      # Utility scripts
│   ├── init_postgres.sql        # PostgreSQL schema
│   ├── init_neo4j.cypher        # Neo4j graph structure
│   ├── extract_all_features.py  # Feature extraction pipeline
│   └── test_*.py                 # Connection test scripts
│
├── data/                         # Data storage
│   └── features/                # Extracted features (Parquet)
│
├── config.yaml                   # Configuration file
├── main.py                       # Main orchestration script
├── docker-compose.yml           # Docker services definition
└── requirements.txt             # Python dependencies
```

---

## 2. System Architecture Diagram

```mermaid
graph TB
    subgraph "Data Sources"
        PG[(PostgreSQL<br/>Transaction Database)]
        CSV[CSV Files<br/>Transaction Data]
    end

    subgraph "Graph Database"
        NEO4J[(Neo4j<br/>Graph Database)]
    end

    subgraph "Message Broker"
        KAFKA[Apache Kafka<br/>Message Queue]
        ZK[Zookeeper<br/>Coordination]
    end

    subgraph "Feature Store"
        REDIS[(Redis<br/>Feature Cache)]
        PARQUET[Parquet Files<br/>Persistent Storage]
    end

    subgraph "Processing Layer"
        GB[Graph Builder]
        FE[Feature Extractor]
        GT[GNN Trainer]
    end

    subgraph "API Layer"
        API[FastAPI<br/>REST API]
        INF[Inference Service]
    end

    PG -->|Sync| KAFKA
    CSV -->|Load| GB
    KAFKA -->|Consume| NEO4J
    GB -->|Store| NEO4J
    NEO4J -->|Query| FE
    FE -->|Store| REDIS
    FE -->|Store| PARQUET
    REDIS -->|Read| GT
    PARQUET -->|Read| GT
    GT -->|Model| INF
    REDIS -->|Read| INF
    INF -->|Serve| API
    ZK -->|Coordinate| KAFKA

    style PG fill:#e1f5ff
    style NEO4J fill:#e1f5ff
    style REDIS fill:#ffe1f5
    style KAFKA fill:#fff5e1
    style API fill:#e1ffe1
```

---

## 3. Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Phase 1: Data Ingestion"
        A[Transaction Data<br/>PostgreSQL/CSV] -->|Batch/CDC| B[Kafka Topic<br/>transactions-raw]
        B -->|Consume| C[Neo4j Loader]
        C -->|Create Graph| D[Neo4j Graph<br/>Nodes & Edges]
    end

    subgraph "Phase 2: Feature Extraction"
        D -->|Query Subgraph| E[Feature Extraction<br/>Workers]
        E -->|Extract 50 Node<br/>+ 12 Edge Features| F[Kafka Topic<br/>features-computed]
        F -->|Aggregate| G[Feature Aggregator]
        G -->|Store| H[Redis Feature Store]
        G -->|Backup| I[Parquet Files]
    end

    subgraph "Phase 3: Model Training"
        H -->|Load Features| J[GNN Trainer<br/>Distributed]
        I -->|Load Features| J
        J -->|Train Model| K[Trained Model<br/>model.pth]
    end

    subgraph "Phase 4: Inference"
        L[API Request] -->|Query| H
        H -->|Get Features| M[Inference Service]
        K -->|Load Model| M
        M -->|Predict| N[Fraud Score<br/>Response]
    end

    style A fill:#e1f5ff
    style D fill:#e1f5ff
    style H fill:#ffe1f5
    style K fill:#e1ffe1
    style N fill:#ffe1f5
```

---

## 4. User Workflow Diagram

```mermaid
flowchart TD
    Start([User Starts System]) --> Setup{Setup Complete?}
    
    Setup -->|No| Init[Initialize Services<br/>- Start Docker Services<br/>- Initialize Databases<br/>- Create Kafka Topics]
    Init --> Load[Load Transaction Data<br/>to PostgreSQL]
    Load --> Sync[Sync to Neo4j Graph]
    Sync --> Extract[Extract Features<br/>50 Node + 12 Edge]
    Extract --> Train[Train GNN Model]
    Train --> Deploy[Deploy API Service]
    
    Setup -->|Yes| Deploy
    
    Deploy --> Wait[System Ready<br/>Waiting for Requests]
    
    Wait --> Request[Receive API Request<br/>POST /predict]
    Request --> Validate{Valid Request?}
    
    Validate -->|No| Error[Return Error<br/>400 Bad Request]
    Error --> Wait
    
    Validate -->|Yes| GetFeatures[Retrieve Features<br/>from Redis]
    GetFeatures --> LoadModel[Load GNN Model]
    LoadModel --> Predict[Run Inference<br/>Generate Predictions]
    Predict --> Response[Return Fraud Scores<br/>200 OK]
    Response --> Wait
    
    style Start fill:#e1ffe1
    style Wait fill:#fff5e1
    style Response fill:#e1ffe1
    style Error fill:#ffe1e1
```

---

## 5. Component Interaction Diagram

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Redis
    participant Neo4j
    participant Kafka
    participant PostgreSQL
    participant Trainer
    participant FeatureExtractor

    Note over PostgreSQL,FeatureExtractor: Initialization Phase
    PostgreSQL->>Kafka: Publish transactions
    Kafka->>Neo4j: Consume & create graph
    Neo4j->>FeatureExtractor: Query subgraph
    FeatureExtractor->>FeatureExtractor: Extract 62 features
    FeatureExtractor->>Redis: Store features
    FeatureExtractor->>Trainer: Provide features
    Trainer->>Trainer: Train GNN model
    
    Note over Client,API: Inference Phase
    Client->>API: POST /predict {node_ids}
    API->>Redis: GET features:node:{id}
    Redis-->>API: Return features
    API->>API: Load GNN model
    API->>API: Run inference
    API-->>Client: Return fraud scores
```

---

## 6. Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Network: fraud-detection-network"
        subgraph "Database Services"
            PG[(PostgreSQL<br/>:5432)]
            NEO[(Neo4j<br/>:7474, :7687)]
        end
        
        subgraph "Message Queue Services"
            ZK[Zookeeper<br/>:2181]
            KF[Kafka<br/>:9092, :9093]
            KUI[Kafka UI<br/>:8080]
        end
        
        subgraph "Storage Services"
            RD[(Redis<br/>:6379)]
        end
        
        subgraph "Application Services"
            API[API Service<br/>:8000]
            FL[Feature Loader<br/>One-time]
            T0[Trainer-0<br/>Master]
            T1[Trainer-1]
            T2[Trainer-2]
            T3[Trainer-3]
        end
    end
    
    subgraph "External Access"
        USER[Users/Client]
        ADMIN[Admin/Developer]
    end
    
    USER -->|HTTP| API
    ADMIN -->|HTTP| KUI
    ADMIN -->|HTTP| NEO
    ADMIN -->|psql| PG
    
    PG -->|Sync| KF
    KF -->|Consume| NEO
    NEO -->|Query| FL
    FL -->|Store| RD
    T0 -->|Coordinate| T1
    T0 -->|Coordinate| T2
    T0 -->|Coordinate| T3
    T0 -->|Read| RD
    T1 -->|Read| RD
    T2 -->|Read| RD
    T3 -->|Read| RD
    API -->|Read| RD
    API -->|Query| NEO
    
    ZK -->|Coordinate| KF
    
    style PG fill:#e1f5ff
    style NEO fill:#e1f5ff
    style RD fill:#ffe1f5
    style KF fill:#fff5e1
    style API fill:#e1ffe1
```

---

## 7. Feature Extraction Pipeline

```mermaid
flowchart TD
    Start([Start Feature Extraction]) --> LoadGraph[Load Graph from Neo4j]
    LoadGraph --> Query[Query Subgraph<br/>for Nodes & Edges]
    
    Query --> NodeFeatures[Extract Node Features]
    Query --> EdgeFeatures[Extract Edge Features]
    
    subgraph "Node Features (50 total)"
        NodeFeatures --> NF1[Transaction Statistics<br/>15 features]
        NodeFeatures --> NF2[Graph Topology<br/>12 features]
        NodeFeatures --> NF3[Temporal Features<br/>10 features]
        NodeFeatures --> NF4[Behavioral Features<br/>8 features]
        NodeFeatures --> NF5[Fraud Propagation<br/>5 features]
    end
    
    subgraph "Edge Features (12 total)"
        EdgeFeatures --> EF1[Amount & Timestamp<br/>5 features]
        EdgeFeatures --> EF2[Temporal Patterns<br/>3 features]
        EdgeFeatures --> EF3[Relationship Features<br/>4 features]
    end
    
    NF1 --> Aggregate[Aggregate Features]
    NF2 --> Aggregate
    NF3 --> Aggregate
    NF4 --> Aggregate
    NF5 --> Aggregate
    EF1 --> Aggregate
    EF2 --> Aggregate
    EF3 --> Aggregate
    
    Aggregate --> StoreRedis[Store in Redis<br/>Fast Access]
    Aggregate --> StoreParquet[Store in Parquet<br/>Persistent Backup]
    
    StoreRedis --> End([Features Ready])
    StoreParquet --> End
    
    style Start fill:#e1ffe1
    style End fill:#e1ffe1
    style Aggregate fill:#fff5e1
```

---

## 8. Training Pipeline Flow

```mermaid
flowchart TD
    Start([Start Training]) --> LoadData[Load Features<br/>from Redis/Parquet]
    LoadData --> LoadGraph[Load Graph Structure<br/>from Neo4j]
    
    LoadGraph --> Partition[Partition Graph<br/>for Distributed Training]
    
    Partition --> InitModel[Initialize GNN Model<br/>HeteroRGCN]
    InitModel --> InitTrainer[Initialize Distributed Trainer<br/>4 Workers]
    
    InitTrainer --> TrainLoop[Training Loop]
    
    TrainLoop --> Forward[Forward Pass<br/>Compute Predictions]
    Forward --> Loss[Calculate Loss<br/>Cross-Entropy]
    Loss --> Backward[Backward Pass<br/>Compute Gradients]
    Backward --> Update[Update Model Weights<br/>SGD Optimizer]
    
    Update --> Eval{Evaluation<br/>Epoch?}
    Eval -->|No| TrainLoop
    Eval -->|Yes| Metrics[Calculate Metrics<br/>Accuracy, F1, ROC-AUC]
    
    Metrics --> Check{Converged?}
    Check -->|No| TrainLoop
    Check -->|Yes| Save[Save Model<br/>model.pth + metadata]
    
    Save --> End([Training Complete])
    
    style Start fill:#e1ffe1
    style End fill:#e1ffe1
    style TrainLoop fill:#fff5e1
    style Save fill:#e1f5ff
```

---

## 9. Kafka Message Flow

```mermaid
graph LR
    subgraph "Producers"
        PG_P[PostgreSQL<br/>Connector]
        NEO_P[Neo4j<br/>Loader]
        FE_P[Feature<br/>Extractor]
    end
    
    subgraph "Kafka Topics"
        T1[transactions-raw<br/>6 partitions]
        T2[graph-updates<br/>3 partitions]
        T3[feature-extraction<br/>4 partitions]
        T4[features-computed<br/>4 partitions]
    end
    
    subgraph "Consumers"
        NEO_C[Neo4j<br/>Loader]
        FE_C[Feature<br/>Workers]
        FA_C[Feature<br/>Aggregator]
    end
    
    PG_P -->|Publish| T1
    T1 -->|Consume| NEO_C
    NEO_C -->|Publish| T2
    T2 -->|Trigger| FE_P
    FE_P -->|Publish| T3
    T3 -->|Consume| FE_C
    FE_C -->|Publish| T4
    T4 -->|Consume| FA_C
    FA_C -->|Store| Redis[(Redis)]
    
    style T1 fill:#fff5e1
    style T2 fill:#fff5e1
    style T3 fill:#fff5e1
    style T4 fill:#fff5e1
    style Redis fill:#ffe1f5
```

---

## 10. Graph Structure Diagram

```mermaid
graph LR
    subgraph "Node Types"
        U[User Node<br/>user_id, features]
        T[Transaction Node<br/>transaction_id, amount, timestamp]
        L[Location Node<br/>location_id, coordinates]
        D[Device Node<br/>device_id, type]
        C[Card Node<br/>card_id, type]
    end
    
    subgraph "Relationship Types"
        U -->|SENT| T
        T -->|TO| U
        U -->|LOCATED_AT| L
        T -->|USED_DEVICE| D
        U -->|HAS_CARD| C
    end
    
    style U fill:#e1f5ff
    style T fill:#ffe1f5
    style L fill:#e1ffe1
    style D fill:#fff5e1
    style C fill:#f5e1ff
```

---

## 11. API Endpoint Flow

```mermaid
flowchart TD
    Client[Client Application] -->|HTTP Request| API[FastAPI Server<br/>:8000]
    
    API --> Route{Route Handler}
    
    Route -->|GET /health| Health[Health Check<br/>Return Status]
    Route -->|POST /predict| Predict[Predict Endpoint]
    Route -->|GET /features| Features[Get Features Endpoint]
    
    Predict --> ValidateReq{Validate<br/>Request Body}
    ValidateReq -->|Invalid| Error1[400 Bad Request]
    ValidateReq -->|Valid| GetFeat[Get Features<br/>from Redis]
    
    GetFeat --> LoadMod[Load GNN Model]
    LoadMod --> RunInf[Run Inference]
    RunInf --> Format[Format Response]
    Format --> Return1[200 OK<br/>Fraud Scores]
    
    Features --> ValidateQuery{Validate<br/>Query Params}
    ValidateQuery -->|Invalid| Error2[400 Bad Request]
    ValidateQuery -->|Valid| QueryFeat[Query Features<br/>from Redis]
    QueryFeat --> Return2[200 OK<br/>Feature Data]
    
    Health --> Return3[200 OK<br/>Service Status]
    
    Return1 --> Client
    Return2 --> Client
    Return3 --> Client
    Error1 --> Client
    Error2 --> Client
    
    style Client fill:#e1ffe1
    style API fill:#fff5e1
    style Return1 fill:#e1f5ff
    style Return2 fill:#e1f5ff
    style Return3 fill:#e1f5ff
    style Error1 fill:#ffe1e1
    style Error2 fill:#ffe1e1
```

---

## 12. Distributed Training Architecture

```mermaid
graph TB
    subgraph "Master Node (Rank 0)"
        M[Trainer-0<br/>Master Process]
        M -->|Broadcast| W1
        M -->|Broadcast| W2
        M -->|Broadcast| W3
        M -->|Aggregate| AG[Aggregate Gradients]
    end
    
    subgraph "Worker Nodes"
        W1[Trainer-1<br/>Rank 1]
        W2[Trainer-2<br/>Rank 2]
        W3[Trainer-3<br/>Rank 3]
    end
    
    subgraph "Shared Resources"
        G[Graph Partition 0]
        G1[Graph Partition 1]
        G2[Graph Partition 2]
        G3[Graph Partition 3]
        F[(Feature Store<br/>Redis/Parquet)]
        M_STORE[(Model Storage)]
    end
    
    M -->|Load| G
    M -->|Read| F
    W1 -->|Load| G1
    W1 -->|Read| F
    W2 -->|Load| G2
    W2 -->|Read| F
    W3 -->|Load| G3
    W3 -->|Read| F
    
    W1 -->|Send Gradients| AG
    W2 -->|Send Gradients| AG
    W3 -->|Send Gradients| AG
    
    AG -->|Update Model| M_STORE
    M_STORE -->|Sync| M
    M_STORE -->|Sync| W1
    M_STORE -->|Sync| W2
    M_STORE -->|Sync| W3
    
    style M fill:#e1ffe1
    style AG fill:#fff5e1
    style M_STORE fill:#e1f5ff
```

---

## Diagram Usage Notes

### For Reports:
- **Mermaid diagrams** render in most modern markdown viewers (GitHub, GitLab, VS Code, etc.)
- For presentations, you can:
  1. Use online Mermaid editors (mermaid.live) to export as PNG/SVG
  2. Use tools like `mermaid-cli` to convert to images
  3. Copy ASCII diagrams directly into documents

### Diagram Types:
1. **Project Structure** - Shows code organization
2. **System Architecture** - High-level component overview
3. **Data Flow** - How data moves through the system
4. **User Workflow** - End-to-end user journey
5. **Component Interaction** - Sequence of operations
6. **Deployment Architecture** - Docker container layout
7. **Feature Extraction Pipeline** - Feature computation flow
8. **Training Pipeline** - Model training process
9. **Kafka Message Flow** - Message queue architecture
10. **Graph Structure** - Neo4j graph schema
11. **API Endpoint Flow** - Request/response handling
12. **Distributed Training** - Multi-worker training setup

---

*Last Updated: 2024*

