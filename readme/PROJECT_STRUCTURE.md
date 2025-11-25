# Project Structure

This document explains the organized project structure after documentation reorganization.

---

## 📁 Root Directory

The root directory now contains only essential project files:

```
GNN-project/
├── README.md                    # Main project README (points to readme/ folder)
├── config.yaml                  # Main configuration file
├── docker-compose.yml           # Docker services configuration
├── Dockerfile                   # Docker image definition
├── requirements.txt            # Python dependencies
├── main.py                      # Main orchestration script
│
├── readme/                      # 📚 ALL DOCUMENTATION (organized)
│   ├── README.md               # Documentation index
│   ├── setup/                  # Setup guides
│   ├── architecture/           # Architecture docs
│   ├── features/               # Feature docs
│   ├── guides/                 # Step-by-step guides
│   └── troubleshooting/        # Troubleshooting guides
│
├── api/                         # FastAPI application
├── feature_engineering/         # Feature extraction
├── gnn_training/               # GNN model training
├── graph_processing/           # Graph construction
├── storage/                     # Storage backends & data generation
├── scripts/                     # Utility scripts
│
├── data/                        # Data directory
├── model/                       # Trained models
├── output/                      # Training outputs
└── logs/                        # Log files
```

---

## 📚 Documentation Organization

### **`readme/`** - All Documentation

All markdown documentation files have been moved here and organized by category:

#### **`readme/setup/`** - Setup & Installation
- `COMPLETE_SETUP_GUIDE.md` - Full setup from scratch
- `SETUP_CHECKLIST.md` - Step-by-step checklist
- `HOW_TO_RUN_SERVICES.md` - Running Docker services
- `QUICK_DOCKER_START.md` - Quick Docker setup
- `QUICKSTART.md` - Fast setup guide

#### **`readme/architecture/`** - Architecture & Design
- `PROJECT_OVERVIEW.md` - Complete project overview
- `PROJECT_DIAGRAMS.md` - Visual architecture diagrams
- `DATA_LIFECYCLE_ARCHITECTURE.md` - Detailed data flow
- `DISTRIBUTED_COMPUTING_STATUS.md` - Distributed computing status
- `DISTRIBUTED_IMPLEMENTATION_GUIDE.md` - Distributed feature extraction guide

#### **`readme/features/`** - Feature Engineering
- `FEATURES_DOCUMENTATION.md` - All 62 features explained

#### **`readme/guides/`** - Step-by-Step Guides
- `NEXT_STEPS.md` - What to do after data generation
- `WHAT_TO_DO_NEXT.md` - Implementation roadmap
- `STORAGE_OPTIONS.md` - Storage backend options

#### **`readme/troubleshooting/`** - Troubleshooting
- `FIX_NEO4J_EMPTY.md` - Fix empty Neo4j graph
- `POSTGRES_CHOICE_EXPLANATION.md` - PostgreSQL setup explanation
- `DOCKER_DEPLOYMENT.md` - Docker deployment guide

---

## 🎯 Benefits of This Organization

### **Before:**
- 18+ markdown files in root directory
- Hard to find specific documentation
- Cluttered project structure

### **After:**
- Clean root directory
- Logical categorization
- Easy navigation via `readme/README.md`
- Professional project structure

---

## 🔍 Finding Documentation

### **Quick Access:**
- **Main Index**: [`readme/README.md`](./README.md)
- **Root README**: Points to organized documentation

### **By Category:**
- Need setup help? → `readme/setup/`
- Understanding architecture? → `readme/architecture/`
- Feature details? → `readme/features/`
- Step-by-step guide? → `readme/guides/`
- Having issues? → `readme/troubleshooting/`

---

## 📝 File Locations Reference

| Document | Old Location | New Location |
|----------|--------------|--------------|
| Complete Setup Guide | `COMPLETE_SETUP_GUIDE.md` | `readme/setup/COMPLETE_SETUP_GUIDE.md` |
| Project Overview | `PROJECT_OVERVIEW.md` | `readme/architecture/PROJECT_OVERVIEW.md` |
| Features Documentation | `FEATURES_DOCUMENTATION.md` | `readme/features/FEATURES_DOCUMENTATION.md` |
| Next Steps | `NEXT_STEPS.md` | `readme/guides/NEXT_STEPS.md` |
| Troubleshooting | `FIX_NEO4J_EMPTY.md` | `readme/troubleshooting/FIX_NEO4J_EMPTY.md` |

---

## 🚀 Quick Start

1. **Read the main README**: [`README.md`](../README.md) (in root)
2. **Browse documentation**: [`readme/README.md`](./README.md)
3. **Start setup**: [`readme/setup/COMPLETE_SETUP_GUIDE.md`](./setup/COMPLETE_SETUP_GUIDE.md)

---

**Project structure is now clean and professional!** 🎉

