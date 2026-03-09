# 🧠 Intelligent VM Migration Strategy Using Machine Learning

A complete simulation + ML project that predicts virtual machine overload on physical hosts and automatically triggers live migration to balance workloads.

---

## 📌 Overview

| Component | Description |
|-----------|-------------|
| **Simulation** | 5 hosts × 20 VMs with dynamic CPU/RAM/Network metrics |
| **ML Model** | Random Forest Classifier predicting host overload |
| **Decision Engine** | Real-time overload prediction + automatic VM migration |
| **Evaluation** | SLA violations, energy consumption, migration count |
| **Comparison** | Rule-based vs ML-based strategy analysis with plots |
| **Dashboard** | CPU/RAM trends, migration frequency, energy savings |
| **AWS (Optional)** | EC2 launch, CloudWatch monitoring, live migration |

---

## 📁 Project Structure

```
vm_migration_project/
│
├── data/                          # Generated CSV datasets
│   └── vm_metrics.csv
│
├── simulation/                    # Simulation layer
│   ├── __init__.py
│   ├── vm.py                      # Virtual Machine class
│   ├── host.py                    # Physical Host class
│   ├── simulator.py               # Orchestrator (5 hosts, 20 VMs)
│   ├── data_generator.py          # CSV export
│   └── migration.py               # VM migration logic
│
├── model/                         # Machine Learning
│   ├── __init__.py
│   ├── preprocess.py              # Data cleaning & normalization
│   ├── train.py                   # Random Forest training & evaluation
│   ├── trained_model.pkl          # Saved model (after training)
│   ├── scaler.pkl                 # Saved scaler (after training)
│   └── confusion_matrix.png       # Confusion matrix plot
│
├── decision/                      # Decision engine
│   ├── __init__.py
│   └── engine.py                  # ML-based migration decisions
│
├── evaluation/                    # Strategy evaluation
│   ├── __init__.py
│   ├── evaluate.py                # SLA, energy, migration metrics
│   ├── compare.py                 # Rule-based vs ML-based comparison
│   └── comparison.png             # Comparison chart
│
├── dashboard/                     # Visualization
│   ├── __init__.py
│   ├── plots.py                   # All dashboard plots
│   ├── cpu_ram_trends.png
│   ├── host_utilization_heatmap.png
│   ├── migration_frequency.png
│   └── energy_comparison.png
│
├── aws/                           # AWS EC2 integration (optional)
│   ├── __init__.py
│   └── ec2_manager.py             # EC2 launch, CloudWatch, migration
│
├── main.py                        # Entry point with menu
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone or navigate to the project
cd vm_migration_project

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Run all steps automatically
python main.py --all

# OR use the interactive menu
python main.py
```

---

## 📝 Step-by-Step Guide

### Step 1: Simulation & Data Generation
Creates 5 hosts with 20 VMs, runs 50 simulation ticks, and saves metrics to CSV.

```bash
# Select option [1] in the menu, or run --all
```

**Output:** `data/vm_metrics.csv` with columns:
`tick, host_id, vm_id, cpu, ram, network, total_host_cpu, total_host_ram, overloaded`

### Step 2: Data Preprocessing
- Cleans missing values
- Normalizes features using MinMaxScaler (0-1 range)
- Creates binary labels: `1 = overloaded, 0 = normal`
- Splits into 80% train / 20% test

### Step 3: Train ML Model
- Algorithm: **Random Forest Classifier** (100 trees)
- Metrics: Accuracy, Precision, Recall, F1-Score
- Outputs: `model/trained_model.pkl`, `model/confusion_matrix.png`

### Step 4: Decision Engine
- Loads the trained model
- Runs real-time simulation ticks
- Predicts overload for each host
- If overloaded → migrates highest-CPU VM to least-loaded host
- Logs all migration events with before/after status

### Step 5: Strategy Comparison
Runs both strategies on identical seeded simulations:

| Metric | Rule-Based | ML-Based |
|--------|-----------|----------|
| Trigger | CPU > 85% | ML prediction |
| SLA Violations | Measured | Measured |
| Energy | Estimated | Estimated |
| Migrations | Counted | Counted |

**Output:** `evaluation/comparison.png`

### Step 6: Visualization Dashboard
Generates publication-quality plots:
- CPU/RAM usage trends over time
- Host utilization heatmap
- Migration frequency (from/to each host)
- Energy consumption comparison

### Step 7: AWS Integration (Optional)
Works in **simulation mode** without AWS credentials.

For real AWS deployment:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Simulation  │────▶│  CSV Dataset │────▶│ Preprocessing│
│  (5H × 20VM)│     │  vm_metrics  │     │  (Normalize) │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Migration  │◀────│   Decision   │◀────│  ML Model    │
│  (VM Move)  │     │   Engine     │     │(RandomForest)│
└──────┬──────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│  Evaluation │────▶│  Dashboard   │
│ (SLA/Energy)│     │  (Plots)     │
└─────────────┘     └──────────────┘
```

---

## 📊 PPT Outline

1. **Title Slide** — Intelligent VM Migration Strategy Using Machine Learning
2. **Problem Statement** — Manual VM migration is reactive, causes SLA violations
3. **Objective** — Proactive, ML-driven migration to reduce overloads
4. **Architecture** — System diagram (see above)
5. **Simulation Layer** — 5 hosts, 20 VMs, dynamic resource generation
6. **ML Model** — Random Forest, dataset, preprocessing pipeline
7. **Decision Engine** — Real-time prediction → migration trigger
8. **Results** — Accuracy metrics, confusion matrix
9. **Comparison** — Rule-based vs ML-based bar charts
10. **Dashboard** — CPU/RAM trends, heatmaps, energy plots
11. **AWS Deployment** — EC2 + CloudWatch integration
12. **Conclusion** — ML reduces SLA violations, improves energy efficiency
13. **Future Work** — Deep learning, reinforcement learning, live cloud deployment

---

## 📄 License

This project is for educational and academic purposes.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

*Built with Python, scikit-learn, matplotlib, and seaborn*
