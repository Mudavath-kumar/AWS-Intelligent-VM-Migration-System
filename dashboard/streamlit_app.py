"""
streamlit_app.py - VM Migration Project Dashboard
===================================================
Interactive Streamlit dashboard to visualize all project outputs.

Sections:
  1. Project Overview
  2. ML Model Performance
  3. Evaluation Comparison
  4. Dashboard Visualizations
  5. Dataset Preview
  6. Migration Logs

Run:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "vm_metrics.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
DASHBOARD_DIR = os.path.dirname(__file__)


def main():
    """Main dashboard entry point."""
    if not STREAMLIT_AVAILABLE:
        print("[DASHBOARD] Streamlit not installed.")
        print("  Install with: pip install streamlit")
        return

    # Page configuration
    st.set_page_config(
        page_title="VM Migration Dashboard",
        page_icon="🖥️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main Title
    st.markdown('<p class="main-header">🖥️ Intelligent VM Migration Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Machine Learning-based VM Workload Prediction & Migration Strategy**")
    st.markdown("---")

    # Sidebar Navigation
    st.sidebar.title("📑 Navigation")
    section = st.sidebar.radio("Go to Section", [
        "🏠 Project Overview",
        "🤖 ML Model Performance",
        "📊 Evaluation Comparison",
        "📈 Dashboard Visualizations",
        "📋 Dataset Preview",
        "📝 Migration Logs"
    ])

    # Route to sections
    if section == "🏠 Project Overview":
        section_project_overview()
    elif section == "🤖 ML Model Performance":
        section_ml_performance()
    elif section == "📊 Evaluation Comparison":
        section_evaluation()
    elif section == "📈 Dashboard Visualizations":
        section_dashboard_plots()
    elif section == "📋 Dataset Preview":
        section_dataset_preview()
    elif section == "📝 Migration Logs":
        section_migration_logs()

    # Footer
    st.markdown("---")
    st.markdown(
        "<small>🔧 Built with Streamlit | "
        "📊 Powered by Random Forest ML | "
        "☁️ AWS EC2 Integration Ready</small>",
        unsafe_allow_html=True
    )


# ============================================================================
# SECTION 1: Project Overview
# ============================================================================
def section_project_overview():
    """Display project overview with key statistics."""
    st.markdown('<p class="section-header">🏠 Project Overview</p>', unsafe_allow_html=True)

    # Load data for statistics
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        num_hosts = df["host_id"].nunique()
        num_vms = df["vm_id"].nunique()
        dataset_size = len(df)
        num_ticks = df["tick"].nunique()
        overloaded_count = df["overloaded"].sum()
        overload_rate = (overloaded_count / len(df)) * 100
    else:
        num_hosts = 5
        num_vms = 20
        dataset_size = 0
        num_ticks = 0
        overloaded_count = 0
        overload_rate = 0

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🖧 Number of Hosts",
            value=num_hosts,
            help="Physical servers in the simulation"
        )

    with col2:
        st.metric(
            label="💻 Number of VMs",
            value=num_vms,
            help="Virtual machines distributed across hosts"
        )

    with col3:
        st.metric(
            label="📊 Dataset Size",
            value=f"{dataset_size:,}",
            help="Total records in vm_metrics.csv"
        )

    with col4:
        st.metric(
            label="⏱️ Simulation Ticks",
            value=num_ticks,
            help="Number of simulation time steps"
        )

    st.markdown("")

    # Second row of metrics
    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric(
            label="⚠️ Overload Events",
            value=f"{overloaded_count:,}",
            delta=f"{overload_rate:.1f}% of records",
            delta_color="inverse"
        )

    with col6:
        st.metric(
            label="🎯 ML Model",
            value="Random Forest",
            help="Classification algorithm used"
        )

    with col7:
        st.metric(
            label="☁️ Cloud Integration",
            value="AWS EC2",
            help="Production deployment target"
        )

    # Project description
    st.markdown("### 📖 About This Project")
    st.info("""
    **Intelligent VM Migration Strategy Using Machine Learning** is a system that:
    
    - 🔮 **Predicts** host overload using ML (Random Forest classifier)
    - 🔄 **Migrates** VMs proactively to prevent SLA violations
    - ⚡ **Optimizes** energy consumption across the data center
    - 📊 **Compares** ML-based vs rule-based migration strategies
    - ☁️ **Integrates** with AWS EC2 for real cloud deployment
    
    **Pipeline:** Simulation → ML Training → Decision Engine → Evaluation → AWS Deployment
    """)

    # Architecture diagram (text-based)
    st.markdown("### 🏗️ System Architecture")
    st.code("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    SIMULATION LAYER                          │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
    │  │ Host_1  │  │ Host_2  │  │ Host_3  │  │ Host_4  │  ...    │
    │  │ VM1,VM2 │  │ VM3,VM4 │  │ VM5,VM6 │  │ VM7,VM8 │         │
    │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
    └──────────────────────┬──────────────────────────────────────┘
                           │ Metrics (CPU, RAM, Network)
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    ML MODEL LAYER                            │
    │  ┌──────────────┐    ┌────────────────┐    ┌─────────────┐  │
    │  │ Preprocessing │ →  │ Random Forest  │ →  │ Prediction  │  │
    │  │   (Scaler)   │    │  Classifier    │    │ (Overload?) │  │
    │  └──────────────┘    └────────────────┘    └─────────────┘  │
    └──────────────────────────┬──────────────────────────────────┘
                               │ Overload Probability
                               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  DECISION ENGINE                             │
    │  ┌────────────────┐         ┌─────────────────────┐         │
    │  │ Cost-Aware VM  │    OR   │  Simple (Max CPU)   │         │
    │  │   Selection    │         │    Selection        │         │
    │  └───────┬────────┘         └─────────┬───────────┘         │
    │          └────────────┬───────────────┘                     │
    │                       ▼                                      │
    │              Migrate VM to Least Loaded Host                 │
    └──────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 AWS INTEGRATION (Optional)                   │
    │  CloudWatch Metrics → ML Prediction → EC2 Scale Out/In      │
    └─────────────────────────────────────────────────────────────┘
    """, language="text")


# ============================================================================
# SECTION 2: ML Model Performance
# ============================================================================
def section_ml_performance():
    """Display ML model performance visualizations."""
    st.markdown('<p class="section-header">🤖 ML Model Performance</p>', unsafe_allow_html=True)

    st.markdown("""
    The Random Forest classifier was trained to predict host overload based on:
    - **Features:** CPU, RAM, Network, Total Host CPU, Total Host RAM
    - **Target:** Binary classification (0 = Normal, 1 = Overloaded)
    """)

    # Confusion Matrix
    st.markdown("### 📊 Confusion Matrix")
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(cm_path, caption="Confusion Matrix - Model Predictions vs Actual", use_container_width=True)
    else:
        st.warning("⚠️ Confusion matrix not found. Run model training first (Step 3).")

    st.markdown("---")

    # Feature Importance
    st.markdown("### 📈 Feature Importance")
    fi_path = os.path.join(MODEL_DIR, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Feature Importance - Which features matter most", use_container_width=True)
    else:
        st.warning("⚠️ Feature importance plot not found. Run model training first (Step 3).")

    st.markdown("---")

    # ROC and PR Curves
    st.markdown("### 📉 ROC & Precision-Recall Curves")
    roc_path = os.path.join(MODEL_DIR, "roc_pr_curves.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curve (left) and Precision-Recall Curve (right)", use_container_width=True)
    else:
        st.warning("⚠️ ROC/PR curves not found. Run model training first (Step 3).")

    # Model info box
    st.markdown("### ℹ️ Model Information")
    st.success("""
    **Model Details:**
    - **Algorithm:** Random Forest Classifier
    - **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
    - **Parameters:** n_estimators, max_depth, min_samples_split, min_samples_leaf
    - **Saved Model:** `model/trained_model.pkl`
    - **Saved Scaler:** `model/scaler.pkl`
    """)


# ============================================================================
# SECTION 3: Evaluation Comparison
# ============================================================================
def section_evaluation():
    """Display strategy comparison visualization."""
    st.markdown('<p class="section-header">📊 Evaluation Comparison</p>', unsafe_allow_html=True)

    st.markdown("""
    **Comparing two migration strategies:**
    - **Rule-Based:** Simple threshold (CPU > 85%) → Migrate highest-CPU VM
    - **ML-Based:** Predict overload with Random Forest → Cost-aware VM selection
    """)

    # Comparison chart
    comparison_path = os.path.join(EVAL_DIR, "comparison.png")
    if os.path.exists(comparison_path):
        st.image(comparison_path, caption="Rule-Based vs ML-Based Strategy Comparison", use_container_width=True)
    else:
        st.warning("⚠️ Comparison chart not found. Run evaluation first (Step 5).")

    # Metrics explanation
    st.markdown("### 📋 Evaluation Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Performance Metrics:**
        - **SLA Violations:** Hosts exceeding 90% CPU threshold
        - **Migration Count:** Total VMs migrated during simulation
        - **Load Imbalance:** Variance of CPU across hosts
        """)

    with col2:
        st.markdown("""
        **Energy Metrics:**
        - **Total Energy:** Sum of power consumption across hosts
        - **Power Model:** P = P_idle + (P_max - P_idle) × utilization
        - **Efficiency:** Energy per unit workload processed
        """)


# ============================================================================
# SECTION 4: Dashboard Visualizations
# ============================================================================
def section_dashboard_plots():
    """Display dashboard visualization plots."""
    st.markdown('<p class="section-header">📈 Dashboard Visualizations</p>', unsafe_allow_html=True)

    # CPU/RAM Trends
    st.markdown("### 📊 CPU & RAM Trends Over Time")
    trends_path = os.path.join(DASHBOARD_DIR, "cpu_ram_trends.png")
    if os.path.exists(trends_path):
        st.image(trends_path, caption="CPU and RAM utilization trends across hosts", use_container_width=True)
    else:
        st.warning("⚠️ CPU/RAM trends plot not found. Run visualization step (Step 6).")

    st.markdown("---")

    # Host Utilization Heatmap
    st.markdown("### 🔥 Host Utilization Heatmap")
    heatmap_path = os.path.join(DASHBOARD_DIR, "host_utilization_heatmap.png")
    if os.path.exists(heatmap_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(heatmap_path, caption="Host CPU utilization over time (darker = higher load)", use_container_width=True)
    else:
        st.warning("⚠️ Heatmap not found. Run visualization step (Step 6).")

    st.markdown("---")

    # Migration Frequency and Energy Comparison side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔀 Migration Frequency")
        mig_path = os.path.join(DASHBOARD_DIR, "migration_frequency.png")
        if os.path.exists(mig_path):
            st.image(mig_path, caption="Migrations per host", use_container_width=True)
        else:
            st.warning("⚠️ Migration frequency plot not found.")

    with col2:
        st.markdown("### ⚡ Energy Comparison")
        energy_path = os.path.join(DASHBOARD_DIR, "energy_comparison.png")
        if os.path.exists(energy_path):
            st.image(energy_path, caption="Energy consumption by strategy", use_container_width=True)
        else:
            st.warning("⚠️ Energy comparison plot not found.")


# ============================================================================
# SECTION 5: Dataset Preview
# ============================================================================
def section_dataset_preview():
    """Display dataset preview table."""
    st.markdown('<p class="section-header">📋 Dataset Preview</p>', unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset not found. Run simulation first (Step 1).")
        return

    df = pd.read_csv(DATA_PATH)

    # Dataset info
    st.markdown("### 📁 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Overloaded", f"{df['overloaded'].sum():,}")
    with col4:
        st.metric("Normal", f"{(df['overloaded'] == 0).sum():,}")

    # Column descriptions
    st.markdown("### 📝 Column Descriptions")
    st.markdown("""
    | Column | Description |
    |--------|-------------|
    | `tick` | Simulation time step |
    | `host_id` | Physical host identifier |
    | `vm_id` | Virtual machine identifier |
    | `cpu` | VM CPU utilization (%) |
    | `ram` | VM RAM utilization (%) |
    | `network` | VM network utilization (%) |
    | `total_host_cpu` | Average CPU across all VMs on host |
    | `total_host_ram` | Average RAM across all VMs on host |
    | `overloaded` | Binary label (1 = host overloaded) |
    """)

    # Preview options
    st.markdown("### 🔍 Data Preview")
    num_rows = st.slider("Number of rows to display", 5, 50, 10)
    show_option = st.radio("Show:", ["First N rows", "Last N rows", "Random sample"], horizontal=True)

    if show_option == "First N rows":
        preview_df = df.head(num_rows)
    elif show_option == "Last N rows":
        preview_df = df.tail(num_rows)
    else:
        preview_df = df.sample(min(num_rows, len(df)))

    st.dataframe(preview_df, use_container_width=True)

    # Download button
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="vm_metrics.csv",
        mime="text/csv"
    )

    # Basic statistics
    st.markdown("### 📊 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)


# ============================================================================
# SECTION 6: Migration Logs
# ============================================================================
def section_migration_logs():
    """Display migration logs from the decision engine."""
    st.markdown('<p class="section-header">📝 Migration Logs</p>', unsafe_allow_html=True)

    st.markdown("""
    Migration events logged by the Decision Engine during simulation.
    Each migration moves a VM from an overloaded host to the least-loaded host.
    """)

    # Try to get migration log from the module
    try:
        from simulation.migration import get_migration_log
        log = get_migration_log()
    except Exception:
        log = []

    if log:
        st.success(f"✅ Found {len(log)} migration events")
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True)

        # Summary stats
        if "source_host" in log_df.columns:
            st.markdown("### 📊 Migration Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Migrations FROM each host:**")
                st.dataframe(log_df["source_host"].value_counts().reset_index())
            with col2:
                st.markdown("**Migrations TO each host:**")
                st.dataframe(log_df["target_host"].value_counts().reset_index())
    else:
        st.info("ℹ️ No migration events recorded yet.")

        # Show sample log format
        st.markdown("### 📋 Sample Migration Log Format")
        sample_log = pd.DataFrame([
            {
                "timestamp": "2026-03-09 10:15:23",
                "vm_id": "VM_05",
                "source_host": "Host_2",
                "target_host": "Host_4",
                "cpu_before": 92.5,
                "cpu_after": 68.3
            },
            {
                "timestamp": "2026-03-09 10:15:24",
                "vm_id": "VM_12",
                "source_host": "Host_1",
                "target_host": "Host_5",
                "cpu_before": 88.7,
                "cpu_after": 71.2
            },
            {
                "timestamp": "2026-03-09 10:15:25",
                "vm_id": "VM_08",
                "source_host": "Host_3",
                "target_host": "Host_4",
                "cpu_before": 95.1,
                "cpu_after": 74.8
            }
        ])
        st.dataframe(sample_log, use_container_width=True)

        st.warning("⚠️ Run the Decision Engine (Step 4) to generate actual migration logs.")

    # Decision Engine info
    st.markdown("### ⚙️ Decision Engine Configuration")
    st.info("""
    **Current Settings:**
    - **Strategy:** Cost-Aware (optimizes SLA violations + load balance + migration cost)
    - **Overload Threshold:** 85% CPU
    - **Migration Cost Penalty:** 5.0 units per migration
    - **SLA Violation Weight:** 10.0
    - **Load Balance Weight:** 2.0
    
    The cost-aware strategy selects the VM that minimizes total cost after migration.
    """)


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    main()


# Legacy function for backward compatibility
def run_dashboard():
    """Legacy entry point."""
    main()
