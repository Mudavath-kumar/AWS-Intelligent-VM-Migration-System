"""
streamlit_app.py - Interactive Streamlit Dashboard
=====================================================
Replaces static matplotlib plots with a live, interactive dashboard.

Features:
  - Live host monitoring (CPU, RAM, power)
  - Migration event log
  - Overload alerts
  - Strategy comparison charts
  - Energy model visualization

Run:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def run_dashboard():
    """Main dashboard entry point."""
    if not STREAMLIT_AVAILABLE:
        print("[DASHBOARD] Streamlit/Plotly not installed.")
        print("  Install with: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="VM Migration Dashboard",
        page_icon="🖥️",
        layout="wide",
    )

    st.title("🖥️ Intelligent VM Migration Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("⚙️ Controls")
    page = st.sidebar.radio("Navigate", [
        "📊 Host Monitoring",
        "📈 CPU/RAM Trends",
        "🔀 Migration Log",
        "⚡ Energy Analysis",
        "🏆 Strategy Comparison",
        "🔥 Overload Alerts",
    ])

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "vm_metrics.csv")

    if not os.path.exists(csv_path):
        st.error("No simulation data found. Run the simulation first (Step 1 in main.py).")
        st.stop()

    df = pd.read_csv(csv_path)

    if page == "📊 Host Monitoring":
        _page_host_monitoring(df)
    elif page == "📈 CPU/RAM Trends":
        _page_cpu_ram_trends(df)
    elif page == "🔀 Migration Log":
        _page_migration_log()
    elif page == "⚡ Energy Analysis":
        _page_energy_analysis(df)
    elif page == "🏆 Strategy Comparison":
        _page_strategy_comparison()
    elif page == "🔥 Overload Alerts":
        _page_overload_alerts(df)


def _page_host_monitoring(df):
    """Live host monitoring page."""
    st.header("📊 Host Monitoring")

    latest_tick = df["tick"].max()
    selected_tick = st.slider("Select Tick", int(df["tick"].min()), int(latest_tick), int(latest_tick))

    tick_data = df[df["tick"] == selected_tick]
    host_agg = tick_data.groupby("host_id").agg(
        avg_cpu=("cpu", "mean"),
        avg_ram=("ram", "mean"),
        avg_network=("network", "mean"),
        num_vms=("vm_id", "count"),
        host_cpu=("total_host_cpu", "first"),
    ).reset_index()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total VMs", int(host_agg["num_vms"].sum()))
    with col2:
        st.metric("Avg CPU", f"{host_agg['avg_cpu'].mean():.1f}%")
    with col3:
        overloaded = int((host_agg["host_cpu"] > 85).sum())
        st.metric("Overloaded Hosts", overloaded, delta_color="inverse")

    # Bar chart
    fig = px.bar(
        host_agg, x="host_id", y=["avg_cpu", "avg_ram", "avg_network"],
        barmode="group", title=f"Host Metrics at Tick {selected_tick}",
        labels={"value": "Utilization (%)", "host_id": "Host"},
        color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71"],
    )
    fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Overload Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # Detailed VM table
    st.subheader("VM Details")
    st.dataframe(tick_data[["vm_id", "host_id", "cpu", "ram", "network"]].sort_values("cpu", ascending=False))


def _page_cpu_ram_trends(df):
    """CPU/RAM trends over time."""
    st.header("📈 CPU & RAM Trends")

    host_agg = df.groupby(["tick", "host_id"]).agg(
        avg_cpu=("cpu", "mean"),
        avg_ram=("ram", "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig_cpu = px.line(
            host_agg, x="tick", y="avg_cpu", color="host_id",
            title="Average CPU per Host",
            labels={"avg_cpu": "CPU (%)", "tick": "Tick"},
        )
        fig_cpu.add_hline(y=85, line_dash="dash", line_color="red")
        st.plotly_chart(fig_cpu, use_container_width=True)

    with col2:
        fig_ram = px.line(
            host_agg, x="tick", y="avg_ram", color="host_id",
            title="Average RAM per Host",
            labels={"avg_ram": "RAM (%)", "tick": "Tick"},
        )
        st.plotly_chart(fig_ram, use_container_width=True)

    # Heatmap
    st.subheader("Host CPU Heatmap")
    pivot = host_agg.pivot(index="host_id", columns="tick", values="avg_cpu")
    fig_heat = px.imshow(
        pivot, color_continuous_scale="YlOrRd",
        labels={"color": "CPU %"},
        title="Host CPU Utilization Over Time",
        aspect="auto",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


def _page_migration_log():
    """Display migration events."""
    st.header("🔀 Migration Log")

    try:
        from simulation.migration import get_migration_log
        log = get_migration_log()
    except Exception:
        log = []

    if not log:
        st.info("No migration events recorded. Run the Decision Engine first (Step 4).")
        return

    log_df = pd.DataFrame(log)
    st.dataframe(log_df, use_container_width=True)

    # Migration frequency chart
    if "source_host" in log_df.columns:
        src_counts = log_df["source_host"].value_counts().reset_index()
        src_counts.columns = ["host", "migrations_from"]
        fig = px.bar(src_counts, x="host", y="migrations_from",
                     title="Migrations FROM Each Host",
                     color="migrations_from", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)


def _page_energy_analysis(df):
    """Energy consumption analysis with dynamic power model."""
    st.header("⚡ Energy Analysis")

    st.markdown("""
    **Dynamic Power Model:**
    $$P = P_{idle} + (P_{max} - P_{idle}) \\times utilization$$
    """)

    col1, col2 = st.columns(2)
    with col1:
        p_idle = st.number_input("P_idle (W)", value=100, step=10)
    with col2:
        p_max = st.number_input("P_max (W)", value=300, step=10)

    host_agg = df.groupby(["tick", "host_id"]).agg(
        host_cpu=("total_host_cpu", "first"),
    ).reset_index()

    host_agg["utilization"] = host_agg["host_cpu"] / 100.0
    host_agg["power"] = p_idle + (p_max - p_idle) * host_agg["utilization"]

    fig = px.line(
        host_agg, x="tick", y="power", color="host_id",
        title="Dynamic Power Consumption Over Time",
        labels={"power": "Power (W)", "tick": "Tick"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Total energy
    total = host_agg.groupby("tick")["power"].sum().reset_index()
    fig2 = px.area(total, x="tick", y="power",
                   title="Total Cluster Power Over Time",
                   labels={"power": "Total Power (W)"})
    st.plotly_chart(fig2, use_container_width=True)


def _page_strategy_comparison():
    """Strategy comparison charts."""
    st.header("🏆 Strategy Comparison")

    st.info("Run evaluation (Step 5) to generate comparison data.")

    # Show comparison image if exists
    comp_path = os.path.join(os.path.dirname(__file__), "..", "evaluation", "comparison.png")
    if os.path.exists(comp_path):
        st.image(comp_path, caption="Rule-Based vs ML-Based Comparison")

    # Show RF vs LSTM if exists
    rf_lstm_path = os.path.join(os.path.dirname(__file__), "..", "model", "rf_vs_lstm_comparison.png")
    if os.path.exists(rf_lstm_path):
        st.image(rf_lstm_path, caption="Random Forest vs LSTM Accuracy")

    # Show RL training if exists
    rl_path = os.path.join(os.path.dirname(__file__), "..", "results", "rl_training_rewards.png")
    if os.path.exists(rl_path):
        st.image(rl_path, caption="RL Agent Training Rewards")


def _page_overload_alerts(df):
    """Overload detection and alerts."""
    st.header("🔥 Overload Alerts")

    threshold = st.slider("Overload Threshold (%)", 50, 100, 85)

    overloaded = df[df["total_host_cpu"] > threshold][["tick", "host_id", "total_host_cpu"]].drop_duplicates()

    if overloaded.empty:
        st.success(f"No overload events detected (threshold: {threshold}%)")
    else:
        st.warning(f"Found {len(overloaded)} overload events!")
        st.dataframe(overloaded.sort_values(["tick", "host_id"]), use_container_width=True)

        # Timeline
        fig = px.scatter(
            overloaded, x="tick", y="total_host_cpu", color="host_id",
            title="Overload Events Timeline",
            labels={"total_host_cpu": "Host CPU (%)", "tick": "Tick"},
            size="total_host_cpu",
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    run_dashboard()
