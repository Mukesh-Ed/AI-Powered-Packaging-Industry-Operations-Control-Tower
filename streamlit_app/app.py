# ============================================================
# PACKAGING AI OPERATIONS CONTROL TOWER
# Streamlit Application
# ============================================================

from pathlib import Path
import os
import time
import json

import pandas as pd
import numpy as np
import streamlit as st

# Plotly is optional
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Packaging AI Operations Control Tower",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

# Also support lowercase data folder if it exists
if not PROCESSED_DIR.exists():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# FILE PATHS
# ============================================================

FEATURES_FILE = PROCESSED_DIR / "features.csv"
ANOMALY_FILE = PROCESSED_DIR / "anomaly_results.csv"
RISK_FILE = PROCESSED_DIR / "risk_results.csv"
MACHINE_RISK_FILE = PROCESSED_DIR / "machine_risk_summary.csv"

# Fallback names
HOURLY_FILE = PROCESSED_DIR / "hourly_cleaned.csv"


# ============================================================
# CUSTOM CSS
# IMPORTANT:
# CSS only. No HTML UI content is used.
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: linear-gradient(
            135deg,
            #061826 0%,
            #071a2d 45%,
            #10133b 100%
        );
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        font-weight: 800 !important;
    }

    .title-main {
        text-align: center;
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 5px;
    }

    .subtitle-main {
        text-align: center;
        font-size: 15px;
        margin-bottom: 25px;
        opacity: 0.85;
    }

    .status-box {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(0, 220, 255, 0.5);
        background: rgba(15, 30, 55, 0.75);
        text-align: center;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 850;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background: rgba(20, 35, 65, 0.85);
        border: 1px solid rgba(0, 200, 255, 0.45);
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 900;
    }

    .healthy {
        padding: 10px;
        border-radius: 10px;
        background: rgba(0, 180, 100, 0.18);
        border: 1px solid rgba(0, 220, 120, 0.5);
    }

    .warning {
        padding: 10px;
        border-radius: 10px;
        background: rgba(255, 165, 0, 0.18);
        border: 1px solid rgba(255, 165, 0, 0.5);
    }

    .critical {
        padding: 10px;
        border-radius: 10px;
        background: rgba(255, 60, 80, 0.18);
        border: 1px solid rgba(255, 60, 80, 0.5);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_file(*paths):
    """
    Return the first existing file from the supplied paths.
    """
    for path in paths:
        if path.exists():
            return path
    return None


def safe_read_csv(path):
    """
    Safely read CSV.
    """
    if path is None or not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df
    except Exception as e:
        st.error(f"Unable to read file: {path.name}")
        st.caption(str(e))
        return pd.DataFrame()


def convert_datetime(df):
    """
    Convert interval_start if available.
    """
    if df.empty:
        return df

    if "interval_start" in df.columns:
        df["interval_start"] = pd.to_datetime(
            df["interval_start"],
            errors="coerce"
        )

    return df


def get_machine_column(df):
    """
    Find machine/equipment column.
    """
    possible = [
        "equipment_ID",
        "equipment_id",
        "machine_id",
        "machine"
    ]

    for col in possible:
        if col in df.columns:
            return col

    return None


def get_machine_list(df):
    """
    Return unique machines.
    """
    machine_col = get_machine_column(df)

    if machine_col is None or df.empty:
        return []

    values = (
        df[machine_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


def normalize_machine_column(df):
    """
    Create standard equipment_ID column.
    """
    if df.empty:
        return df

    if "equipment_ID" not in df.columns:

        if "equipment_id" in df.columns:
            df["equipment_ID"] = df["equipment_id"]

        elif "machine_id" in df.columns:
            df["equipment_ID"] = df["machine_id"]

    return df


def get_numeric_value(row, column, default=0.0):
    """
    Safely get numeric value.
    """
    if column not in row.index:
        return default

    try:
        value = pd.to_numeric(row[column], errors="coerce")

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def percent_value(value):
    """
    Convert 0-1 proportions into percentage.
    """
    try:
        value = float(value)

        if value <= 1:
            return value * 100

        return value

    except Exception:
        return 0.0


def calculate_health(row):
    """
    Project-defined health score.

    Important:
    Production is NOT subtracted.
    Downtime, idle and performance loss are
    proportions between 0 and 1.
    """

    downtime = get_numeric_value(row, "%downtime")
    idle = get_numeric_value(row, "%idle")
    performance_loss = get_numeric_value(
        row,
        "%performance_loss"
    )

    health = 100 - (
        downtime +
        idle +
        performance_loss
    )

    return max(0.0, min(100.0, health))


def get_status(health):
    if health >= 90:
        return "HEALTHY"

    if health >= 70:
        return "WARNING"

    return "CRITICAL"


def latest_machine_row(df, machine):
    """
    Return latest row for selected machine.
    """
    if df.empty:
        return pd.Series(dtype=object)

    machine_col = get_machine_column(df)

    if machine_col is None:
        return pd.Series(dtype=object)

    temp = df[
        df[machine_col].astype(str) == str(machine)
    ].copy()

    if temp.empty:
        return pd.Series(dtype=object)

    if "interval_start" in temp.columns:
        temp["interval_start"] = pd.to_datetime(
            temp["interval_start"],
            errors="coerce"
        )

        temp = temp.sort_values(
            "interval_start"
        )

    return temp.iloc[-1]


def machine_filter(df, machine):
    """
    Filter selected machine.
    """
    if df.empty:
        return df

    machine_col = get_machine_column(df)

    if machine_col is None:
        return df

    return df[
        df[machine_col].astype(str) == str(machine)
    ].copy()


def calculate_machine_summary(df, machine):
    """
    Build summary for selected machine.
    """

    temp = machine_filter(df, machine)

    if temp.empty:
        return {}

    production = (
        pd.to_numeric(
            temp.get("%production", 0),
            errors="coerce"
        )
        .mean()
    )

    downtime = (
        pd.to_numeric(
            temp.get("%downtime", 0),
            errors="coerce"
        )
        .mean()
    )

    idle = (
        pd.to_numeric(
            temp.get("%idle", 0),
            errors="coerce"
        )
        .mean()
    )

    performance_loss = (
        pd.to_numeric(
            temp.get("%performance_loss", 0),
            errors="coerce"
        )
        .mean()
    )

    health = 100 - (
        downtime +
        idle +
        performance_loss
    )

    health = max(
        0,
        min(100, health)
    )

    return {
        "records": len(temp),
        "production": percent_value(production),
        "downtime": percent_value(downtime),
        "idle": percent_value(idle),
        "performance_loss": percent_value(
            performance_loss
        ),
        "health": health
    }


def get_anomaly_summary(anomaly_df, machine):
    """
    Get anomaly information.
    """

    temp = machine_filter(
        anomaly_df,
        machine
    )

    if temp.empty:
        return {
            "count": 0,
            "status": "NO ANOMALY DATA"
        }

    if "is_anomaly" in temp.columns:

        anomaly_values = (
            temp["is_anomaly"]
            .astype(str)
            .str.lower()
        )

        count = (
            anomaly_values.isin(
                ["true", "1", "yes"]
            )
        ).sum()

    elif "anomaly_prediction" in temp.columns:

        count = (
            pd.to_numeric(
                temp["anomaly_prediction"],
                errors="coerce"
            ) == -1
        ).sum()

    else:
        count = 0

    return {
        "count": int(count),
        "status": (
            "ANOMALY DETECTED"
            if count > 0
            else "NORMAL"
        )
    }


def get_risk_summary(risk_df, machine):
    """
    Get risk information.
    """

    temp = machine_filter(
        risk_df,
        machine
    )

    if temp.empty:
        return {
            "count": 0,
            "level": "NO DATA",
            "status": "NO RISK DATA"
        }

    count = 0

    if "risk_prediction" in temp.columns:

        count = (
            pd.to_numeric(
                temp["risk_prediction"],
                errors="coerce"
            ) == 1
        ).sum()

    elif "future_risk" in temp.columns:

        count = (
            pd.to_numeric(
                temp["future_risk"],
                errors="coerce"
            ) == 1
        ).sum()

    level = "LOW"

    if "risk_level" in temp.columns:

        levels = (
            temp["risk_level"]
            .dropna()
            .astype(str)
            .str.upper()
        )

        if not levels.empty:

            priority = {
                "CRITICAL": 4,
                "HIGH": 3,
                "MEDIUM": 2,
                "LOW": 1
            }

            level = max(
                levels,
                key=lambda x: priority.get(x, 0)
            )

    return {
        "count": int(count),
        "level": level,
        "status": (
            "RISK DETECTED"
            if count > 0
            else "LOW RISK"
        )
    }


def build_investigation_answer(
    machine,
    summary,
    anomaly,
    risk
):
    """
    Local fallback investigation response.

    This does NOT invent alarm meanings.
    """

    production = summary.get(
        "production",
        0
    )

    downtime = summary.get(
        "downtime",
        0
    )

    idle = summary.get(
        "idle",
        0
    )

    performance_loss = summary.get(
        "performance_loss",
        0
    )

    health = summary.get(
        "health",
        0
    )

    answer = []

    answer.append(
        f"### Incident Summary\n"
        f"Machine **{machine}** is being investigated "
        f"using operational measurements and model outputs."
    )

    answer.append(
        "### Evidence\n"
        f"- Average production: **{production:.2f}%**\n"
        f"- Average downtime: **{downtime:.2f}%**\n"
        f"- Average idle: **{idle:.2f}%**\n"
        f"- Average performance loss: "
        f"**{performance_loss:.2f}%**\n"
        f"- Machine health: **{health:.2f}**\n"
        f"- Anomalous records: **{anomaly['count']}**\n"
        f"- Risk predictions: **{risk['count']}**\n"
        f"- Latest risk level: **{risk['level']}**"
    )

    answer.append(
        "### Possible Root Cause\n"
        "The available data indicates an operational "
        "condition requiring investigation. Possible "
        "contributors include increased downtime, "
        "idle periods, performance loss, repeated alarms, "
        "or abnormal machine behaviour.\n\n"
        "**No specific alarm meaning or physical failure "
        "is assumed without supporting evidence.**"
    )

    confidence = "MEDIUM"

    if anomaly["count"] > 0 and risk["count"] > 0:
        confidence = "MEDIUM-HIGH"

    answer.append(
        f"### Confidence\n**{confidence}**"
    )

    answer.append(
        "### Recommended Investigation\n"
        "1. Review the machine timeline around high-downtime events.\n"
        "2. Check repeated alarm occurrences and their duration.\n"
        "3. Compare production before and after abnormal events.\n"
        "4. Review anomaly-detection results.\n"
        "5. Review predictive-risk results.\n"
        "6. Check whether performance loss increased before the event.\n"
        "7. Use the project knowledge base for further investigation.\n"
        "8. Confirm findings with the maintenance/operations team."
    )

    answer.append(
        "### Human Verification\n"
        "AI output is an investigation aid. Physical "
        "machine inspection, maintenance or intervention "
        "must be verified and approved by a responsible "
        "human operator."
    )

    return "\n\n".join(answer)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_all_data():

    features_path = find_file(
        FEATURES_FILE,
        PROCESSED_DIR / "features.csv"
    )

    anomaly_path = find_file(
        ANOMALY_FILE,
        PROCESSED_DIR / "anomaly_results.csv"
    )

    risk_path = find_file(
        RISK_FILE,
        PROCESSED_DIR / "risk_results.csv"
    )

    hourly_path = find_file(
        HOURLY_FILE
    )

    features = safe_read_csv(
        features_path
    )

    anomaly = safe_read_csv(
        anomaly_path
    )

    risk = safe_read_csv(
        risk_path
    )

    hourly = safe_read_csv(
        hourly_path
    )

    features = normalize_machine_column(
        features
    )

    anomaly = normalize_machine_column(
        anomaly
    )

    risk = normalize_machine_column(
        risk
    )

    hourly = normalize_machine_column(
        hourly
    )

    features = convert_datetime(
        features
    )

    anomaly = convert_datetime(
        anomaly
    )

    risk = convert_datetime(
        risk
    )

    hourly = convert_datetime(
        hourly
    )

    return (
        features,
        anomaly,
        risk,
        hourly
    )


features_df, anomaly_df, risk_df, hourly_df = (
    load_all_data()
)


# ============================================================
# SELECT MAIN OPERATIONAL DATASET
# ============================================================

if not features_df.empty:
    operational_df = features_df.copy()

elif not hourly_df.empty:
    operational_df = hourly_df.copy()

else:
    operational_df = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏭 AI CONTROL TOWER")

    st.divider()

    st.subheader("🎛️ CONTROL PANEL")

    machines = get_machine_list(
        operational_df
    )

    if not machines:

        st.error(
            "No machine data found."
        )

        st.info(
            "Check Data/processed/features.csv"
        )

        st.stop()

    selected_machine = st.selectbox(
        "Select Machine",
        machines,
        index=0
    )

    st.divider()

    # ========================================================
    # REAL-TIME SIMULATION
    # ========================================================

    st.subheader(
        "⚡ REAL-TIME SIMULATION"
    )

    simulation_enabled = st.checkbox(
        "Enable Live Simulation",
        value=False
    )

    simulation_interval = st.slider(
        "Simulation Interval",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )

    if simulation_enabled:

        st.success(
            "🟢 SIMULATION ACTIVE"
        )

    else:

        st.info(
            "⏸️ SIMULATION PAUSED"
        )

    st.divider()

    # ========================================================
    # AI MODULES
    # ========================================================

    st.subheader("🧠 AI MODULES")

    anomaly_enabled = st.checkbox(
        "🚨 Anomaly Detection",
        value=True
    )

    risk_enabled = st.checkbox(
        "⚠️ Risk Prediction",
        value=True
    )

    rag_enabled = st.checkbox(
        "📚 RAG Knowledge",
        value=True
    )

    agent_enabled = st.checkbox(
        "🤖 AI Agent",
        value=True
    )

    root_cause_enabled = st.checkbox(
        "🧠 Root Cause Analysis",
        value=True
    )

    st.divider()

    st.caption(
        "Packaging AI Operations Control Tower"
    )


# ============================================================
# SESSION STATE FOR SIMULATION
# ============================================================

machine_data = machine_filter(
    operational_df,
    selected_machine
)

if "simulation_index" not in st.session_state:

    st.session_state.simulation_index = (
        max(len(machine_data) - 1, 0)
    )


if (
    st.session_state.get("last_machine")
    != selected_machine
):

    st.session_state.simulation_index = (
        max(len(machine_data) - 1, 0)
    )

    st.session_state.last_machine = (
        selected_machine
    )


# ============================================================
# REAL-TIME SIMULATION LOGIC
# ============================================================

if simulation_enabled and not machine_data.empty:

    current_index = (
        st.session_state.simulation_index
    )

    current_index = min(
        current_index,
        len(machine_data) - 1
    )

    simulation_row = machine_data.iloc[
        current_index
    ]

else:

    if not machine_data.empty:
        simulation_row = machine_data.iloc[-1]
    else:
        simulation_row = pd.Series(dtype=object)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title-main">🏭 Packaging AI Operations Control Tower</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-main">'
    'Intelligent Machine Monitoring • '
    'Anomaly Detection • Predictive Risk • '
    'AI Root-Cause Investigation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    '<div class="status-box">',
    unsafe_allow_html=True
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.success("🟢 SYSTEM ONLINE")

with status_col2:
    st.info("⚡ DATA SERVICES READY")

with status_col3:
    st.success("🧠 AI CONTROL TOWER ACTIVE")

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# CURRENT MACHINE
# ============================================================

st.info(
    f"🏭 Currently Monitoring: **{selected_machine}**"
)

st.caption(
    "The control tower combines operational data, "
    "machine-learning models, RAG knowledge and "
    "AI reasoning to investigate machine problems."
)


# ============================================================
# MACHINE SUMMARY
# ============================================================

summary = calculate_machine_summary(
    operational_df,
    selected_machine
)

if not summary:

    st.warning(
        "No operational data available for this machine."
    )

    st.stop()


# ============================================================
# CURRENT ROW VALUES
# ============================================================

current_production = get_numeric_value(
    simulation_row,
    "%production"
)

current_downtime = get_numeric_value(
    simulation_row,
    "%downtime"
)

current_idle = get_numeric_value(
    simulation_row,
    "%idle"
)

current_performance_loss = get_numeric_value(
    simulation_row,
    "%performance_loss"
)

current_health = calculate_health(
    simulation_row
)


# ============================================================
# MACHINE PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 MACHINE PERFORMANCE</div>',
    unsafe_allow_html=True
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        "📈 PRODUCTION",
        f"{percent_value(current_production):.2f}%"
    )

with kpi2:
    st.metric(
        "⏱️ DOWNTIME",
        f"{percent_value(current_downtime):.2f}%"
    )

with kpi3:
    st.metric(
        "💤 IDLE",
        f"{percent_value(current_idle):.2f}%"
    )

with kpi4:
    st.metric(
        "📉 PERFORMANCE LOSS",
        f"{percent_value(current_performance_loss):.2f}%"
    )

with kpi5:
    st.metric(
        "❤️ MACHINE HEALTH",
        f"{current_health:.2f}"
    )


# ============================================================
# MACHINE STATUS
# ============================================================

machine_status = get_status(
    current_health
)

if machine_status == "HEALTHY":

    st.success(
        f"🟢 Machine Status: {machine_status}"
    )

elif machine_status == "WARNING":

    st.warning(
        f"🟠 Machine Status: {machine_status}"
    )

else:

    st.error(
        f"🔴 Machine Status: {machine_status}"
    )


# ============================================================
# ALERT DATA
# ============================================================

anomaly_summary = get_anomaly_summary(
    anomaly_df,
    selected_machine
)

risk_summary = get_risk_summary(
    risk_df,
    selected_machine
)


# ============================================================
# INTELLIGENT ALERT CENTER
# ============================================================

st.markdown(
    '<div class="section-title">🚨 INTELLIGENT ALERT CENTER</div>',
    unsafe_allow_html=True
)

alert1, alert2 = st.columns(2)

with alert1:

    if anomaly_enabled:

        if anomaly_summary["count"] > 0:

            st.error(
                f"""
                🚨 ANOMALY DETECTED

                Machine: {selected_machine}

                Anomalous Records: {anomaly_summary["count"]}

                Status: ANOMALY DETECTED
                """
            )

        else:

            st.success(
                f"""
                ✅ NO ANOMALY DETECTED

                Machine: {selected_machine}
                """
            )

    else:

        st.info(
            "Anomaly Detection module disabled."
        )


with alert2:

    if risk_enabled:

        if risk_summary["count"] > 0:

            if risk_summary["level"] in [
                "CRITICAL",
                "HIGH"
            ]:

                st.error(
                    f"""
                    ⚠️ PREDICTIVE RISK DETECTED

                    Machine: {selected_machine}

                    Risk Predictions: {risk_summary["count"]}

                    Latest Risk Level: {risk_summary["level"]}
                    """
                )

            else:

                st.warning(
                    f"""
                    ⚠️ PREDICTIVE RISK DETECTED

                    Machine: {selected_machine}

                    Risk Predictions: {risk_summary["count"]}

                    Latest Risk Level: {risk_summary["level"]}
                    """
                )

        else:

            st.success(
                "✅ No predictive risk detected."
            )

    else:

        st.info(
            "Risk Prediction module disabled."
        )


# ============================================================
# LIVE MACHINE DATA
# ============================================================

st.markdown(
    '<div class="section-title">🔴 LIVE MACHINE DATA</div>',
    unsafe_allow_html=True
)

live_columns = [
    col for col in [
        "equipment_ID",
        "interval_start",
        "%production",
        "%downtime",
        "%idle",
        "%performance_loss",
        "%scheduled_downtime",
        "#changes",
        "health_score"
    ]
    if col in machine_data.columns
]

if live_columns:

    display_live = machine_data[
        live_columns
    ].tail(10).copy()

    st.dataframe(
        display_live,
        use_container_width=True,
        hide_index=True
    )

else:

    st.dataframe(
        machine_data.tail(10),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PERFORMANCE TREND
# ============================================================

st.markdown(
    '<div class="section-title">📈 PERFORMANCE TREND</div>',
    unsafe_allow_html=True
)

trend_columns = [
    col for col in [
        "%production",
        "%downtime",
        "%idle",
        "%performance_loss"
    ]
    if col in machine_data.columns
]

if (
    PLOTLY_AVAILABLE
    and "interval_start" in machine_data.columns
    and trend_columns
):

    trend_df = machine_data[
        ["interval_start"] + trend_columns
    ].copy()

    trend_df = trend_df.tail(200)

    for col in trend_columns:
        trend_df[col] = pd.to_numeric(
            trend_df[col],
            errors="coerce"
        ) * 100

    fig = px.line(
        trend_df,
        x="interval_start",
        y=trend_columns,
        labels={
            "value": "Percentage",
            "interval_start": "Time",
            "variable": "Metric"
        }
    )

    fig.update_layout(
        height=430,
        legend_title="Metrics"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif trend_columns:

    st.line_chart(
        machine_data[
            trend_columns
        ].tail(100) * 100
    )

else:

    st.info(
        "Performance trend data unavailable."
    )


# ============================================================
# ANOMALY ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 ANOMALY ANALYSIS</div>',
    unsafe_allow_html=True
)

if anomaly_enabled:

    machine_anomaly = machine_filter(
        anomaly_df,
        selected_machine
    )

    if not machine_anomaly.empty:

        ac1, ac2, ac3 = st.columns(3)

        anomaly_count = anomaly_summary[
            "count"
        ]

        total_anomaly_records = len(
            machine_anomaly
        )

        anomaly_rate = (
            anomaly_count /
            total_anomaly_records *
            100
            if total_anomaly_records > 0
            else 0
        )

        with ac1:
            st.metric(
                "Anomalous Records",
                f"{anomaly_count:,}"
            )

        with ac2:
            st.metric(
                "Total Records",
                f"{total_anomaly_records:,}"
            )

        with ac3:
            st.metric(
                "Anomaly Rate",
                f"{anomaly_rate:.2f}%"
            )

        anomaly_display_cols = [
            col for col in [
                "equipment_ID",
                "interval_start",
                "anomaly_prediction",
                "anomaly_score",
                "is_anomaly"
            ]
            if col in machine_anomaly.columns
        ]

        if anomaly_display_cols:

            st.dataframe(
                machine_anomaly[
                    anomaly_display_cols
                ].tail(20),
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "No anomaly data available."
        )

else:

    st.info(
        "Anomaly Detection module is disabled."
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ PREDICTIVE RISK ANALYSIS</div>',
    unsafe_allow_html=True
)

if risk_enabled:

    machine_risk = machine_filter(
        risk_df,
        selected_machine
    )

    if not machine_risk.empty:

        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.metric(
                "Risk Predictions",
                f"{risk_summary['count']:,}"
            )

        with rc2:
            st.metric(
                "Latest Risk Level",
                risk_summary["level"]
            )

        with rc3:

            if "risk_probability" in machine_risk.columns:

                probability = pd.to_numeric(
                    machine_risk[
                        "risk_probability"
                    ],
                    errors="coerce"
                ).max()

                st.metric(
                    "Maximum Risk Probability",
                    f"{probability:.2f}%"
                )

            else:

                st.metric(
                    "Risk Records",
                    f"{len(machine_risk):,}"
                )

        risk_display_cols = [
            col for col in [
                "equipment_ID",
                "interval_start",
                "future_risk",
                "risk_prediction",
                "risk_probability",
                "risk_level"
            ]
            if col in machine_risk.columns
        ]

        if risk_display_cols:

            st.dataframe(
                machine_risk[
                    risk_display_cols
                ].tail(20),
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "No risk data available."
        )

else:

    st.info(
        "Risk Prediction module is disabled."
    )


# ============================================================
# RAG KNOWLEDGE
# ============================================================

st.markdown(
    '<div class="section-title">📚 RAG KNOWLEDGE BASE</div>',
    unsafe_allow_html=True
)

if rag_enabled:

    documents_dir = (
        PROJECT_ROOT / "documents"
    )

    if documents_dir.exists():

        txt_files = list(
            documents_dir.rglob("*.txt")
        )

        if txt_files:

            st.success(
                f"📚 Knowledge Base Ready — "
                f"{len(txt_files)} documents available"
            )

            for file in txt_files:

                st.write(
                    f"• {file.relative_to(PROJECT_ROOT)}"
                )

        else:

            st.warning(
                "Knowledge base folder exists, "
                "but no TXT documents were found."
            )

    else:

        st.warning(
            "documents folder not found."
        )

else:

    st.info(
        "RAG Knowledge module is disabled."
    )


# ============================================================
# AI ROOT CAUSE INVESTIGATION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 AI ROOT CAUSE INVESTIGATION</div>',
    unsafe_allow_html=True
)

if root_cause_enabled:

    st.subheader(
        "🤖 AI Operations Assistant"
    )

    st.write(
        "Investigate machine anomalies, operational "
        "risks and possible root causes using the "
        "available project evidence."
    )

    default_question = (
        f"Why is machine {selected_machine} "
        f"experiencing abnormal operational behaviour? "
        f"What should be investigated?"
    )

    investigation_question = st.text_area(
        "Investigation Question",
        value=default_question,
        height=110
    )

    if st.button(
        "🚀 RUN AI INVESTIGATION",
        type="primary",
        use_container_width=False
    ):

        with st.spinner(
            "AI Operations Assistant is investigating..."
        ):

            answer = build_investigation_answer(
                selected_machine,
                summary,
                anomaly_summary,
                risk_summary
            )

            st.session_state[
                "investigation_answer"
            ] = answer

        st.success(
            "✅ AI investigation completed."
        )

    if (
        "investigation_answer"
        in st.session_state
    ):

        st.markdown(
            st.session_state[
                "investigation_answer"
            ]
        )

else:

    st.info(
        "Root Cause Analysis module is disabled."
    )


# ============================================================
# SUPPORTING EVIDENCE
# ============================================================

st.markdown("---")

st.markdown(
    """
    <h2 style="margin-bottom:5px;">
        🔎 Supporting Evidence
    </h2>
    <p style="color:#9ca3af;">
        Operational, anomaly and risk information used by the
        AI Operations Control Tower.
    </p>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

# ==========================================
# MACHINE DATA
# ==========================================

st.subheader("🏭 Machine Data")

# Use the already filtered operational dataframe for the selected machine.
machine_df = machine_data.copy()

if machine_df.empty:
    st.warning(f"No machine data available for {selected_machine}")
else:

    machine_summary = {
        "Machine": selected_machine,
        "Records": int(machine_df.shape[0]),
    }

    st.dataframe(
        machine_df,
        use_container_width=True
    )

# ------------------------------------------------------------
# MACHINE EVIDENCE
# ------------------------------------------------------------

with col1:

    st.markdown("### 🏭 Machine Data")

    machine_evidence = {
        "Equipment ID": selected_machine,
        "Records": int(machine_df.shape[0]),
        "Average Production": f"{machine_df['%production'].mean() * 100:.2f}%",
        "Average Downtime": f"{machine_df['%downtime'].mean() * 100:.2f}%",
        "Average Idle": f"{machine_df['%idle'].mean() * 100:.2f}%",
        "Average Performance Loss":
            f"{machine_df['%performance_loss'].mean() * 100:.2f}%",
        "Average Health Score":
            f"{machine_df['health_score'].mean():.2f}"
    }

    for key, value in machine_evidence.items():

        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:space-between;
                padding:8px 4px;
                border-bottom:1px solid rgba(255,255,255,0.08);
            ">
                <span style="color:#9ca3af;">{key}</span>
                <strong>{value}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------------------------------------------
# ANOMALY EVIDENCE
# ------------------------------------------------------------

with col2:

    st.markdown("### 🚨 Anomaly Data")

    anomaly_count = int(
        anomaly_df["is_anomaly"].sum()
    )

    if anomaly_count > 0:
        anomaly_status = "ANOMALY DETECTED"
    else:
        anomaly_status = "NORMAL"

    anomaly_evidence = {
        "Equipment ID": selected_machine,
        "Anomaly Count": anomaly_count,
        "Status": anomaly_status
    }

    for key, value in anomaly_evidence.items():

        if key == "Status":

            if anomaly_count > 0:
                value_html = (
                    '<span style="color:#ff4b4b;font-weight:800;">'
                    f'{value}'
                    '</span>'
                )
            else:
                value_html = (
                    '<span style="color:#00d084;font-weight:800;">'
                    f'{value}'
                    '</span>'
                )

        else:
            value_html = f"<strong>{value}</strong>"

        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:space-between;
                padding:8px 4px;
                border-bottom:1px solid rgba(255,255,255,0.08);
            ">
                <span style="color:#9ca3af;">{key}</span>
                {value_html}
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------------------------------------------
# RISK EVIDENCE
# ------------------------------------------------------------

with col3:

    st.markdown("### ⚠️ Risk Data")

    risk_count = int(
        risk_df["risk_prediction"].sum()
    )

    if len(risk_df) > 0:

        latest_risk_level = str(
            risk_df.iloc[-1]["risk_level"]
        ).upper()

    else:

        latest_risk_level = "LOW"

    if risk_count > 0:
        risk_status = "RISK DETECTED"
    else:
        risk_status = "NO RISK"

    risk_evidence = {
        "Equipment ID": selected_machine,
        "Risk Predictions": risk_count,
        "Risk Status": risk_status,
        "Latest Risk Level": latest_risk_level
    }

    for key, value in risk_evidence.items():

        if key == "Latest Risk Level":

            if value == "CRITICAL":
                value_html = (
                    '<span style="color:#ff4b4b;font-weight:800;">'
                    f'{value}'
                    '</span>'
                )

            elif value == "HIGH":
                value_html = (
                    '<span style="color:#ff9800;font-weight:800;">'
                    f'{value}'
                    '</span>'
                )

            elif value == "MEDIUM":
                value_html = (
                    '<span style="color:#ffd166;font-weight:800;">'
                    f'{value}'
                    '</span>'
                )

            else:
                value_html = (
                    '<span style="color:#00d084;font-weight:800;">'
                    f'{value}'
                    '</span>'
                )

        else:

            value_html = f"<strong>{value}</strong>"

        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:space-between;
                padding:8px 4px;
                border-bottom:1px solid rgba(255,255,255,0.08);
            ">
                <span style="color:#9ca3af;">{key}</span>
                {value_html}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# AI AGENT STATUS
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI AGENT STATUS</div>',
    unsafe_allow_html=True
)

agent1, agent2, agent3, agent4 = st.columns(4)

with agent1:

    if agent_enabled:
        st.success("🟢 Agent Ready")
    else:
        st.info("Agent Disabled")

with agent2:

    if rag_enabled:
        st.success("📚 RAG Ready")
    else:
        st.info("RAG Disabled")

with agent3:

    if anomaly_enabled:
        st.success("🚨 Anomaly Model Ready")
    else:
        st.info("Anomaly Disabled")

with agent4:

    if risk_enabled:
        st.success("⚠️ Risk Model Ready")
    else:
        st.info("Risk Disabled")


# ============================================================
# REAL-TIME SIMULATION CONTROLLER
# ============================================================

if simulation_enabled and not machine_data.empty:

    st.markdown(
        '<div class="section-title">⚡ REAL-TIME SIMULATION</div>',
        unsafe_allow_html=True
    )

    sim1, sim2, sim3 = st.columns(3)

    with sim1:

        st.metric(
            "Simulation Row",
            f"{st.session_state.simulation_index + 1:,}"
        )

    with sim2:

        st.metric(
            "Available Records",
            f"{len(machine_data):,}"
        )

    with sim3:

        if "interval_start" in simulation_row.index:

            st.metric(
                "Simulation Time",
                str(
                    simulation_row[
                        "interval_start"
                    ]
                )
            )

    if st.button(
        "▶️ NEXT SIMULATION STEP",
        use_container_width=False
    ):

        next_index = (
            st.session_state.simulation_index + 1
        )

        if next_index >= len(machine_data):

            next_index = 0

        st.session_state.simulation_index = (
            next_index
        )

        st.rerun()

    st.info(
        "Simulation uses historical machine records "
        "as a real-time operational stream."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer1, footer2, footer3 = st.columns(3)

with footer1:
    st.caption(
        "🏭 Packaging AI Operations Control Tower"
    )

with footer2:
    st.caption(
        "Isolation Forest • XGBoost • RAG • Gemini"
    )

with footer3:
    st.caption(
        "FastAPI • Streamlit • Real-Time Simulation"
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if simulation_enabled:

    time.sleep(
        simulation_interval
    )

    next_index = (
        st.session_state.simulation_index + 1
    )

    if not machine_data.empty:

        if next_index >= len(machine_data):
            next_index = 0

        st.session_state.simulation_index = (
            next_index
        )

    st.rerun()
