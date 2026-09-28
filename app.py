import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="B-LIFE | Bearing Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GENERAL ---------- */

    .stApp {
        background: #0b1220;
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* ---------- HEADER ---------- */

    .main-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 2px;
        color: #f8fafc;
    }

    .subtitle {
        font-size: 15px;
        color: #94a3b8;
        margin-bottom: 25px;
    }

    .system-status {
        text-align: right;
        color: #22c55e;
        font-size: 14px;
        font-weight: 600;
        padding-top: 10px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: #111827;
        border: 1px solid #243044;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
    }

    .card-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .card-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 800;
    }

    .card-small {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }

    /* ---------- RUL CARD ---------- */

    .rul-card {
        background: linear-gradient(135deg, #172033, #111827);
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        min-height: 210px;
    }

    .rul-label {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .rul-value {
        color: #f8fafc;
        font-size: 52px;
        font-weight: 850;
        line-height: 1.1;
        margin: 12px 0;
    }

    .rul-unit {
        color: #94a3b8;
        font-size: 14px;
    }

    /* ---------- HEALTH ---------- */

    .health-box {
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        min-height: 210px;
    }

    .health-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #cbd5e1;
    }

    .health-status {
        font-size: 32px;
        font-weight: 850;
        margin: 15px 0;
    }

    .health-action {
        font-size: 14px;
        color: #cbd5e1;
        line-height: 1.5;
    }

    .healthy {
        background: rgba(22, 163, 74, 0.13);
        border: 1px solid rgba(34,197,94,0.45);
    }

    .warning {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234,179,8,0.45);
    }

    .critical {
        background: rgba(220, 38, 38, 0.13);
        border: 1px solid rgba(248,113,113,0.45);
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #f8fafc;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .section-description {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #243044;
        padding: 15px;
        border-radius: 12px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding-top: 25px;
        border-top: 1px solid #1e293b;
        margin-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD AI OUTPUT
# ============================================================

DATA_FILE = "../02_AI_Output/AI_Bearing_Prediction_Final.csv"

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"Could not find {DATA_FILE}. "
        "Make sure the AI handover folder structure is preserved."
    )
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values(
    ["Bearing_ID", "Timestamp"]
).reset_index(drop=True)

bearing_ids = sorted(df["Bearing_ID"].unique())


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="main-title">⚙️ B-LIFE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Based Bearing Health & Remaining Useful Life Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="system-status">● AI ENGINE ONLINE</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ B-LIFE")

    st.caption("Bearing Lifecycle Intelligence")

    st.markdown("---")

    st.markdown("### Bearing Selection")

    selected_bearing = st.selectbox(
        "Select Bearing",
        bearing_ids
    )

    bearing_df = df[df["Bearing_ID"] == selected_bearing].copy()

    if len(bearing_df) == 0:
        st.warning("No data available for this bearing.")
        st.stop()

    sample_index = st.slider(
        "Operating Sample",
        min_value=0,
        max_value=len(bearing_df) - 1,
        value=0
    )

    selected_row = bearing_df.iloc[sample_index]

    st.markdown("---")

    st.markdown("### Dashboard Mode")

    mode = st.radio(
        "View",
        ["Operator View", "Engineer View"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption(
        "AI results are generated from the NASA IMS bearing dataset "
        "using the trained ML model."
    )


# ============================================================
# SELECTED RESULT
# ============================================================

predicted_rul = float(selected_row["Predicted_RUL_hours"])
actual_rul = float(selected_row["Actual_RUL_hours"])

health_status = str(selected_row["Health_Status"])
maintenance_action = str(selected_row["Recommended_Action"])

timestamp = selected_row["Timestamp"]


# ============================================================
# HEALTH STYLE
# ============================================================

if health_status == "Healthy":
    health_class = "healthy"
    health_icon = "🟢"

elif health_status == "Warning":
    health_class = "warning"
    health_icon = "🟡"

else:
    health_class = "critical"
    health_icon = "🔴"

# --------------------------------------------------------
# AI SYSTEM OVERVIEW
# --------------------------------------------------------

st.markdown("### AI System Overview")

overview1, overview2, overview3, overview4 = st.columns(4)

with overview1:
    st.metric(
        label="AI Engine",
        value="ONLINE",
        delta="ML Model Active"
    )

with overview2:
    st.metric(
        label="Dataset",
        value="NASA IMS",
        delta="Run-to-Failure"
    )

with overview3:
    st.metric(
        label="ML Model",
        value="GBR",
        delta="RUL Prediction"
    )

with overview4:
    st.metric(
        label="Input Features",
        value="6",
        delta="Vibration Features"
    )

st.divider()

# ============================================================
# HEALTH OVERVIEW
# ============================================================

health_file = "../02_AI_Output/AI_Bearing_Prediction_Final.csv"
health_data = pd.read_csv(health_file)

healthy_count = (health_data["Health_Status"] == "Healthy").sum()
warning_count = (health_data["Health_Status"] == "Warning").sum()
critical_count = (health_data["Health_Status"] == "Critical").sum()

total_evaluated = len(health_data)

st.markdown("### Health Overview")

st.caption(
    f"Health classification across {total_evaluated} evaluated test samples"
)

health1, health2, health3 = st.columns(3)

with health1:
    st.metric(
        label="🟢 Healthy",
        value=healthy_count,
        delta="Continue monitoring"
    )

with health2:
    st.metric(
        label="🟡 Warning",
        value=warning_count,
        delta="Schedule inspection"
    )

with health3:
    st.metric(
        label="🔴 Critical",
        value=critical_count,
        delta="Plan replacement"
    )

st.divider()


# ============================================================
# OPERATOR VIEW
# ============================================================

if mode == "Operator View":

    st.markdown(
        '<div class="section-title">Current Bearing Condition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Simple condition assessment for maintenance and operating personnel.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MAIN STATUS CARDS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)
    with col1:

        rul_html = (
            '<div class="rul-card">'
            '<div class="rul-label">Estimated Remaining Useful Life</div>'
            f'<div class="rul-value">{predicted_rul:,.1f}</div>'
            '<div class="rul-unit">hours</div>'
            '</div>'
        )

        st.markdown(rul_html, unsafe_allow_html=True)

   

    with col2:

        health_html = (
            f'<div class="health-box {health_class}">'
            '<div class="health-label">Current Health Condition</div>'
            f'<div class="health-status">{health_icon} {health_status.upper()}</div>'
            '<div class="health-action">'
            '<strong>Recommended Action</strong><br>'
            f'{maintenance_action}'
            '</div>'
            '</div>'
        )

        st.markdown(health_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # SENSOR PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Vibration Condition Indicators</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "RMS",
            f"{float(selected_row['RMS']):.5f}"
        )

    with m2:
        st.metric(
            "Peak",
            f"{float(selected_row['Peak']):.5f}"
        )

    with m3:
        st.metric(
            "Kurtosis",
            f"{float(selected_row['Kurtosis']):.4f}"
        )

    m4, m5, m6 = st.columns(3)

    with m4:
        st.metric(
            "Standard Deviation",
            f"{float(selected_row['Std_Dev']):.5f}"
        )

    with m5:
        st.metric(
            "Skewness",
            f"{float(selected_row['Skewness']):.4f}"
        )

    with m6:
        st.metric(
            "Crest Factor",
            f"{float(selected_row['Crest_Factor']):.4f}"
        )

    # --------------------------------------------------------
    # RUL TREND
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Remaining Useful Life Trend</div>',
        unsafe_allow_html=True
    )

    trend_df = bearing_df.copy()

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(
        range(len(trend_df)),
        trend_df["Actual_RUL_hours"],
        label="Actual RUL",
        linewidth=2
    )

    ax.plot(
        range(len(trend_df)),
        trend_df["Predicted_RUL_hours"],
        label="Predicted RUL",
        linewidth=2
    )

    ax.axvline(
        sample_index,
        linestyle="--",
        linewidth=1.5,
        label="Selected Sample"
    )

    ax.set_xlabel("Operating Sample")
    ax.set_ylabel("RUL (hours)")
    ax.set_title(f"Bearing {selected_bearing} — RUL Trend")
    ax.grid(alpha=0.2)
    ax.legend()

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    st.caption(
        f"Selected operating timestamp: "
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )


# ============================================================
# ENGINEER VIEW
# ============================================================

else:

    st.markdown(
        '<div class="section-title">Engineer View</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Detailed AI prediction, vibration features and model information.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.metric(
            "Bearing ID",
            str(selected_bearing)
        )

    with e2:
        st.metric(
            "Predicted RUL",
            f"{predicted_rul:.2f} h"
        )

    with e3:
        st.metric(
            "Actual RUL",
            f"{actual_rul:.2f} h"
        )

    with e4:
        st.metric(
            "Health",
            health_status
        )

    # --------------------------------------------------------
    # FEATURE TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Input Features</div>',
        unsafe_allow_html=True
    )

    feature_data = pd.DataFrame({
        "Feature": [
            "RMS",
            "Standard Deviation",
            "Kurtosis",
            "Skewness",
            "Peak",
            "Crest Factor"
        ],
        "Value": [
            selected_row["RMS"],
            selected_row["Std_Dev"],
            selected_row["Kurtosis"],
            selected_row["Skewness"],
            selected_row["Peak"],
            selected_row["Crest_Factor"]
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RUL TREND
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Bearing RUL Analysis</div>',
        unsafe_allow_html=True
    )

    trend_df = bearing_df.copy()

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(
        range(len(trend_df)),
        trend_df["Actual_RUL_hours"],
        label="Actual RUL",
        linewidth=2
    )

    ax.plot(
        range(len(trend_df)),
        trend_df["Predicted_RUL_hours"],
        label="Predicted RUL",
        linewidth=2
    )

    ax.axvline(
        sample_index,
        linestyle="--",
        linewidth=1.5,
        label="Selected Sample"
    )

    ax.set_xlabel("Operating Sample")
    ax.set_ylabel("RUL (hours)")
    ax.set_title(f"Bearing {selected_bearing} — Actual vs Predicted RUL")
    ax.grid(alpha=0.2)
    ax.legend()

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # SELECTED SAMPLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Selected Prediction</div>',
        unsafe_allow_html=True
    )

    selected_display = selected_row.to_frame().T

    st.dataframe(
        selected_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MAINTENANCE DECISION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Engineering Recommendation</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Health Status: {health_status}\n\n"
        f"Predicted RUL: {predicted_rul:.2f} hours\n\n"
        f"Recommended Action: {maintenance_action}"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        B-LIFE | AI-Based Bearing Health & Remaining Useful Life Intelligence
        <br>
        Mechanical Engineering Project • AI / Predictive Maintenance / PLM
    </div>
    """,
    unsafe_allow_html=True
)