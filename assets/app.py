from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
ASSETS_DIR = ROOT_DIR / "assets"
LOGS_DIR = ROOT_DIR / "logs"

DATASET_PATH = DATA_DIR / "aquaponic_synthetic_dataset_v31_bogor_realistic_balanced.csv"
TRAIN_PATH = DATA_DIR / "aquaponic_train_v31.csv"
TEST_PATH = DATA_DIR / "aquaponic_test_v31.csv"
WATER_MODEL_PATH = MODEL_DIR / "water_quality_classifier_v31.joblib"
DOSING_MODEL_PATH = MODEL_DIR / "dosing_action_classifier_v31.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns_v31.json"
FINAL_REPORT_PATH = REPORTS_DIR / "final_model_report_v31_realistic_balanced.md"
SIMULATION_REPORT_PATH = REPORTS_DIR / "simulation_decision_v31.md"
LOG_PATH = LOGS_DIR / "dashboard_decision_log.csv"
AUTOMATION_LOG_PATH = LOGS_DIR / "automation_cycle_log.csv"

DEFAULT_FEATURES = [
    "ph",
    "temperature_c",
    "water_level_pct",
    "ammonia_ppm",
    "nitrite_ppm",
    "nitrate_ppm",
    "sensor_status",
    "time_since_last_dosing_min",
    "dosing_cycle_count",
    "confidence_score",
]

LOG_COLUMNS = [
    "timestamp",
    "scenario_name",
    "ph",
    "temperature_c",
    "water_level_pct",
    "ammonia_ppm",
    "nitrite_ppm",
    "nitrate_ppm",
    "sensor_status",
    "time_since_last_dosing_min",
    "dosing_cycle_count",
    "confidence_score",
    "predicted_water_quality_status",
    "predicted_dosing_action",
    "safety_status",
    "safety_reason",
    "pump_status",
    "dashboard_alert",
]

AUTOMATION_LOG_COLUMNS = [
    "timestamp",
    "cycle",
    "scenario_name",
    "data_packet_id",
    "last_packet_timestamp",
    "ph_before",
    "ph_after_simulated",
    "delta_ph",
    "dosing_success",
    "temperature_c",
    "water_level_pct",
    "ammonia_ppm",
    "nitrite_ppm",
    "nitrate_ppm",
    "sensor_status",
    "confidence_score",
    "predicted_water_quality_status",
    "predicted_dosing_action",
    "safety_status",
    "safety_reason",
    "pump_status",
    "virtual_acid_pump",
    "virtual_base_pump",
    "automation_status",
    "dashboard_alert",
]

PRESET_SCENARIOS: dict[str, dict[str, Any]] = {
    "normal_condition": {
        "ph": 6.8,
        "temperature_c": 27.0,
        "water_level_pct": 85.0,
        "ammonia_ppm": 0.10,
        "nitrite_ppm": 0.05,
        "nitrate_ppm": 60.0,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.90,
    },
    "high_ph_condition": {
        "ph": 7.8,
        "temperature_c": 28.0,
        "water_level_pct": 85.0,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 80.0,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.85,
    },
    "low_ph_condition": {
        "ph": 6.0,
        "temperature_c": 27.0,
        "water_level_pct": 82.0,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 70.0,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.85,
    },
    "low_water_level_condition": {
        "ph": 7.7,
        "temperature_c": 28.0,
        "water_level_pct": 40.0,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 80.0,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.85,
    },
    "sensor_error_condition": {
        "ph": 7.8,
        "temperature_c": 28.0,
        "water_level_pct": 85.0,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 80.0,
        "sensor_status": "error",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.60,
    },
}

PERFORMANCE_INFO = pd.DataFrame(
    [
        {
            "target": "water_quality_status",
            "model": "RandomForestClassifier(max_depth=3)",
            "accuracy": 0.5540,
            "f1_macro": 0.5705,
        },
        {
            "target": "dosing_action",
            "model": "DecisionTreeClassifier(max_depth=3)",
            "accuracy": 0.6850,
            "f1_macro": 0.5505,
        },
    ]
)


def configure_page() -> None:
    st.set_page_config(
        page_title="Aquaponic IoT-ML-MBSE Dashboard",
        page_icon="AQ",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --ipb-blue: #0b2f5f;
            --ipb-light-blue: #d8ecff;
            --ipb-gray: #f3f6f9;
            --ipb-green: #1b7f4c;
            --ipb-yellow: #b98700;
            --ipb-orange: #c45b16;
            --ipb-red: #b42318;
        }
        .main .block-container { padding-top: 1.4rem; }
        h1, h2, h3 { color: var(--ipb-blue); }
        .hero {
            background: linear-gradient(135deg, #0b2f5f 0%, #174f8f 100%);
            color: white;
            padding: 1.2rem 1.4rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .hero h1 { color: white; margin-bottom: .3rem; }
        .info-box {
            background: #f6fbff;
            border-left: 5px solid #1f77b4;
            padding: .85rem 1rem;
            border-radius: 6px;
            margin: .65rem 0;
        }
        .warn-box {
            background: #fff8e6;
            border-left: 5px solid #b98700;
            padding: .85rem 1rem;
            border-radius: 6px;
            margin: .65rem 0;
        }
        .status-card {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 1rem;
            background: white;
            min-height: 115px;
        }
        .status-card .label {
            color: #52606d;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .02em;
        }
        .status-card .value {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: .25rem;
        }
        .green { color: var(--ipb-green); }
        .yellow { color: var(--ipb-yellow); }
        .orange { color: var(--ipb-orange); }
        .red { color: var(--ipb-red); }
        .blue { color: var(--ipb-blue); }
        .footer {
            margin-top: 2rem;
            padding-top: .8rem;
            border-top: 1px solid #d9e2ec;
            color: #52606d;
            font-size: .85rem;
            text-align: center;
        }
        .pump-panel {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
            min-height: 110px;
        }
        .pump-title {
            color: #52606d;
            font-size: .8rem;
            text-transform: uppercase;
        }
        .pump-state {
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: .25rem;
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 rgba(31, 119, 180, .0); transform: translateY(0); }
            50% { box-shadow: 0 0 20px rgba(31, 119, 180, .35); transform: translateY(-1px); }
            100% { box-shadow: 0 0 0 rgba(31, 119, 180, .0); transform: translateY(0); }
        }
        @keyframes pipeFlow {
            from { background-position: 0 0; }
            to { background-position: 34px 0; }
        }
        .schematic-wrap,
        .live-schematic {
            display: flex;
            align-items: stretch;
            gap: .45rem;
            flex-wrap: wrap;
            margin: .4rem 0 1rem 0;
        }
        .schematic-step {
            flex: 1 1 120px;
            min-width: 116px;
            border-radius: 8px;
            border: 1px solid #d9e2ec;
            padding: .7rem .65rem;
            background: white;
            min-height: 86px;
        }
        .schematic-kicker {
            font-size: .68rem;
            color: #52606d;
            text-transform: uppercase;
            letter-spacing: .03em;
        }
        .schematic-title {
            font-size: .88rem;
            color: #102a43;
            font-weight: 750;
            line-height: 1.15rem;
            margin-top: .22rem;
        }
        .schematic-status {
            font-size: .76rem;
            color: #52606d;
            margin-top: .3rem;
        }
        .schematic-arrow {
            align-self: center;
            color: #9fb3c8;
            font-weight: 800;
        }
        .flow-blue { border-color: #8fc5ff; background: #f2f8ff; }
        .flow-green { border-color: #8ed6ad; background: #f2fbf6; }
        .flow-gray { border-color: #d9e2ec; background: #f7f9fb; }
        .flow-red { border-color: #f0a5a0; background: #fff4f2; }
        .animated-pump-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .85rem;
            margin: .4rem 0 1rem 0;
        }
        .animated-pump {
            border-radius: 8px;
            border: 1px solid #d9e2ec;
            padding: 1rem;
            min-height: 142px;
            position: relative;
            overflow: hidden;
        }
        .pump-active {
            border-color: #4fb3d9;
            background: linear-gradient(135deg, #eaf8ff 0%, #edfdf3 100%);
            animation: pulseGlow 1.6s ease-in-out infinite;
        }
        .pump-off {
            background: #f7f9fb;
            border-color: #d9e2ec;
        }
        .pump-blocked {
            background: #fff4f2;
            border-color: #f0a5a0;
        }
        .animated-pump-title {
            color: #52606d;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .03em;
        }
        .animated-pump-state {
            font-size: 1.55rem;
            font-weight: 850;
            margin-top: .35rem;
            color: #0b2f5f;
        }
        .animated-pump-meta {
            color: #52606d;
            font-size: .85rem;
            margin-top: .25rem;
        }
        .pump-flow-symbol {
            font-size: 1.4rem;
            letter-spacing: .2rem;
            margin-top: .45rem;
            color: #1f77b4;
            font-weight: 800;
        }
        .dosing-flow {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
            padding: .95rem;
            margin-bottom: 1rem;
        }
        .flow-row {
            display: grid;
            grid-template-columns: minmax(100px, 1fr) minmax(50px, .55fr) minmax(110px, 1fr) minmax(50px, .55fr) minmax(130px, 1fr);
            gap: .6rem;
            align-items: center;
            margin: .55rem 0;
        }
        .flow-node {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #f7f9fb;
            padding: .65rem;
            text-align: center;
            font-weight: 750;
            color: #102a43;
        }
        .flow-node.active {
            border-color: #8ed6ad;
            background: #f2fbf6;
            color: #1b7f4c;
        }
        .flow-node.blocked {
            border-color: #f0a5a0;
            background: #fff4f2;
            color: #b42318;
        }
        .flow-line {
            height: 10px;
            border-radius: 999px;
            background: #cbd5df;
            position: relative;
        }
        .flow-line.active-acid {
            background: repeating-linear-gradient(90deg, #1f77b4 0, #1f77b4 13px, #a8def5 13px, #a8def5 24px);
            animation: pipeFlow .8s linear infinite;
        }
        .flow-line.active-base {
            background: repeating-linear-gradient(90deg, #1b7f4c 0, #1b7f4c 13px, #b8e5cc 13px, #b8e5cc 24px);
            animation: pipeFlow .8s linear infinite;
        }
        .flow-line.blocked {
            background: repeating-linear-gradient(90deg, #b42318 0, #b42318 10px, #f0a5a0 10px, #f0a5a0 20px);
        }
        .ph-meter {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .ph-meter-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .65rem;
            margin-bottom: .85rem;
        }
        .ph-stat {
            background: #f7f9fb;
            border-radius: 8px;
            padding: .65rem;
            border: 1px solid #d9e2ec;
        }
        .ph-stat-label {
            font-size: .72rem;
            color: #52606d;
            text-transform: uppercase;
        }
        .ph-stat-value {
            font-size: 1.1rem;
            font-weight: 800;
            color: #102a43;
        }
        .ph-track {
            position: relative;
            height: 18px;
            border-radius: 999px;
            background: linear-gradient(90deg, #b42318 0%, #c45b16 20%, #b98700 38%, #1b7f4c 48%, #1b7f4c 55%, #b98700 68%, #c45b16 82%, #b42318 100%);
            overflow: hidden;
        }
        .ph-marker {
            position: absolute;
            top: -5px;
            width: 4px;
            height: 28px;
            border-radius: 3px;
            background: #0b2f5f;
            box-shadow: 0 0 0 2px #ffffff;
        }
        .ph-target-band {
            position: absolute;
            top: 0;
            height: 18px;
            background: rgba(255, 255, 255, .38);
            border-left: 2px solid #ffffff;
            border-right: 2px solid #ffffff;
        }
        .ph-meter-note {
            color: #52606d;
            font-size: .86rem;
            margin-top: .45rem;
        }
        .automation-timeline {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: .5rem;
            margin: .4rem 0 1rem 0;
        }
        .timeline-step {
            border-radius: 8px;
            border: 1px solid #d9e2ec;
            padding: .65rem .5rem;
            background: #f7f9fb;
            min-height: 76px;
        }
        .timeline-number {
            font-size: .7rem;
            color: #52606d;
            font-weight: 750;
        }
        .timeline-title {
            font-size: .85rem;
            color: #102a43;
            font-weight: 800;
            line-height: 1.1rem;
            margin-top: .2rem;
        }
        .timeline-pass { background: #f2fbf6; border-color: #8ed6ad; }
        .timeline-fail { background: #fff4f2; border-color: #f0a5a0; }
        .timeline-neutral { background: #f2f8ff; border-color: #8fc5ff; }
        .timeline-off { background: #f7f9fb; border-color: #d9e2ec; }
        @media (max-width: 900px) {
            .animated-pump-grid,
            .ph-meter-grid,
            .automation-timeline {
                grid-template-columns: 1fr;
            }
            .flow-row {
                grid-template-columns: 1fr;
            }
            .schematic-arrow { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Any, Any, dict[str, Any]]:
    """Load v3.1 ML models and feature metadata."""
    water_model = joblib.load(WATER_MODEL_PATH)
    dosing_model = joblib.load(DOSING_MODEL_PATH)
    feature_metadata = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    return water_model, dosing_model, feature_metadata


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_split_counts() -> tuple[int, int]:
    train_rows = len(pd.read_csv(TRAIN_PATH)) if TRAIN_PATH.exists() else 0
    test_rows = len(pd.read_csv(TEST_PATH)) if TEST_PATH.exists() else 0
    return train_rows, test_rows


def ensure_log_file() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


def ensure_automation_log_file() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    if not AUTOMATION_LOG_PATH.exists():
        pd.DataFrame(columns=AUTOMATION_LOG_COLUMNS).to_csv(AUTOMATION_LOG_PATH, index=False)


def apply_safety_rule(sensor_input: dict[str, Any], predicted_dosing_action: str) -> dict[str, str]:
    """Safety Rule Controller for final pump decision."""
    checks = [
        (sensor_input["sensor_status"] != "valid", "sensor_not_valid"),
        (float(sensor_input["water_level_pct"]) < 50, "water_level_low"),
        (float(sensor_input["confidence_score"]) < 0.70, "low_confidence"),
        (float(sensor_input["time_since_last_dosing_min"]) < 10, "cooldown_not_met"),
        (int(sensor_input["dosing_cycle_count"]) >= 3, "max_cycle_exceeded"),
        (predicted_dosing_action == "manual_check", "manual_check_required"),
        (float(sensor_input["ph"]) < 5.8, "ph_too_low"),
        (float(sensor_input["ph"]) > 8.2, "ph_too_high"),
        (float(sensor_input["ammonia_ppm"]) > 1.0, "ammonia_high"),
        (float(sensor_input["nitrite_ppm"]) > 1.0, "nitrite_high"),
    ]
    for failed, reason in checks:
        if failed:
            return {
                "safety_status": "safety_fail",
                "safety_reason": reason,
                "pump_status": "blocked",
            }

    if predicted_dosing_action == "no_action":
        return {
            "safety_status": "safety_pass",
            "safety_reason": "no_action_required",
            "pump_status": "off",
        }

    return {
        "safety_status": "safety_pass",
        "safety_reason": "safety_pass_all_checks",
        "pump_status": "on",
    }


def generate_dashboard_alert(
    predicted_dosing_action: str,
    safety_status: str,
    safety_reason: str,
) -> str:
    """Create operator-facing dashboard alert."""
    safety_alerts = {
        "sensor_not_valid": "Sensor bermasalah, pompa diblokir",
        "water_level_low": "Level air rendah, pompa diblokir",
        "low_confidence": "Confidence rendah, perlu manual check",
        "cooldown_not_met": "Cooldown belum terpenuhi, pompa diblokir",
        "max_cycle_exceeded": "Siklus dosing maksimum tercapai",
        "manual_check_required": "Manual check diperlukan",
        "ph_too_low": "pH ekstrem, perlu pemeriksaan manual",
        "ph_too_high": "pH ekstrem, perlu pemeriksaan manual",
        "ammonia_high": "Nitrogen tinggi, perlu pemeriksaan manual",
        "nitrite_high": "Nitrogen tinggi, perlu pemeriksaan manual",
    }
    if safety_status == "safety_fail":
        return safety_alerts.get(safety_reason, "Safety rule gagal, pompa diblokir")

    action_alerts = {
        "no_action": "Kondisi normal/tidak memerlukan dosing, pompa off",
        "acid_low_dose": "pH tinggi ringan, pompa acid low dose aktif",
        "acid_medium_dose": "pH tinggi sedang, pompa acid medium dose aktif",
        "base_low_dose": "pH rendah ringan, pompa base low dose aktif",
        "base_medium_dose": "pH rendah sedang, pompa base medium dose aktif",
    }
    return action_alerts.get(predicted_dosing_action, "Manual check diperlukan")


def simulate_case(sensor_input: dict[str, Any], scenario_name: str = "manual_input") -> dict[str, Any]:
    """Run sensor input through ML predictions and Safety Rule Controller."""
    water_model, dosing_model, feature_metadata = load_models()
    feature_columns = feature_metadata.get("main_features", DEFAULT_FEATURES)
    input_df = pd.DataFrame([sensor_input])[feature_columns]

    predicted_water_quality_status = str(water_model.predict(input_df)[0])
    predicted_dosing_action = str(dosing_model.predict(input_df)[0])
    safety_result = apply_safety_rule(sensor_input, predicted_dosing_action)
    dashboard_alert = generate_dashboard_alert(
        predicted_dosing_action,
        safety_result["safety_status"],
        safety_result["safety_reason"],
    )

    return {
        "scenario_name": scenario_name,
        **sensor_input,
        "predicted_water_quality_status": predicted_water_quality_status,
        "predicted_dosing_action": predicted_dosing_action,
        "safety_status": safety_result["safety_status"],
        "safety_reason": safety_result["safety_reason"],
        "pump_status": safety_result["pump_status"],
        "dashboard_alert": dashboard_alert,
    }


def append_log(result: dict[str, Any]) -> None:
    ensure_log_file()
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **result}
    log_df = pd.read_csv(LOG_PATH)
    updated_log = pd.concat([log_df, pd.DataFrame([row])[LOG_COLUMNS]], ignore_index=True)
    updated_log.to_csv(LOG_PATH, index=False)


def append_automation_log(result: dict[str, Any]) -> None:
    ensure_automation_log_file()
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **result}
    log_df = pd.read_csv(AUTOMATION_LOG_PATH)
    new_row_df = pd.DataFrame([row])
    updated_log = pd.concat([log_df, new_row_df], ignore_index=True)
    for column in AUTOMATION_LOG_COLUMNS:
        if column not in updated_log.columns:
            updated_log[column] = ""
    updated_log[AUTOMATION_LOG_COLUMNS].to_csv(AUTOMATION_LOG_PATH, index=False)


def get_current_result() -> dict[str, Any] | None:
    return st.session_state.get("latest_result")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def init_automation_state(scenario_name: str) -> None:
    st.session_state["current_virtual_sensor_state"] = PRESET_SCENARIOS[scenario_name].copy()
    st.session_state["automation_history"] = []
    st.session_state["latest_automation_result"] = None
    st.session_state["automation_cycle_count"] = 0
    st.session_state["automation_running"] = False


def ensure_automation_session_state(default_scenario: str) -> None:
    if "automation_history" not in st.session_state:
        init_automation_state(default_scenario)
    if "current_virtual_sensor_state" not in st.session_state:
        st.session_state["current_virtual_sensor_state"] = PRESET_SCENARIOS[default_scenario].copy()
    if "automation_cycle_count" not in st.session_state:
        st.session_state["automation_cycle_count"] = 0
    if "automation_running" not in st.session_state:
        st.session_state["automation_running"] = False


def evolve_virtual_sensor_state(state: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    """Create the next virtual sensor reading with small realistic drift."""
    next_state = state.copy()
    next_state["ph"] = round(clamp(float(next_state["ph"]) + random.uniform(-0.035, 0.035), 4.5, 9.5), 3)
    next_state["temperature_c"] = round(
        clamp(float(next_state["temperature_c"]) + random.uniform(-0.12, 0.12), 15.0, 40.0),
        2,
    )
    next_state["water_level_pct"] = round(
        clamp(float(next_state["water_level_pct"]) + random.uniform(-0.45, 0.25), 0.0, 100.0),
        2,
    )
    next_state["ammonia_ppm"] = round(
        clamp(float(next_state["ammonia_ppm"]) + random.uniform(-0.008, 0.010), 0.0, 2.5),
        3,
    )
    next_state["nitrite_ppm"] = round(
        clamp(float(next_state["nitrite_ppm"]) + random.uniform(-0.006, 0.008), 0.0, 2.5),
        3,
    )
    next_state["nitrate_ppm"] = round(
        clamp(float(next_state["nitrate_ppm"]) + random.uniform(-0.8, 1.0), 0.0, 260.0),
        2,
    )
    next_state["confidence_score"] = round(
        clamp(float(next_state["confidence_score"]) + random.uniform(-0.025, 0.025), 0.0, 1.0),
        3,
    )
    next_state["time_since_last_dosing_min"] = int(next_state["time_since_last_dosing_min"]) + 2
    next_state["dosing_cycle_count"] = int(next_state["dosing_cycle_count"])
    next_state["sensor_status"] = "error" if scenario_name == "sensor_error_condition" else "valid"
    return next_state


def virtual_actuator_state(result: dict[str, Any]) -> tuple[str, str, str]:
    action = result["predicted_dosing_action"]
    pump_status = result["pump_status"]
    if pump_status == "blocked":
        return "BLOCKED / OFF", "BLOCKED / OFF", "Pump Blocked"
    if pump_status == "off":
        return "OFF", "OFF", "Monitoring Only"
    if pump_status == "on" and "acid" in action:
        return "ON", "OFF", "Dosing Active"
    if pump_status == "on" and "base" in action:
        return "OFF", "ON", "Dosing Active"
    return "OFF", "OFF", "Monitoring Only"


def virtual_relay_state(pump_state: str) -> str:
    if pump_state == "ON":
        return "ON"
    if "BLOCKED" in pump_state:
        return "BLOCKED / OFF"
    return "OFF"


def simulate_recheck_ph(ph_before: float, action: str, pump_status: str) -> tuple[float, float, bool]:
    """Simulate pH recheck after virtual dosing or monitoring drift."""
    target_ph = 6.9
    before_distance = abs(ph_before - target_ph)
    if pump_status == "on" and "acid" in action:
        change = -min(max(ph_before - target_ph, 0.03) * 0.35 + random.uniform(0.015, 0.055), 0.22)
    elif pump_status == "on" and "base" in action:
        change = min(max(target_ph - ph_before, 0.03) * 0.35 + random.uniform(0.015, 0.055), 0.22)
    else:
        change = random.uniform(-0.018, 0.018)

    ph_after = round(clamp(ph_before + change, 4.5, 9.5), 3)
    delta_ph = round(ph_after - ph_before, 3)
    after_distance = abs(ph_after - target_ph)
    dosing_success = after_distance < before_distance if pump_status == "on" else abs(delta_ph) <= 0.03
    return ph_after, delta_ph, bool(dosing_success)


def build_automation_cycle_result(
    scenario_name: str,
    cycle_number: int,
    sensor_state: dict[str, Any],
) -> dict[str, Any]:
    """Run one virtual automation cycle from virtual sensor to recheck pH."""
    ph_before = float(sensor_state["ph"])
    packet_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    packet_id = f"VHW-{cycle_number:04d}-{datetime.now().strftime('%H%M%S')}"
    ml_result = simulate_case(sensor_state, scenario_name)
    append_log(ml_result)
    acid_pump, base_pump, automation_status = virtual_actuator_state(ml_result)
    ph_after, delta_ph, dosing_success = simulate_recheck_ph(
        ph_before,
        ml_result["predicted_dosing_action"],
        ml_result["pump_status"],
    )
    cycle_result = {
        "cycle": cycle_number,
        "scenario_name": scenario_name,
        "data_packet_id": packet_id,
        "last_packet_timestamp": packet_timestamp,
        "ph_before": ph_before,
        "ph_after_simulated": ph_after,
        "delta_ph": delta_ph,
        "dosing_success": dosing_success,
        "temperature_c": sensor_state["temperature_c"],
        "water_level_pct": sensor_state["water_level_pct"],
        "ammonia_ppm": sensor_state["ammonia_ppm"],
        "nitrite_ppm": sensor_state["nitrite_ppm"],
        "nitrate_ppm": sensor_state["nitrate_ppm"],
        "sensor_status": sensor_state["sensor_status"],
        "confidence_score": sensor_state["confidence_score"],
        "predicted_water_quality_status": ml_result["predicted_water_quality_status"],
        "predicted_dosing_action": ml_result["predicted_dosing_action"],
        "safety_status": ml_result["safety_status"],
        "safety_reason": ml_result["safety_reason"],
        "pump_status": ml_result["pump_status"],
        "virtual_acid_pump": acid_pump,
        "virtual_base_pump": base_pump,
        "automation_status": automation_status,
        "dashboard_alert": ml_result["dashboard_alert"],
    }

    next_sensor_state = sensor_state.copy()
    next_sensor_state["ph"] = ph_after
    if ml_result["pump_status"] == "on":
        next_sensor_state["dosing_cycle_count"] = min(int(next_sensor_state["dosing_cycle_count"]) + 1, 10)
        next_sensor_state["time_since_last_dosing_min"] = 0
    st.session_state["current_virtual_sensor_state"] = next_sensor_state
    st.session_state["latest_automation_result"] = cycle_result
    st.session_state["latest_result"] = ml_result
    return cycle_result


def status_color(value: str) -> str:
    mapping = {
        "Normal": "green",
        "Waspada": "yellow",
        "Kritis": "orange",
        "Darurat": "red",
        "safety_pass": "green",
        "safety_fail": "red",
        "off": "blue",
        "on": "green",
        "blocked": "red",
    }
    return mapping.get(value, "blue")


def render_card(label: str, value: str) -> None:
    color = status_color(value)
    st.markdown(
        f"""
        <div class="status-card">
            <div class="label">{label}</div>
            <div class="value {color}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<div class="footer">TIN2511 - Analisis dan Desain Sistem Produksi Agro Industri | Fadlilah Akbar | 2026</div>',
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Prototype Dashboard Sistem Otomasi Aquaponic IoT-ML-MBSE</h1>
            <p>Proof of concept akademik untuk pemantauan, klasifikasi risiko, dan koreksi pH kualitas air.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="warn-box">
        Dashboard ini adalah proof of concept akademik TIN2511 ADSPA. Dataset yang digunakan masih
        synthetic-realistic v3.1 dan perlu validasi data aquaponic aktual sebelum implementasi nyata.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Ringkasan Sistem")
    st.write(
        "Sistem menerapkan ML-assisted automation. Model ML memprediksi status kualitas air dan rekomendasi dosing, "
        "tetapi keputusan akhir pompa tetap dikunci oleh Safety Rule Controller."
    )
    st.info(
        "Alur: Input Sensor -> Water Quality Classifier -> Dosing Action Classifier -> "
        "Safety Rule Controller -> Pump Decision -> Dashboard Alert & Log"
    )

    architecture_candidates = [
        ASSETS_DIR / "physical_architecture.png",
        ASSETS_DIR / "architecture.png",
        ASSETS_DIR / "arsitektur.png",
    ]
    architecture_path = next((path for path in architecture_candidates if path.exists()), None)
    if architecture_path:
        st.image(str(architecture_path), caption="Arsitektur fisik/sistem aquaponic", width="stretch")
    else:
        st.caption("Gambar arsitektur belum ditemukan di folder assets/. Dashboard tetap dapat dijalankan.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset", "v3.1 Bogor")
    with col2:
        st.metric("Train/Test", "8000 / 2000")
    with col3:
        st.metric("Mode", "ML-assisted")


def render_input_page() -> None:
    st.header("Input Sensor & Simulasi")
    st.write("Pilih preset skenario atau isi input sensor secara manual, lalu jalankan prediksi ML dan safety check.")

    selected_preset = st.selectbox("Preset scenario", list(PRESET_SCENARIOS.keys()), key="preset_select")
    preset = PRESET_SCENARIOS[selected_preset]

    with st.form("sensor_input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ph = st.number_input("ph", min_value=4.5, max_value=9.5, value=float(preset["ph"]), step=0.01)
            temperature_c = st.number_input(
                "temperature_c",
                min_value=15.0,
                max_value=40.0,
                value=float(preset["temperature_c"]),
                step=0.1,
            )
            water_level_pct = st.number_input(
                "water_level_pct",
                min_value=0.0,
                max_value=100.0,
                value=float(preset["water_level_pct"]),
                step=0.1,
            )
        with c2:
            ammonia_ppm = st.number_input(
                "ammonia_ppm",
                min_value=0.0,
                max_value=2.5,
                value=float(preset["ammonia_ppm"]),
                step=0.01,
            )
            nitrite_ppm = st.number_input(
                "nitrite_ppm",
                min_value=0.0,
                max_value=2.5,
                value=float(preset["nitrite_ppm"]),
                step=0.01,
            )
            nitrate_ppm = st.number_input(
                "nitrate_ppm",
                min_value=0.0,
                max_value=260.0,
                value=float(preset["nitrate_ppm"]),
                step=1.0,
            )
        with c3:
            sensor_status = st.selectbox(
                "sensor_status",
                ["valid", "perlu_kalibrasi", "error"],
                index=["valid", "perlu_kalibrasi", "error"].index(str(preset["sensor_status"])),
            )
            time_since_last_dosing_min = st.number_input(
                "time_since_last_dosing_min",
                min_value=0,
                max_value=300,
                value=int(preset["time_since_last_dosing_min"]),
                step=1,
            )
            dosing_cycle_count = st.number_input(
                "dosing_cycle_count",
                min_value=0,
                max_value=10,
                value=int(preset["dosing_cycle_count"]),
                step=1,
            )
            confidence_score = st.slider(
                "confidence_score",
                min_value=0.0,
                max_value=1.0,
                value=float(preset["confidence_score"]),
                step=0.01,
            )

        submitted = st.form_submit_button("Run ML Prediction & Safety Check", type="primary")

    sensor_input = {
        "ph": ph,
        "temperature_c": temperature_c,
        "water_level_pct": water_level_pct,
        "ammonia_ppm": ammonia_ppm,
        "nitrite_ppm": nitrite_ppm,
        "nitrate_ppm": nitrate_ppm,
        "sensor_status": sensor_status,
        "time_since_last_dosing_min": int(time_since_last_dosing_min),
        "dosing_cycle_count": int(dosing_cycle_count),
        "confidence_score": confidence_score,
    }

    if submitted:
        result = simulate_case(sensor_input, selected_preset)
        st.session_state["latest_input"] = sensor_input
        st.session_state["latest_result"] = result
        append_log(result)
        st.success("Prediksi ML dan safety check selesai. Hasil juga sudah ditambahkan ke log.")
        render_prediction_summary(result)


def render_prediction_summary(result: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_card("Water Quality", result["predicted_water_quality_status"])
    with c2:
        render_card("Dosing Action", result["predicted_dosing_action"])
    with c3:
        render_card("Safety Status", result["safety_status"])
    with c4:
        render_card("Pump Status", result["pump_status"])
    st.info(result["dashboard_alert"])


def render_prediction_page() -> None:
    st.header("Hasil Prediksi ML")
    result = get_current_result()
    if not result:
        st.warning("Belum ada hasil prediksi. Jalankan simulasi dari menu Input Sensor & Simulasi.")
        return

    render_prediction_summary(result)
    st.subheader("Input yang Digunakan Model")
    st.dataframe(pd.DataFrame([{key: result[key] for key in DEFAULT_FEATURES}]), width="stretch")
    st.markdown(
        """
        <div class="info-box">
        Catatan: output ML masih berupa rekomendasi. Status aktuator final ditentukan setelah Safety Rule Controller.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_safety_page() -> None:
    st.header("Safety Rule & Status Pompa")
    result = get_current_result()
    if not result:
        st.warning("Belum ada hasil safety check. Jalankan simulasi terlebih dahulu.")
        return

    render_prediction_summary(result)
    st.subheader("Rule yang Aktif")
    active_rule = pd.DataFrame(
        [
            {
                "safety_status": result["safety_status"],
                "safety_reason": result["safety_reason"],
                "pump_status": result["pump_status"],
                "dashboard_alert": result["dashboard_alert"],
            }
        ]
    )
    st.dataframe(active_rule, width="stretch")

    st.subheader("Daftar Safety Rule Controller")
    rules = pd.DataFrame(
        [
            ["sensor_status != valid", "sensor_not_valid", "blocked"],
            ["water_level_pct < 50", "water_level_low", "blocked"],
            ["confidence_score < 0.70", "low_confidence", "blocked"],
            ["time_since_last_dosing_min < 10", "cooldown_not_met", "blocked"],
            ["dosing_cycle_count >= 3", "max_cycle_exceeded", "blocked"],
            ["predicted_dosing_action == manual_check", "manual_check_required", "blocked"],
            ["ph < 5.8", "ph_too_low", "blocked"],
            ["ph > 8.2", "ph_too_high", "blocked"],
            ["ammonia_ppm > 1.0", "ammonia_high", "blocked"],
            ["nitrite_ppm > 1.0", "nitrite_high", "blocked"],
            ["all pass and no_action", "no_action_required", "off"],
            ["all pass and acid/base dose", "safety_pass_all_checks", "on"],
        ],
        columns=["condition", "safety_reason", "pump_status"],
    )
    st.dataframe(rules, width="stretch")


def render_evaluation_page() -> None:
    st.header("Evaluasi Model")
    st.write("Dataset v3.1 menggunakan split time-based 80:20 dan model sengaja dibuat realistis, bukan mengejar 100%.")

    train_rows, test_rows = load_split_counts()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dataset", "v3.1")
    with c2:
        st.metric("Train rows", f"{train_rows:,}")
    with c3:
        st.metric("Test rows", f"{test_rows:,}")

    st.subheader("Model Performance")
    st.dataframe(PERFORMANCE_INFO, width="stretch")

    df = load_dataset()
    if not df.empty:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Distribusi water_quality_status")
            st.bar_chart(df["water_quality_status"].value_counts().sort_index())
        with chart_col2:
            st.caption("Distribusi dosing_action")
            st.bar_chart(df["dosing_action"].value_counts().sort_index())

        with st.expander("Preview dataset v3.1"):
            st.dataframe(df.head(100), width="stretch")
    else:
        st.warning("Dataset v3.1 tidak ditemukan di folder data/.")

    if FINAL_REPORT_PATH.exists():
        with st.expander("Cuplikan final_model_report_v31_realistic_balanced.md"):
            report_text = FINAL_REPORT_PATH.read_text(encoding="utf-8")
            st.markdown(report_text[:6000])

    st.info("Hasil model perlu dibaca sebagai baseline synthetic-realistic, bukan validasi lapangan.")


def render_data_log_page() -> None:
    st.header("Data Log")
    ensure_log_file()
    ensure_automation_log_file()
    log_df = pd.read_csv(LOG_PATH)
    automation_log_df = pd.read_csv(AUTOMATION_LOG_PATH)
    st.write("Setiap hasil simulasi dari dashboard disimpan otomatis ke log CSV.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Jumlah log", f"{len(log_df):,}")
    with c2:
        st.download_button(
            "Download dashboard_decision_log.csv",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="dashboard_decision_log.csv",
            mime="text/csv",
        )

    st.subheader("Log Keputusan Dashboard")
    st.dataframe(log_df.tail(100), width="stretch")

    st.subheader("Automation Cycle Log")
    st.write("Log khusus untuk Simulated Automation / Digital Twin Mode.")
    st.download_button(
        "Download automation_cycle_log.csv",
        data=automation_log_df.to_csv(index=False).encode("utf-8"),
        file_name="automation_cycle_log.csv",
        mime="text/csv",
    )
    st.dataframe(automation_log_df.tail(100), width="stretch")

    st.subheader("Dataset Explorer")
    df = load_dataset()
    if df.empty:
        st.warning("Dataset v3.1 belum tersedia.")
        return
    selected_columns = st.multiselect(
        "Pilih kolom dataset",
        df.columns.tolist(),
        default=[
            "timestamp",
            "ph",
            "water_quality_status",
            "dosing_action",
            "safety_status",
            "pump_status",
        ],
    )
    max_rows = st.slider("Jumlah baris preview", 10, 300, 50, step=10)
    st.dataframe(df[selected_columns].head(max_rows), width="stretch")


def render_pump_panel(title: str, state: str) -> None:
    color = "green" if state == "ON" else "red" if "BLOCKED" in state else "blue"
    st.markdown(
        f"""
        <div class="pump-panel">
            <div class="pump-title">{title}</div>
            <div class="pump-state {color}">{state}</div>
            <div style="color:#52606d;font-size:.85rem;margin-top:.3rem;">Virtual actuator only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def schematic_status_class(status: str, default: str = "flow-blue") -> str:
    if status == "safety_pass" or status == "on":
        return "flow-green"
    if status == "safety_fail" or status == "blocked":
        return "flow-red"
    if status == "off":
        return "flow-gray"
    return default


def render_live_automation_schematic(latest: dict[str, Any]) -> None:
    """Render the virtual end-to-end automation flow without changing control logic."""
    safety_class = schematic_status_class(str(latest["safety_status"]))
    pump_class = schematic_status_class(str(latest["pump_status"]))
    sensor_class = "flow-red" if latest["sensor_status"] != "valid" else "flow-blue"
    recheck_class = "flow-green" if latest.get("dosing_success") else "flow-gray"
    steps = [
        ("01", "Virtual Sensor", f"pH {float(latest['ph_after_simulated']):.2f}", sensor_class),
        ("02", "Virtual ESP32", str(latest.get("data_packet_id", "-")), "flow-blue"),
        ("03", "ML Prediction", str(latest["predicted_dosing_action"]), "flow-blue"),
        ("04", "Safety Rule Controller", str(latest["safety_status"]), safety_class),
        ("05", "Virtual Relay", str(latest["pump_status"]).upper(), pump_class),
        ("06", "Virtual Acid/Base Pump", str(latest["automation_status"]), pump_class),
        ("07", "Recheck pH", f"delta {float(latest['delta_ph']):+.3f}", recheck_class),
        ("08", "Log", "Cycle saved" if latest.get("cycle", 0) else "Waiting", "flow-blue"),
    ]
    parts = []
    for index, (kicker, title, status, css_class) in enumerate(steps):
        parts.append(
            f'<div class="schematic-step {css_class}">'
            f'<div class="schematic-kicker">{kicker}</div>'
            f'<div class="schematic-title">{title}</div>'
            f'<div class="schematic-status">{status}</div>'
            "</div>"
        )
        if index < len(steps) - 1:
            parts.append('<div class="schematic-arrow">-&gt;</div>')
    html_schematic = (
        '<div class="schematic-wrap">'
        f"{''.join(parts)}"
        "</div>"
        '<div class="ph-meter-note">'
        "Virtual Sensor &rarr; Virtual ESP32 &rarr; ML Prediction &rarr; Safety Rule Controller &rarr; "
        "Virtual Relay &rarr; Virtual Pump &rarr; Recheck pH &rarr; Log"
        "</div>"
    )
    st.subheader("Live Automation Schematic")
    st.markdown(html_schematic, unsafe_allow_html=True)


def pump_panel_css(state: str) -> tuple[str, str, str]:
    if state == "ON":
        return "pump-active", "ON", "~~~"
    if "BLOCKED" in state:
        return "pump-blocked", "BLOCKED / OFF", "!"
    return "pump-off", "OFF / Monitoring Only", "-"


def render_animated_virtual_pump_panel(latest: dict[str, Any]) -> None:
    acid_css, acid_text, acid_symbol = pump_panel_css(str(latest["virtual_acid_pump"]))
    base_css, base_text, base_symbol = pump_panel_css(str(latest["virtual_base_pump"]))
    st.subheader("Animated Virtual Pump Panel")
    st.markdown(
        f"""
        <div class="animated-pump-grid">
            <div class="animated-pump {acid_css}">
                <div class="animated-pump-title">Virtual Acid Pump</div>
                <div class="animated-pump-state">ACID PUMP {acid_text}</div>
                <div class="animated-pump-meta">Virtual actuator only - No physical hardware connected</div>
                <div class="pump-flow-symbol">{acid_symbol}</div>
            </div>
            <div class="animated-pump {base_css}">
                <div class="animated-pump-title">Virtual Base Pump</div>
                <div class="animated-pump-state">BASE PUMP {base_text}</div>
                <div class="animated-pump-meta">Virtual actuator only - No physical hardware connected</div>
                <div class="pump-flow-symbol">{base_symbol}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_animated_dosing_flow(latest: dict[str, Any]) -> None:
    acid_on = latest["virtual_acid_pump"] == "ON"
    base_on = latest["virtual_base_pump"] == "ON"
    blocked = latest["pump_status"] == "blocked"
    acid_line = "active-acid" if acid_on else "blocked" if blocked else ""
    base_line = "active-base" if base_on else "blocked" if blocked else ""
    acid_node = "active" if acid_on else "blocked" if blocked else ""
    base_node = "active" if base_on else "blocked" if blocked else ""
    tank_node = "active" if latest["pump_status"] == "on" else "blocked" if blocked else ""
    st.subheader("Animated Dosing Flow")
    st.markdown(
        f"""
        <div class="dosing-flow">
            <div class="ph-meter-note">Simulated dosing flow - virtual actuator only - No physical hardware connected.</div>
            <div class="flow-row">
                <div class="flow-node {acid_node}">Acid Tank</div>
                <div class="flow-line {acid_line}"></div>
                <div class="flow-node {acid_node}">Acid Pump</div>
                <div class="flow-line {acid_line}"></div>
                <div class="flow-node {tank_node}">Aquaponic Tank</div>
            </div>
            <div class="flow-row">
                <div class="flow-node {base_node}">Base Tank</div>
                <div class="flow-line {base_line}"></div>
                <div class="flow-node {base_node}">Base Pump</div>
                <div class="flow-line {base_line}"></div>
                <div class="flow-node {tank_node}">Aquaponic Tank</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ph_distance_to_target(ph_value: float) -> float:
    if 6.8 <= ph_value <= 7.0:
        return 0.0
    return min(abs(ph_value - 6.8), abs(ph_value - 7.0))


def ph_meter_status(ph_before: float, ph_after: float, pump_status: str) -> str:
    before_distance = ph_distance_to_target(ph_before)
    after_distance = ph_distance_to_target(ph_after)
    if after_distance < before_distance - 0.005:
        return "toward target"
    if pump_status == "off" or abs(ph_after - ph_before) <= 0.02:
        return "stable"
    return "needs monitoring"


def ph_to_percent(ph_value: float) -> float:
    return clamp((ph_value - 5.5) / (8.5 - 5.5) * 100, 0, 100)


def render_ph_status_meter(latest: dict[str, Any]) -> None:
    ph_before = float(latest["ph_before"])
    ph_after = float(latest["ph_after_simulated"])
    delta_ph = float(latest["delta_ph"])
    status = ph_meter_status(ph_before, ph_after, str(latest["pump_status"]))
    marker_pct = ph_to_percent(ph_after)
    target_left = ph_to_percent(6.8)
    target_width = ph_to_percent(7.0) - target_left
    st.subheader("pH Status Meter")
    st.markdown(
        f"""
        <div class="ph-meter">
            <div class="ph-meter-grid">
                <div class="ph-stat">
                    <div class="ph-stat-label">ph_before</div>
                    <div class="ph-stat-value">{ph_before:.3f}</div>
                </div>
                <div class="ph-stat">
                    <div class="ph-stat-label">ph_after_simulated</div>
                    <div class="ph-stat-value">{ph_after:.3f}</div>
                </div>
                <div class="ph-stat">
                    <div class="ph-stat-label">delta_ph</div>
                    <div class="ph-stat-value">{delta_ph:+.3f}</div>
                </div>
                <div class="ph-stat">
                    <div class="ph-stat-label">target 6.8-7.0</div>
                    <div class="ph-stat-value">{status}</div>
                </div>
            </div>
            <div class="ph-track">
                <div class="ph-target-band" style="left:{target_left:.2f}%; width:{target_width:.2f}%;"></div>
                <div class="ph-marker" style="left:{marker_pct:.2f}%;"></div>
            </div>
            <div class="ph-meter-note">Scale shown from pH 5.5 to 8.5. The highlighted band marks the simulated target pH 6.8-7.0.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_automation_timeline(latest: dict[str, Any]) -> None:
    safety_failed = latest["safety_status"] == "safety_fail"
    pump_on = latest["pump_status"] == "on"
    pump_off = latest["pump_status"] == "off"
    safety_class = "timeline-fail" if safety_failed else "timeline-pass" if latest["safety_status"] == "safety_pass" else "timeline-off"
    pump_class = "timeline-fail" if safety_failed else "timeline-pass" if pump_on else "timeline-off" if pump_off else "timeline-neutral"
    recheck_class = "timeline-pass" if latest.get("dosing_success") else "timeline-off"
    steps = [
        ("01", "Sensor Read", "timeline-neutral"),
        ("02", "ESP32 Packet", "timeline-neutral"),
        ("03", "ML Prediction", "timeline-neutral"),
        ("04", "Safety Check", safety_class),
        ("05", "Pump Decision", pump_class),
        ("06", "Recheck pH", recheck_class),
        ("07", "Log Saved", "timeline-neutral"),
    ]
    timeline_html = "".join(
        f"""
        <div class="timeline-step {css_class}">
            <div class="timeline-number">{number}</div>
            <div class="timeline-title">{title}</div>
        </div>
        """
        for number, title, css_class in steps
    )
    st.subheader("Automation Timeline")
    st.markdown(
        f"""
        <div class="automation-timeline">
            {timeline_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_virtual_hardware_layer(latest: dict[str, Any]) -> None:
    st.subheader("Virtual Hardware Layer")
    st.caption(
        "Komponen berikut adalah virtual hardware layer untuk simulasi akademik, bukan koneksi hardware fisik."
    )

    controller_col, sensor_col, relay_col = st.columns(3)
    with controller_col:
        st.markdown("#### Virtual ESP32 / Edge Controller")
        controller_df = pd.DataFrame(
            [
                ["Controller status", "Simulated Online"],
                ["WiFi status", "Simulated Connected"],
                ["Communication mode", "Simulated MQTT/Serial"],
                ["Last packet timestamp", latest.get("last_packet_timestamp", "-")],
                ["Data packet ID", latest.get("data_packet_id", "-")],
            ],
            columns=["item", "status"],
        )
        st.dataframe(controller_df, hide_index=True, width="stretch")

    with sensor_col:
        st.markdown("#### Virtual Sensor Node")
        sensor_df = pd.DataFrame(
            [
                ["Virtual pH Sensor", f"{latest['ph_after_simulated']:.3f} pH"],
                ["Virtual Temperature Sensor", f"{float(latest['temperature_c']):.2f} C"],
                ["Virtual Water Level Sensor", f"{float(latest['water_level_pct']):.2f}%"],
                [
                    "Virtual Nitrogen Input",
                    f"NH3 {float(latest['ammonia_ppm']):.3f}, NO2 {float(latest['nitrite_ppm']):.3f}, NO3 {float(latest['nitrate_ppm']):.2f}",
                ],
                ["Sensor status", latest["sensor_status"]],
            ],
            columns=["virtual component", "reading"],
        )
        st.dataframe(sensor_df, hide_index=True, width="stretch")

    with relay_col:
        st.markdown("#### Virtual Relay & Pump")
        relay_df = pd.DataFrame(
            [
                ["Virtual Relay Acid Pump", virtual_relay_state(latest["virtual_acid_pump"])],
                ["Virtual Relay Base Pump", virtual_relay_state(latest["virtual_base_pump"])],
                ["Virtual Acid Pump status", latest["virtual_acid_pump"]],
                ["Virtual Base Pump status", latest["virtual_base_pump"]],
                ["Pump command source", "Safety Rule Controller"],
                ["Pump status", latest["pump_status"].upper()],
            ],
            columns=["virtual component", "status"],
        )
        st.dataframe(relay_df, hide_index=True, width="stretch")

    readiness_df = pd.DataFrame(
        [
            ["ESP32/Arduino", "planned physical controller"],
            ["pH sensor", "planned analog input"],
            ["Temperature sensor", "planned digital input"],
            ["Water level sensor", "planned analog/digital input"],
            ["Relay module", "planned output actuator driver"],
            ["Acid/Base pump", "planned actuator"],
            ["MQTT/Serial communication", "planned data bridge"],
        ],
        columns=["hardware component", "integration readiness"],
    )
    st.markdown("#### Hardware Readiness Table")
    st.dataframe(readiness_df, hide_index=True, width="stretch")


def render_simulated_dosing_panel(latest: dict[str, Any]) -> None:
    st.subheader("Simulated Dosing & Recheck pH")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ph_before", f"{latest['ph_before']:.3f}")
    with c2:
        st.metric("ph_after_simulated", f"{latest['ph_after_simulated']:.3f}")
    with c3:
        st.metric("delta_ph", f"{latest['delta_ph']:+.3f}")
    with c4:
        st.metric("dosing_success", str(latest["dosing_success"]))


def build_idle_automation_snapshot() -> dict[str, Any]:
    sensor_state = st.session_state.get("current_virtual_sensor_state", PRESET_SCENARIOS["normal_condition"])
    ph_value = float(sensor_state["ph"])
    return {
        "cycle": int(st.session_state.get("automation_cycle_count", 0)),
        "scenario_name": "idle_preview",
        "data_packet_id": "VHW-IDLE",
        "last_packet_timestamp": "-",
        "ph_before": ph_value,
        "ph_after_simulated": ph_value,
        "delta_ph": 0.0,
        "dosing_success": False,
        "temperature_c": sensor_state["temperature_c"],
        "water_level_pct": sensor_state["water_level_pct"],
        "ammonia_ppm": sensor_state["ammonia_ppm"],
        "nitrite_ppm": sensor_state["nitrite_ppm"],
        "nitrate_ppm": sensor_state["nitrate_ppm"],
        "sensor_status": sensor_state["sensor_status"],
        "confidence_score": sensor_state["confidence_score"],
        "predicted_water_quality_status": "Not run",
        "predicted_dosing_action": "Not run",
        "safety_status": "Not run",
        "safety_reason": "waiting_for_simulation",
        "pump_status": "off",
        "virtual_acid_pump": "OFF",
        "virtual_base_pump": "OFF",
        "automation_status": "Monitoring Only",
        "dashboard_alert": "Simulasi belum dijalankan",
    }


def render_automation_status_cards(latest: dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("pH", f"{latest['ph_after_simulated']:.3f}", f"{latest['delta_ph']:+.3f}")
    with c2:
        render_card("Water Quality", latest["predicted_water_quality_status"])
    with c3:
        render_card("Dosing Action", latest["predicted_dosing_action"])
    with c4:
        render_card("Safety", latest["safety_status"])
    with c5:
        render_card("Pump", latest["pump_status"])


def render_automation_visuals(history: list[dict[str, Any]], presentation_mode: bool = False) -> None:
    if not history:
        st.info("Belum ada siklus otomasi. Pilih skenario awal lalu tekan Start Simulation.")
        idle_snapshot = build_idle_automation_snapshot()
        if presentation_mode:
            render_automation_status_cards(idle_snapshot)
        render_live_automation_schematic(idle_snapshot)
        render_animated_virtual_pump_panel(idle_snapshot)
        render_animated_dosing_flow(idle_snapshot)
        if not presentation_mode:
            render_virtual_hardware_layer(idle_snapshot)
            render_simulated_dosing_panel(idle_snapshot)
        render_ph_status_meter(idle_snapshot)
        render_automation_timeline(idle_snapshot)
        return

    latest = history[-1]
    render_automation_status_cards(latest)
    if not presentation_mode:
        st.metric("Automation Status", latest["automation_status"])
        st.caption(latest["dashboard_alert"])
    render_live_automation_schematic(latest)
    if not presentation_mode:
        render_virtual_hardware_layer(latest)
    render_animated_virtual_pump_panel(latest)
    render_animated_dosing_flow(latest)
    if not presentation_mode:
        render_simulated_dosing_panel(latest)
    render_ph_status_meter(latest)
    render_automation_timeline(latest)

    if presentation_mode:
        return

    history_df = pd.DataFrame(history)
    chart_df = history_df.set_index("cycle")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.caption("pH history")
        st.line_chart(chart_df[["ph_before", "ph_after_simulated"]])
    with chart_col2:
        st.caption("water_level_pct history")
        st.line_chart(chart_df[["water_level_pct"]])

    st.subheader("Automation Cycle Log")
    st.dataframe(history_df.tail(50), width="stretch")


def render_simulated_automation_page() -> None:
    st.header("Simulated Automation / Digital Twin Automation Mode")
    st.markdown(
        """
        <div class="warn-box">
        Mode ini adalah simulasi otomasi berbasis digital twin. Komponen ESP32, sensor, relay, dan pompa pada
        dashboard ini merupakan virtual hardware layer untuk simulasi akademik. Belum ada koneksi hardware fisik
        pada tahap ini.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Alur real-time virtual: Virtual Sensor -> Virtual ESP32 -> ML Prediction -> Safety Rule Controller -> "
        "Virtual Relay -> Virtual Pump -> Recheck pH -> Log."
    )

    presentation_mode = st.toggle(
        "Presentation Mode",
        value=False,
        help=(
            "Aktifkan untuk tampilan ringkas yang cocok untuk screenshot PPT. "
            "Detail teknis tetap tersedia saat mode ini dimatikan."
        ),
    )
    if presentation_mode:
        st.caption(
            "Presentation Mode aktif: detail virtual hardware, readiness table, log siklus, dan tabel teknis panjang disembunyikan."
        )

    selected_scenario = st.selectbox(
        "Skenario awal",
        list(PRESET_SCENARIOS.keys()),
        key="automation_scenario_select",
    )
    ensure_automation_session_state(selected_scenario)

    c1, c2 = st.columns(2)
    with c1:
        interval_seconds = st.slider("Interval simulasi (detik)", 2, 10, 2, 1)
    with c2:
        total_cycles = st.slider("Jumlah siklus simulasi", 5, 50, 5, 1)

    b1, b2, b3 = st.columns(3)
    with b1:
        start_clicked = st.button("Start Simulation", type="primary")
    with b2:
        stop_clicked = st.button("Stop Simulation")
    with b3:
        reset_clicked = st.button("Reset Simulation")

    if reset_clicked:
        init_automation_state(selected_scenario)
        ensure_automation_log_file()
        st.success("Automation history dan virtual sensor state sudah direset.")

    if stop_clicked:
        st.session_state["automation_running"] = False
        st.warning("Stop flag diset. Jika simulasi sedang berjalan, siklus berikutnya akan berhenti pada rerun berikutnya.")

    status_placeholder = st.empty()
    visual_placeholder = st.empty()

    if start_clicked:
        init_automation_state(selected_scenario)
        ensure_automation_log_file()
        st.session_state["automation_running"] = True
        for _ in range(total_cycles):
            if not st.session_state.get("automation_running", False):
                break

            current_state = st.session_state["current_virtual_sensor_state"]
            virtual_sensor_state = evolve_virtual_sensor_state(current_state, selected_scenario)
            cycle_number = int(st.session_state["automation_cycle_count"]) + 1
            cycle_result = build_automation_cycle_result(
                selected_scenario,
                cycle_number,
                virtual_sensor_state,
            )
            st.session_state["automation_cycle_count"] = cycle_number
            st.session_state["automation_history"].append(cycle_result)
            append_automation_log(cycle_result)

            with status_placeholder.container():
                st.success(
                    f"Cycle {cycle_number}/{total_cycles} selesai: "
                    f"{cycle_result['automation_status']} | {cycle_result['dashboard_alert']}"
                )
            with visual_placeholder.container():
                render_automation_visuals(
                    st.session_state["automation_history"],
                    presentation_mode=presentation_mode,
                )
            time.sleep(interval_seconds)

        st.session_state["automation_running"] = False
        st.success("Simulasi selesai sesuai jumlah siklus terkontrol.")

    history = st.session_state.get("automation_history", [])
    if not start_clicked:
        render_automation_visuals(history, presentation_mode=presentation_mode)

    if not presentation_mode:
        with st.expander("Current virtual sensor state"):
            st.dataframe(pd.DataFrame([st.session_state["current_virtual_sensor_state"]]), width="stretch")


def route_page(menu: str) -> None:
    if menu == "Overview Sistem":
        render_overview()
    elif menu == "Input Sensor & Simulasi":
        render_input_page()
    elif menu == "Simulated Automation":
        render_simulated_automation_page()
    elif menu == "Hasil Prediksi ML":
        render_prediction_page()
    elif menu == "Safety Rule & Status Pompa":
        render_safety_page()
    elif menu == "Evaluasi Model":
        render_evaluation_page()
    elif menu == "Data Log":
        render_data_log_page()


def main() -> None:
    configure_page()
    ensure_log_file()
    ensure_automation_log_file()

    st.sidebar.title("Aquaponic Dashboard")
    st.sidebar.caption("IoT - ML - MBSE | v3.1")
    menu = st.sidebar.radio(
        "Menu",
        [
            "Overview Sistem",
            "Input Sensor & Simulasi",
            "Simulated Automation",
            "Hasil Prediksi ML",
            "Safety Rule & Status Pompa",
            "Evaluasi Model",
            "Data Log",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.info("Proof of concept akademik. Safety rule wajib sebelum pompa aktif.")

    try:
        route_page(menu)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    render_footer()


if __name__ == "__main__":
    main()
