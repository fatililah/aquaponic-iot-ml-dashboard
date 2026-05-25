from __future__ import annotations

import json
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


def get_current_result() -> dict[str, Any] | None:
    return st.session_state.get("latest_result")


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
        st.image(str(architecture_path), caption="Arsitektur fisik/sistem aquaponic", use_container_width=True)
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
    st.dataframe(pd.DataFrame([{key: result[key] for key in DEFAULT_FEATURES}]), use_container_width=True)
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
    st.dataframe(active_rule, use_container_width=True)

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
    st.dataframe(rules, use_container_width=True)


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
    st.dataframe(PERFORMANCE_INFO, use_container_width=True)

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
            st.dataframe(df.head(100), use_container_width=True)
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
    log_df = pd.read_csv(LOG_PATH)
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
    st.dataframe(log_df.tail(100), use_container_width=True)

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
    st.dataframe(df[selected_columns].head(max_rows), use_container_width=True)


def route_page(menu: str) -> None:
    if menu == "Overview Sistem":
        render_overview()
    elif menu == "Input Sensor & Simulasi":
        render_input_page()
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

    st.sidebar.title("Aquaponic Dashboard")
    st.sidebar.caption("IoT - ML - MBSE | v3.1")
    menu = st.sidebar.radio(
        "Menu",
        [
            "Overview Sistem",
            "Input Sensor & Simulasi",
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
