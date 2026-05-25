from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils import PROJECT_ROOT, ensure_project_dirs


MODEL_DIR = PROJECT_ROOT / "models"
REPORT_PATH = PROJECT_ROOT / "reports" / "simulation_decision_v31.md"
WATER_MODEL_PATH = MODEL_DIR / "water_quality_classifier_v31.joblib"
DOSING_MODEL_PATH = MODEL_DIR / "dosing_action_classifier_v31.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns_v31.json"

FEATURE_COLUMNS = [
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


SIMULATION_CASES = [
    {
        "scenario_name": "normal_condition",
        "ph": 6.8,
        "temperature_c": 27,
        "water_level_pct": 85,
        "ammonia_ppm": 0.10,
        "nitrite_ppm": 0.05,
        "nitrate_ppm": 60,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.90,
    },
    {
        "scenario_name": "high_ph_condition",
        "ph": 7.8,
        "temperature_c": 28,
        "water_level_pct": 85,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 80,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.85,
    },
    {
        "scenario_name": "low_ph_condition",
        "ph": 6.0,
        "temperature_c": 27,
        "water_level_pct": 82,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 70,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.85,
    },
    {
        "scenario_name": "low_water_level_condition",
        "ph": 7.7,
        "temperature_c": 28,
        "water_level_pct": 40,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 80,
        "sensor_status": "valid",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.85,
    },
    {
        "scenario_name": "sensor_error_condition",
        "ph": 7.8,
        "temperature_c": 28,
        "water_level_pct": 85,
        "ammonia_ppm": 0.20,
        "nitrite_ppm": 0.10,
        "nitrate_ppm": 80,
        "sensor_status": "error",
        "time_since_last_dosing_min": 30,
        "dosing_cycle_count": 0,
        "confidence_score": 0.60,
    },
]


def load_models() -> tuple[Any, Any, dict[str, Any]]:
    """Load v3.1 classifiers and feature metadata from disk."""
    missing_paths = [
        path
        for path in (WATER_MODEL_PATH, DOSING_MODEL_PATH, FEATURE_COLUMNS_PATH)
        if not path.exists()
    ]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required v3.1 model artifact(s): {missing_text}")

    water_quality_model = joblib.load(WATER_MODEL_PATH)
    dosing_action_model = joblib.load(DOSING_MODEL_PATH)
    metadata = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    return water_quality_model, dosing_action_model, metadata


def apply_safety_rule(case: dict[str, Any], dosing_action: str) -> dict[str, str]:
    """Apply Safety Rule Controller before any pump activation."""
    safety_checks = [
        (case["sensor_status"] != "valid", "sensor_not_valid"),
        (float(case["water_level_pct"]) < 50, "water_level_low"),
        (float(case["confidence_score"]) < 0.70, "low_confidence"),
        (float(case["time_since_last_dosing_min"]) < 10, "cooldown_not_met"),
        (int(case["dosing_cycle_count"]) >= 3, "max_cycle_exceeded"),
        (dosing_action == "manual_check", "manual_check_required"),
        (float(case["ph"]) < 5.8, "ph_too_low"),
        (float(case["ph"]) > 8.2, "ph_too_high"),
        (float(case["ammonia_ppm"]) > 1.0, "ammonia_high"),
        (float(case["nitrite_ppm"]) > 1.0, "nitrite_high"),
    ]

    for failed, reason in safety_checks:
        if failed:
            return {
                "safety_status": "safety_fail",
                "safety_reason": reason,
                "pump_status": "blocked",
            }

    if dosing_action == "no_action":
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
    case: dict[str, Any],
    predicted_water_quality_status: str,
    predicted_dosing_action: str,
    safety_result: dict[str, str],
) -> str:
    """Generate a concise dashboard alert for the simulated decision."""
    if safety_result["safety_status"] == "safety_fail":
        reason = safety_result["safety_reason"]
        if reason == "sensor_not_valid":
            return "Sensor bermasalah, pompa diblokir"
        if reason == "water_level_low":
            return "Level air rendah, pompa diblokir"
        if reason == "low_confidence":
            return "Confidence rendah, perlu manual check"
        if reason == "cooldown_not_met":
            return "Cooldown belum terpenuhi, pompa diblokir"
        if reason == "max_cycle_exceeded":
            return "Siklus dosing maksimum tercapai"
        if reason == "manual_check_required":
            return "Manual check diperlukan"
        if reason in {"ph_too_low", "ph_too_high"}:
            return "pH ekstrem, perlu pemeriksaan manual"
        if reason in {"ammonia_high", "nitrite_high"}:
            return "Nitrogen tinggi, perlu pemeriksaan manual"
        return "Safety rule gagal, pompa diblokir"

    if predicted_dosing_action == "no_action":
        if predicted_water_quality_status == "Normal":
            return "Kondisi normal, pompa off"
        return "Tidak ada dosing, pantau kondisi air"

    action_alerts = {
        "acid_low_dose": "pH tinggi ringan, pompa acid low dose aktif",
        "acid_medium_dose": "pH tinggi sedang, pompa acid medium dose aktif",
        "base_low_dose": "pH rendah ringan, pompa base low dose aktif",
        "base_medium_dose": "pH rendah sedang, pompa base medium dose aktif",
    }
    return action_alerts.get(predicted_dosing_action, "Keputusan sistem selesai")


def simulate_case(
    case: dict[str, Any],
    water_quality_model: Any,
    dosing_action_model: Any,
) -> dict[str, Any]:
    """Run one case through classifier predictions and safety controller."""
    input_df = pd.DataFrame([case])[FEATURE_COLUMNS]
    predicted_water_quality_status = str(water_quality_model.predict(input_df)[0])
    predicted_dosing_action = str(dosing_action_model.predict(input_df)[0])
    safety_result = apply_safety_rule(case, predicted_dosing_action)
    dashboard_alert = generate_dashboard_alert(
        case,
        predicted_water_quality_status,
        predicted_dosing_action,
        safety_result,
    )

    return {
        "scenario_name": case["scenario_name"],
        "predicted_water_quality_status": predicted_water_quality_status,
        "predicted_dosing_action": predicted_dosing_action,
        "safety_status": safety_result["safety_status"],
        "safety_reason": safety_result["safety_reason"],
        "pump_status": safety_result["pump_status"],
        "dashboard_alert": dashboard_alert,
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a small DataFrame into markdown without optional tabulate dependency."""
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(lines)


def write_report(
    input_df: pd.DataFrame,
    result_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Write the end-to-end decision simulation report."""
    ensure_project_dirs()
    report_lines = [
        "# Simulation Decision V3.1",
        "",
        "## Ringkasan Tujuan",
        "",
        "Simulasi ini menguji alur end-to-end sistem aquaponic ML-assisted automation: input sensor masuk ke Water Quality Classifier, kemudian Dosing Action Classifier, lalu hasil rekomendasi melewati Safety Rule Controller sebelum menentukan status pompa dan alert dashboard.",
        "",
        "## Model yang Digunakan",
        "",
        f"- Water Quality Classifier: `{WATER_MODEL_PATH.name}`",
        f"- Dosing Action Classifier: `{DOSING_MODEL_PATH.name}`",
        f"- Feature metadata: `{FEATURE_COLUMNS_PATH.name}`",
        f"- Water quality model detail: `{metadata.get('water_quality_model', 'unknown')}`",
        f"- Dosing action model detail: `{metadata.get('dosing_action_model', 'unknown')}`",
        "",
        "## Fitur Input Utama",
        "",
        ", ".join(FEATURE_COLUMNS),
        "",
        "## Tabel Input Tiap Skenario",
        "",
        dataframe_to_markdown(input_df),
        "",
        "## Tabel Hasil Prediksi dan Keputusan Sistem",
        "",
        dataframe_to_markdown(result_df),
        "",
        "## Catatan Keputusan Sistem",
        "",
        "Output machine learning tidak langsung mengaktifkan pompa. Prediksi `dosing_action` hanya menjadi rekomendasi awal. Safety Rule Controller tetap memeriksa sensor_status, water_level_pct, confidence_score, cooldown, dosing_cycle_count, pH ekstrem, ammonia_ppm, nitrite_ppm, dan kondisi manual_check sebelum pompa boleh aktif.",
        "",
        "Jika Safety Rule Controller gagal, `safety_status` menjadi `safety_fail` dan `pump_status` menjadi `blocked`. Jika aksi adalah `no_action`, pompa tetap `off`. Pompa hanya `on` ketika rekomendasi dosing adalah acid/base dose dan seluruh pemeriksaan safety lolos.",
        "",
        "## Keterbatasan",
        "",
        "Simulasi ini masih berbasis model yang dilatih dari synthetic-realistic dataset v3.1. Hasilnya berguna untuk demonstrasi alur sistem dan pengujian awal, tetapi belum boleh dianggap sebagai validasi lapangan. Threshold, akurasi model, dan respons pompa tetap perlu diuji menggunakan data aquaponic nyata.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    water_quality_model, dosing_action_model, metadata = load_models()
    input_df = pd.DataFrame(SIMULATION_CASES)
    results = [
        simulate_case(case, water_quality_model, dosing_action_model)
        for case in SIMULATION_CASES
    ]
    result_df = pd.DataFrame(results)
    write_report(input_df, result_df, metadata)

    print("Simulation decision v3.1 completed.")
    print(f"Report: {REPORT_PATH}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
