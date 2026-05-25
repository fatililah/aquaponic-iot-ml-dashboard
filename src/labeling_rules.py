from __future__ import annotations

from typing import Any


WATER_QUALITY_CLASSES = ["Normal", "Waspada", "Kritis", "Darurat"]
DOSING_ACTION_CLASSES = [
    "no_action",
    "acid_low_dose",
    "acid_medium_dose",
    "base_low_dose",
    "base_medium_dose",
    "manual_check",
]


def label_water_quality(row: dict[str, Any]) -> str:
    """Assign the water quality risk class using aquaponic rule thresholds."""
    ph = float(row["ph"])
    temperature_c = float(row["temperature_c"])
    water_level_pct = float(row["water_level_pct"])
    ammonia_ppm = float(row["ammonia_ppm"])
    nitrite_ppm = float(row["nitrite_ppm"])
    nitrate_ppm = float(row["nitrate_ppm"])
    sensor_status = row["sensor_status"]

    if (
        sensor_status == "error"
        or water_level_pct < 45
        or ph < 5.8
        or ph > 8.2
        or ammonia_ppm > 1.0
        or nitrite_ppm > 1.0
        or nitrate_ppm > 200
        or temperature_c < 20
        or temperature_c > 34
    ):
        return "Darurat"

    if (
        5.8 <= ph < 6.2
        or 7.5 < ph <= 8.2
        or 45 <= water_level_pct < 60
        or 0.50 < ammonia_ppm <= 1.0
        or 0.50 < nitrite_ppm <= 1.0
        or 150 < nitrate_ppm <= 200
        or 20 <= temperature_c < 22
        or 32 < temperature_c <= 34
    ):
        return "Kritis"

    if (
        6.2 <= ph < 6.5
        or 7.2 < ph <= 7.5
        or 60 <= water_level_pct < 75
        or 0.25 < ammonia_ppm <= 0.50
        or 0.10 < nitrite_ppm <= 0.50
        or 100 < nitrate_ppm <= 150
        or 22 <= temperature_c < 24
        or 30 < temperature_c <= 32
        or sensor_status == "perlu_kalibrasi"
    ):
        return "Waspada"

    if (
        6.5 <= ph <= 7.2
        and 24 <= temperature_c <= 30
        and water_level_pct >= 75
        and ammonia_ppm <= 0.25
        and nitrite_ppm <= 0.10
        and 20 <= nitrate_ppm <= 100
        and sensor_status == "valid"
    ):
        return "Normal"

    return "Waspada"


def label_dosing_action(row: dict[str, Any]) -> str:
    """Assign the pH dosing action recommendation before safety enforcement."""
    ph = float(row["ph"])
    sensor_status = row["sensor_status"]
    water_level_pct = float(row["water_level_pct"])
    ammonia_ppm = float(row["ammonia_ppm"])
    nitrite_ppm = float(row["nitrite_ppm"])
    confidence_score = float(row["confidence_score"])
    dosing_cycle_count = int(row["dosing_cycle_count"])
    time_since_last_dosing_min = float(row["time_since_last_dosing_min"])
    cooldown_time_min = float(row["cooldown_time_min"])

    if (
        sensor_status != "valid"
        or water_level_pct < 50
        or ph < 5.8
        or ph > 8.2
        or ammonia_ppm > 1.0
        or nitrite_ppm > 1.0
        or confidence_score < 0.70
        or dosing_cycle_count >= 3
        or time_since_last_dosing_min < cooldown_time_min
    ):
        return "manual_check"

    if 6.5 <= ph <= 7.2 and sensor_status == "valid" and water_level_pct >= 50:
        return "no_action"

    if (
        6.2 <= ph < 6.5
        and sensor_status == "valid"
        and water_level_pct >= 50
        and time_since_last_dosing_min >= cooldown_time_min
        and dosing_cycle_count < 3
    ):
        return "base_low_dose"

    if (
        5.8 <= ph < 6.2
        and sensor_status == "valid"
        and water_level_pct >= 50
        and time_since_last_dosing_min >= cooldown_time_min
        and dosing_cycle_count < 3
    ):
        return "base_medium_dose"

    if (
        7.2 < ph <= 7.5
        and sensor_status == "valid"
        and water_level_pct >= 50
        and time_since_last_dosing_min >= cooldown_time_min
        and dosing_cycle_count < 3
    ):
        return "acid_low_dose"

    if (
        7.5 < ph <= 8.2
        and sensor_status == "valid"
        and water_level_pct >= 50
        and time_since_last_dosing_min >= cooldown_time_min
        and dosing_cycle_count < 3
    ):
        return "acid_medium_dose"

    return "manual_check"


def dosing_type_from_action(dosing_action: str) -> str:
    """Map the classifier action into the physical dosing type."""
    mapping = {
        "no_action": "none",
        "acid_low_dose": "acid",
        "acid_medium_dose": "acid",
        "base_low_dose": "base",
        "base_medium_dose": "base",
        "manual_check": "manual",
    }
    return mapping.get(dosing_action, "manual")
