from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from labeling_rules import dosing_type_from_action
from safety_rules import apply_safety_rule
from utils import PROJECT_ROOT, ensure_project_dirs, load_config


RANDOM_SEED = 42
DATASET_VERSION = "v3.1-bogor-realistic-balanced"
N_ROWS = 10000
OUTPUT_DATASET = PROJECT_ROOT / "data" / "aquaponic_synthetic_dataset_v31_bogor_realistic_balanced.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "aquaponic_train_v31.csv"
TEST_PATH = PROJECT_ROOT / "data" / "aquaponic_test_v31.csv"

WATER_CLASSES = ["Normal", "Waspada", "Kritis", "Darurat"]
DOSING_CLASSES = [
    "no_action",
    "acid_low_dose",
    "acid_medium_dose",
    "base_low_dose",
    "base_medium_dose",
    "manual_check",
]

SCENARIOS = [
    "normal_strong",
    "normal_strong",
    "normal_boundary",
    "manual_safety_signature",
    "manual_safety_signature",
    "bogor_normal_humid",
    "bogor_rain_ph_drop",
    "ph_low_mild",
    "ph_low_moderate",
    "ph_high_mild",
    "ph_high_moderate",
    "nitrogen_boundary",
    "low_water_boundary",
    "semi_outdoor_sensor_drift",
    "biofilter_immature",
    "post_dosing_variable_response",
    "operator_measurement_variation",
]

OUTPUT_COLUMNS = [
    "data_id",
    "timestamp",
    "dataset_version",
    "input_source",
    "scenario_type",
    "sensor_id",
    "ph",
    "temperature_c",
    "water_level_pct",
    "ammonia_ppm",
    "nitrite_ppm",
    "nitrate_ppm",
    "sensor_status",
    "ph_before",
    "time_since_last_dosing_min",
    "dosing_cycle_count",
    "mixing_time_min",
    "confidence_score",
    "rain_event",
    "recent_feeding_level",
    "biofilter_maturity",
    "water_turbidity_ntu",
    "pump_response_factor",
    "operator_measurement_error",
    "water_quality_status",
    "recommended_action",
    "dosing_action",
    "dosing_type",
    "pump_duration_sec",
    "max_pump_duration_sec",
    "cooldown_time_min",
    "safety_status",
    "safety_reason",
    "pump_status",
    "ph_after",
    "delta_ph",
    "dosing_success",
    "overdosing_risk",
    "manual_override",
    "dominant_feature",
    "dashboard_alert",
]


def clip_round(value: float, low: float, high: float, digits: int = 2) -> float:
    """Clip a numeric value and round it for storage."""
    return round(float(np.clip(value, low, high)), digits)


def choose_other(label: str, labels: list[str], rng: np.random.Generator) -> str:
    """Choose a different label from the same target space."""
    choices = [item for item in labels if item != label]
    return str(rng.choice(choices))


def build_scenarios(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Create a balanced, time-ordered scenario sequence with mild seasonal drift."""
    base = np.array(SCENARIOS * math.ceil(n_rows / len(SCENARIOS)))[:n_rows]
    blocks = np.array_split(base, 10)
    shuffled_blocks = []
    for block in blocks:
        rng.shuffle(block)
        shuffled_blocks.append(block)
    return np.concatenate(shuffled_blocks)


def base_true_values(scenario: str, rng: np.random.Generator, progress: float) -> dict[str, object]:
    """Generate latent Bogor aquaponic values before measurement noise."""
    rain_probability = 0.34 + (0.08 if scenario in {"bogor_rain_ph_drop", "low_water_boundary"} else 0.0)
    if scenario == "normal_strong":
        rain_probability = 0.18
    elif scenario == "normal_boundary":
        rain_probability = 0.28
    rain_event = int(rng.random() < rain_probability)
    recent_feeding_level = str(rng.choice(["low", "medium", "high"], p=[0.25, 0.50, 0.25]))
    biofilter_maturity = str(rng.choice(["immature", "maturing", "mature"], p=[0.18, 0.30, 0.52]))

    temp_seasonal = 1.1 * math.sin(progress * 2 * math.pi)
    ph = rng.normal(6.9, 0.28)
    temperature = rng.normal(27.3 + temp_seasonal, 1.2)
    water_level = rng.normal(78.0, 9.0)
    ammonia = rng.lognormal(mean=-2.45, sigma=0.65)
    nitrite = rng.lognormal(mean=-3.0, sigma=0.75)
    nitrate = rng.normal(78.0, 34.0)

    if scenario == "normal_strong":
        ph = rng.uniform(6.70, 7.00)
        temperature = rng.uniform(25.0, 29.0)
        water_level = rng.uniform(82, 95)
        ammonia = rng.uniform(0.02, 0.16)
        nitrite = rng.uniform(0.005, 0.055)
        nitrate = rng.uniform(35, 90)
        recent_feeding_level = str(rng.choice(["low", "medium"], p=[0.45, 0.55]))
        biofilter_maturity = "mature"
    elif scenario == "normal_boundary":
        ph = float(rng.choice([rng.uniform(6.45, 6.60), rng.uniform(7.10, 7.25)]))
        temperature = rng.uniform(24.5, 30.0)
        water_level = rng.uniform(76, 92)
        ammonia = rng.uniform(0.05, 0.23)
        nitrite = rng.uniform(0.01, 0.095)
        nitrate = rng.uniform(25, 105)
        biofilter_maturity = str(rng.choice(["maturing", "mature"], p=[0.25, 0.75]))
    elif scenario == "manual_safety_signature":
        signature = str(rng.choice(["sensor", "level", "cooldown", "cycle", "ph_extreme", "nitrogen", "confidence"]))
        ph = rng.uniform(6.2, 7.6)
        water_level = rng.uniform(64, 90)
        if signature == "level":
            water_level = rng.uniform(35, 49)
        elif signature == "ph_extreme":
            ph = float(rng.choice([rng.uniform(5.25, 5.78), rng.uniform(8.22, 8.75)]))
        elif signature == "nitrogen":
            ammonia = rng.uniform(1.05, 1.55)
            nitrite = rng.uniform(0.85, 1.45)
            nitrate = rng.uniform(130, 230)
    elif scenario == "bogor_rain_ph_drop":
        ph = rng.uniform(6.25, 7.25)
        water_level = rng.uniform(70, 94)
    elif scenario == "ph_low_mild":
        ph = rng.uniform(6.10, 6.55)
    elif scenario == "ph_low_moderate":
        ph = rng.uniform(5.70, 6.35)
    elif scenario == "ph_high_mild":
        ph = rng.uniform(7.05, 7.60)
    elif scenario == "ph_high_moderate":
        ph = rng.uniform(7.40, 8.28)
    elif scenario == "nitrogen_boundary":
        ph = rng.uniform(6.45, 7.35)
        ammonia = rng.uniform(0.32, 0.78)
        nitrite = rng.uniform(0.20, 0.72)
        nitrate = rng.uniform(95, 180)
    elif scenario == "low_water_boundary":
        water_level = rng.uniform(42, 63)
    elif scenario == "semi_outdoor_sensor_drift":
        ph = rng.uniform(6.25, 7.55)
        water_level = rng.uniform(58, 90)
    elif scenario == "biofilter_immature":
        biofilter_maturity = "immature"
        ammonia = rng.uniform(0.45, 1.25)
        nitrite = rng.uniform(0.35, 1.20)
        nitrate = rng.uniform(70, 210)
    elif scenario == "post_dosing_variable_response":
        ph = float(rng.choice([rng.uniform(6.0, 6.45), rng.uniform(7.25, 7.95)]))
    elif scenario == "operator_measurement_variation":
        ph = rng.uniform(6.15, 7.65)
        ammonia = rng.uniform(0.18, 0.62)
        nitrite = rng.uniform(0.06, 0.55)

    if rain_event and scenario != "normal_strong":
        water_level += rng.uniform(3, 12)
        ph -= rng.uniform(0.05, 0.20)
        nitrate *= rng.uniform(0.88, 1.02)

    if recent_feeding_level == "high":
        ammonia *= rng.uniform(1.15, 1.55)
        nitrite *= rng.uniform(1.10, 1.45)
        nitrate *= rng.uniform(1.05, 1.18)

    if biofilter_maturity == "immature":
        ammonia *= rng.uniform(1.25, 1.80)
        nitrite *= rng.uniform(1.20, 1.75)
    elif biofilter_maturity == "mature":
        ammonia *= rng.uniform(0.70, 0.95)
        nitrite *= rng.uniform(0.70, 0.95)

    water_turbidity_ntu = float(np.clip(rng.normal(9, 4), 1, 32))
    if rain_event:
        water_turbidity_ntu += rng.uniform(3, 12)
    if recent_feeding_level == "high":
        water_turbidity_ntu += rng.uniform(2, 8)

    pump_response_factor = float(np.clip(rng.normal(0.92, 0.22), 0.4, 1.3))
    operator_measurement_error = float(np.clip(rng.normal(0.0, 0.08), -0.25, 0.25))

    outdoor_bias = scenario in {"bogor_rain_ph_drop", "semi_outdoor_sensor_drift", "operator_measurement_variation"}
    if scenario == "normal_strong":
        sensor_status = "valid"
    elif scenario == "manual_safety_signature" and rng.random() < 0.45:
        sensor_status = str(rng.choice(["perlu_kalibrasi", "error"], p=[0.48, 0.52]))
    else:
        sensor_status = str(
            rng.choice(
                ["valid", "perlu_kalibrasi", "error"],
                p=[0.875, 0.088, 0.037] if outdoor_bias else [0.91, 0.06, 0.03],
            )
        )

    return {
        "rain_event": rain_event,
        "recent_feeding_level": recent_feeding_level,
        "biofilter_maturity": biofilter_maturity,
        "water_turbidity_ntu": round(water_turbidity_ntu, 2),
        "pump_response_factor": round(pump_response_factor, 3),
        "operator_measurement_error": round(operator_measurement_error, 3),
        "ph_true": float(np.clip(ph, 5.1, 8.9)),
        "temperature_true": float(np.clip(temperature, 21, 34)),
        "water_level_true": float(np.clip(water_level, 30, 100)),
        "ammonia_true": float(np.clip(ammonia, 0, 2.2)),
        "nitrite_true": float(np.clip(nitrite, 0, 2.2)),
        "nitrate_true": float(np.clip(nitrate, 5, 250)),
        "sensor_status": sensor_status,
    }


def apply_sensor_noise(values: dict[str, object], rng: np.random.Generator) -> dict[str, float]:
    """Apply realistic sensor and manual measurement noise to stored inputs."""
    ammonia_factor = rng.choice([-1, 1]) * rng.uniform(0.05, 0.20)
    nitrite_factor = rng.choice([-1, 1]) * rng.uniform(0.05, 0.25)
    nitrate_factor = rng.choice([-1, 1]) * rng.uniform(0.05, 0.20)
    operator_error = float(values["operator_measurement_error"])

    ph = float(values["ph_true"]) + rng.normal(0, 0.12) + operator_error
    temperature = float(values["temperature_true"]) + rng.normal(0, 0.5)
    water_level = float(values["water_level_true"]) + rng.normal(0, 3.0)
    ammonia = float(values["ammonia_true"]) * (1 + ammonia_factor)
    nitrite = float(values["nitrite_true"]) * (1 + nitrite_factor)
    nitrate = float(values["nitrate_true"]) * (1 + nitrate_factor)

    return {
        "ph": clip_round(ph, 4.5, 9.5, 2),
        "temperature_c": clip_round(temperature, 15, 40, 2),
        "water_level_pct": clip_round(water_level, 0, 100, 1),
        "ammonia_ppm": clip_round(ammonia, 0, 2.5, 3),
        "nitrite_ppm": clip_round(nitrite, 0, 2.5, 3),
        "nitrate_ppm": clip_round(nitrate, 0, 260, 2),
    }


def deterministic_water_label(row: dict[str, object]) -> str:
    """Create a base water quality label using threshold-like academic rules."""
    ph = float(row["ph"])
    temp = float(row["temperature_c"])
    level = float(row["water_level_pct"])
    ammonia = float(row["ammonia_ppm"])
    nitrite = float(row["nitrite_ppm"])
    nitrate = float(row["nitrate_ppm"])
    sensor = str(row["sensor_status"])
    turbidity = float(row["water_turbidity_ntu"])
    biofilter = str(row["biofilter_maturity"])
    feeding = str(row["recent_feeding_level"])

    if (
        sensor == "error"
        or level < 43
        or ph < 5.75
        or ph > 8.25
        or ammonia > 1.08
        or nitrite > 1.08
        or nitrate > 215
        or temp < 20
        or temp > 34
        or (biofilter == "immature" and ammonia > 0.95 and nitrite > 0.85)
    ):
        return "Darurat"

    if (
        5.75 <= ph < 6.20
        or 7.55 < ph <= 8.25
        or 43 <= level < 60
        or 0.52 < ammonia <= 1.08
        or 0.52 < nitrite <= 1.08
        or 155 < nitrate <= 215
        or temp < 22
        or temp > 32
        or turbidity > 22
    ):
        return "Kritis"

    if (
        6.20 <= ph < 6.52
        or 7.18 < ph <= 7.55
        or 60 <= level < 75
        or 0.24 < ammonia <= 0.58
        or 0.10 < nitrite <= 0.58
        or 100 < nitrate <= 160
        or temp < 24
        or temp > 30.8
        or sensor == "perlu_kalibrasi"
        or feeding == "high"
    ):
        return "Waspada"

    return "Normal"


def apply_boundary_ambiguity(label: str, row: dict[str, object], rng: np.random.Generator) -> str:
    """Flip labels probabilistically near common aquaponic threshold boundaries."""
    ph = float(row["ph"])
    level = float(row["water_level_pct"])
    ammonia = float(row["ammonia_ppm"])
    nitrite = float(row["nitrite_ppm"])

    candidates: list[tuple[str, str, float]] = [
        ("Normal", "Waspada", 0.44) if 7.15 <= ph <= 7.30 else ("", "", 0),
        ("Waspada", "Kritis", 0.42) if 7.45 <= ph <= 7.60 else ("", "", 0),
        ("Waspada", "Kritis", 0.42) if 6.15 <= ph <= 6.30 else ("", "", 0),
        ("Kritis", "Darurat", 0.40) if 45 <= level <= 55 else ("", "", 0),
        ("Waspada", "Kritis", 0.45) if 0.45 <= ammonia <= 0.60 else ("", "", 0),
        ("Waspada", "Kritis", 0.45) if 0.45 <= nitrite <= 0.60 else ("", "", 0),
    ]

    for low_label, high_label, probability in candidates:
        if not low_label:
            continue
        if label in {low_label, high_label} and rng.random() < probability:
            return high_label if label == low_label else low_label
    return label


def is_water_boundary(row: dict[str, object]) -> bool:
    """Return True when a row is close enough to a decision boundary for label noise."""
    ph = float(row["ph"])
    level = float(row["water_level_pct"])
    ammonia = float(row["ammonia_ppm"])
    nitrite = float(row["nitrite_ppm"])
    nitrate = float(row["nitrate_ppm"])
    return (
        6.10 <= ph <= 6.35
        or 7.12 <= ph <= 7.62
        or 43 <= level <= 62
        or 0.42 <= ammonia <= 0.68
        or 0.42 <= nitrite <= 0.68
        or 95 <= nitrate <= 170
        or str(row["sensor_status"]) == "perlu_kalibrasi"
    )


def is_extreme(row: dict[str, object]) -> bool:
    """Protect obvious extreme cases from artificial label noise."""
    return (
        float(row["ph"]) < 5.65
        or float(row["ph"]) > 8.35
        or float(row["water_level_pct"]) < 40
        or float(row["ammonia_ppm"]) > 1.25
        or float(row["nitrite_ppm"]) > 1.25
        or str(row["sensor_status"]) == "error"
    )


def has_manual_safety_signature(row: dict[str, object]) -> bool:
    """Return True when manual_check has visible safety evidence in input features."""
    return (
        str(row["sensor_status"]) != "valid"
        or float(row["water_level_pct"]) < 50
        or float(row["confidence_score"]) < 0.70
        or float(row["time_since_last_dosing_min"]) < float(row["cooldown_time_min"])
        or int(row["dosing_cycle_count"]) >= 3
        or float(row["ph"]) < 5.8
        or float(row["ph"]) > 8.2
        or float(row["ammonia_ppm"]) > 1.0
        or float(row["nitrite_ppm"]) > 1.0
    )


def water_quality_label(row: dict[str, object], rng: np.random.Generator) -> str:
    """Assign probabilistic water quality label with boundary-only label noise."""
    scenario = str(row["scenario_type"])
    if scenario == "normal_strong":
        return "Normal" if rng.random() < 0.83 else "Waspada"
    if scenario == "normal_boundary":
        return "Normal" if rng.random() < 0.33 else "Waspada"

    label = deterministic_water_label(row)
    label = apply_boundary_ambiguity(label, row, rng)
    if is_water_boundary(row) and not is_extreme(row) and rng.random() < 0.18:
        nearby = {
            "Normal": ["Waspada"],
            "Waspada": ["Normal", "Kritis"],
            "Kritis": ["Waspada", "Darurat"],
            "Darurat": ["Kritis"],
        }
        label = str(rng.choice(nearby[label]))
    return label


def dosing_base_label(row: dict[str, object]) -> str:
    """Create a base dosing action recommendation from noisy inputs."""
    ph = float(row["ph"])
    level = float(row["water_level_pct"])
    ammonia = float(row["ammonia_ppm"])
    nitrite = float(row["nitrite_ppm"])
    confidence = float(row["confidence_score"])
    cycle = int(row["dosing_cycle_count"])
    since = float(row["time_since_last_dosing_min"])
    cooldown = float(row["cooldown_time_min"])
    sensor = str(row["sensor_status"])

    if (
        sensor != "valid"
        or level < 47
        or ph < 5.75
        or ph > 8.25
        or ammonia > 1.05
        or nitrite > 1.05
        or confidence < 0.66
        or cycle >= 3
        or since < cooldown
    ):
        return "manual_check"
    if 6.45 <= ph <= 7.25:
        return "no_action"
    if 6.12 <= ph < 6.45:
        return "base_low_dose"
    if 5.75 <= ph < 6.12:
        return "base_medium_dose"
    if 7.25 < ph <= 7.58:
        return "acid_low_dose"
    if 7.58 < ph <= 8.25:
        return "acid_medium_dose"
    return "manual_check"


def is_dosing_boundary(row: dict[str, object]) -> bool:
    """Return True when a row is near dosing action boundaries."""
    ph = float(row["ph"])
    confidence = float(row["confidence_score"])
    level = float(row["water_level_pct"])
    since = float(row["time_since_last_dosing_min"])
    cooldown = float(row["cooldown_time_min"])
    return (
        6.08 <= ph <= 6.52
        or 7.18 <= ph <= 7.65
        or 46 <= level <= 56
        or 0.66 <= confidence <= 0.76
        or cooldown - 4 <= since <= cooldown + 8
    )


def dosing_label(row: dict[str, object], rng: np.random.Generator) -> str:
    """Assign dosing action with boundary ambiguity and boundary-only label noise."""
    label = dosing_base_label(row)
    ph = float(row["ph"])
    ammonia = float(row["ammonia_ppm"])
    nitrite = float(row["nitrite_ppm"])

    if has_manual_safety_signature(row) and rng.random() < 0.96:
        return "manual_check"

    if 7.15 <= ph <= 7.32 and label in {"no_action", "acid_low_dose"} and rng.random() < 0.45:
        label = "acid_low_dose" if label == "no_action" else "no_action"
    elif 7.45 <= ph <= 7.65 and label in {"acid_low_dose", "acid_medium_dose"} and rng.random() < 0.42:
        label = "acid_medium_dose" if label == "acid_low_dose" else "acid_low_dose"
    elif 6.15 <= ph <= 6.32 and label in {"base_low_dose", "base_medium_dose"} and rng.random() < 0.42:
        label = "base_medium_dose" if label == "base_low_dose" else "base_low_dose"
    elif (ammonia > 0.85 or nitrite > 0.85) and label != "manual_check" and rng.random() < 0.35:
        label = "manual_check"

    if is_dosing_boundary(row) and not is_extreme(row) and not has_manual_safety_signature(row) and rng.random() < 0.10:
        nearby = {
            "no_action": ["acid_low_dose", "base_low_dose"],
            "acid_low_dose": ["no_action", "acid_medium_dose"],
            "acid_medium_dose": ["acid_low_dose"],
            "base_low_dose": ["no_action", "base_medium_dose"],
            "base_medium_dose": ["base_low_dose"],
            "manual_check": ["acid_medium_dose", "base_medium_dose"],
        }
        label = str(rng.choice(nearby[label]))
    if label != "manual_check" and not has_manual_safety_signature(row) and not is_extreme(row) and rng.random() < 0.03:
        realistic_neighbors = {
            "no_action": ["acid_low_dose", "base_low_dose"],
            "acid_low_dose": ["no_action", "acid_medium_dose"],
            "acid_medium_dose": ["acid_low_dose"],
            "base_low_dose": ["no_action", "base_medium_dose"],
            "base_medium_dose": ["base_low_dose"],
        }
        label = str(rng.choice(realistic_neighbors.get(label, [label])))
    return label


def pump_duration(action: str, rng: np.random.Generator) -> int:
    """Create a bounded pump duration for the v3 dosing recommendation."""
    if action in {"no_action", "manual_check"}:
        return 0
    if action in {"acid_low_dose", "base_low_dose"}:
        return int(rng.integers(1, 4))
    return int(rng.integers(3, 6))


def simulate_ph_after(row: dict[str, object], rng: np.random.Generator) -> float:
    """Simulate noisy post-dosing pH with variable pump and process response."""
    ph = float(row["ph_before"])
    action = str(row["dosing_action"])
    response = float(row["pump_response_factor"])
    safety = str(row["safety_status"])

    if safety == "safety_fail" or action == "manual_check":
        change = rng.normal(0, 0.06)
    elif action == "no_action":
        change = rng.normal(0, 0.05)
    elif action == "acid_low_dose":
        change = -float(rng.choice([rng.uniform(0.02, 0.10), rng.uniform(0.10, 0.25)], p=[0.30, 0.70])) * response
    elif action == "acid_medium_dose":
        change = -float(rng.choice([rng.uniform(0.08, 0.22), rng.uniform(0.25, 0.55), rng.uniform(0.55, 0.85)], p=[0.25, 0.55, 0.20])) * response
    elif action == "base_low_dose":
        change = float(rng.choice([rng.uniform(0.02, 0.10), rng.uniform(0.10, 0.25)], p=[0.30, 0.70])) * response
    elif action == "base_medium_dose":
        change = float(rng.choice([rng.uniform(0.08, 0.22), rng.uniform(0.25, 0.55), rng.uniform(0.55, 0.85)], p=[0.25, 0.55, 0.20])) * response
    else:
        change = rng.normal(0, 0.06)

    return clip_round(ph + change + rng.normal(0, 0.04), 4.5, 9.5, 2)


def distance_to_target(ph: float) -> float:
    """Distance from pH value to target band 6.5 to 7.2."""
    if 6.5 <= ph <= 7.2:
        return 0.0
    return 6.5 - ph if ph < 6.5 else ph - 7.2


def dosing_success(row: dict[str, object]) -> bool:
    """Evaluate whether post-dosing response moved toward the target pH band."""
    action = str(row["dosing_action"])
    if str(row["safety_status"]) == "safety_fail" or action == "manual_check":
        return False
    before = float(row["ph_before"])
    after = float(row["ph_after"])
    delta = float(row["delta_ph"])
    if abs(delta) > 0.85:
        return False
    if action == "no_action":
        return 6.4 <= after <= 7.3
    if action.startswith("acid"):
        return after < before and distance_to_target(after) < distance_to_target(before)
    if action.startswith("base"):
        return after > before and distance_to_target(after) < distance_to_target(before)
    return False


def overdosing_risk(row: dict[str, object]) -> str:
    """Estimate overdosing risk from noisy process response."""
    delta = abs(float(row["delta_ph"]))
    action = str(row["dosing_action"])
    response = float(row["pump_response_factor"])
    if delta > 0.75 or response > 1.22 or (action.endswith("medium_dose") and delta > 0.58):
        return "tinggi"
    if delta > 0.35 or action.endswith("medium_dose") or response < 0.55:
        return "sedang"
    return "rendah"


def dominant_feature(row: dict[str, object]) -> str:
    """Pick a compact dominant feature indicator for dashboard analysis."""
    if str(row["sensor_status"]) != "valid":
        return "sensor_status"
    if float(row["water_level_pct"]) < 62:
        return "water_level_pct"
    if float(row["ph"]) < 6.45 or float(row["ph"]) > 7.25:
        return "ph"
    if float(row["ammonia_ppm"]) > 0.35:
        return "ammonia_ppm"
    if float(row["nitrite_ppm"]) > 0.20:
        return "nitrite_ppm"
    if float(row["nitrate_ppm"]) > 110:
        return "nitrate_ppm"
    if float(row["temperature_c"]) < 24 or float(row["temperature_c"]) > 31:
        return "temperature_c"
    return "ph"


def dashboard_alert(row: dict[str, object]) -> str:
    """Create concise dashboard alert text."""
    if str(row["safety_status"]) == "safety_fail" and str(row["dosing_action"]) != "no_action":
        return "Safety rule gagal, pompa diblokir"
    if str(row["sensor_status"]) == "perlu_kalibrasi":
        return "Sensor perlu kalibrasi"
    if float(row["water_level_pct"]) < 55:
        return "Level air rendah"
    if float(row["ammonia_ppm"]) > 0.55 or float(row["nitrite_ppm"]) > 0.55:
        return "Risiko nitrogen tinggi"
    return {
        "no_action": "Kondisi normal",
        "base_low_dose": "pH rendah ringan, disarankan base low dose",
        "base_medium_dose": "pH rendah sedang, disarankan base medium dose",
        "acid_low_dose": "pH tinggi ringan, disarankan acid low dose",
        "acid_medium_dose": "pH tinggi sedang, disarankan acid medium dose",
        "manual_check": "Manual check diperlukan",
    }.get(str(row["dosing_action"]), "Manual check diperlukan")


def make_record(index: int, scenario: str, timestamp: pd.Timestamp, rng: np.random.Generator) -> dict[str, object]:
    """Create one complete v3.1 Bogor realistic balanced synthetic record."""
    config = load_config()
    progress = index / max(N_ROWS - 1, 1)
    latent = base_true_values(scenario, rng, progress)
    noisy = apply_sensor_noise(latent, rng)

    if scenario == "normal_strong":
        confidence = float(np.clip(rng.normal(0.91, 0.04), 0.78, 0.99))
    elif scenario == "manual_safety_signature":
        confidence = float(np.clip(rng.normal(0.57, 0.07), 0.40, 0.72))
    else:
        confidence = float(np.clip(rng.normal(0.84, 0.10), 0.45, 0.99))
    if latent["rain_event"]:
        confidence -= rng.uniform(0.02, 0.08)
    if latent["sensor_status"] != "valid":
        confidence -= rng.uniform(0.08, 0.18)

    record: dict[str, object] = {
        "data_id": f"AQD-V3-{index + 1:06d}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": DATASET_VERSION,
        "input_source": "synthetic_realistic_bogor_balanced",
        "scenario_type": scenario,
        "sensor_id": f"BOGOR-AQ-{int(rng.integers(1, 9)):02d}",
        **noisy,
        "sensor_status": latent["sensor_status"],
        "ph_before": noisy["ph"],
        "time_since_last_dosing_min": int(
            rng.choice([rng.integers(0, 10), rng.integers(10, 220)], p=[0.34, 0.66])
            if scenario == "manual_safety_signature"
            else rng.choice([rng.integers(0, 10), rng.integers(10, 220)], p=[0.04, 0.96])
            if scenario == "normal_strong"
            else rng.choice([rng.integers(0, 10), rng.integers(10, 220)], p=[0.08, 0.92])
        ),
        "dosing_cycle_count": int(
            rng.choice([0, 1, 2, 3], p=[0.28, 0.20, 0.14, 0.38])
            if scenario == "manual_safety_signature"
            else rng.choice([0, 1, 2, 3], p=[0.60, 0.28, 0.10, 0.02])
            if scenario == "normal_strong"
            else rng.choice([0, 1, 2, 3], p=[0.48, 0.29, 0.18, 0.05])
        ),
        "mixing_time_min": int(rng.integers(5, 11)),
        "confidence_score": round(float(np.clip(confidence, 0.35, 0.99)), 3),
        "max_pump_duration_sec": int(config.get("max_pump_duration_sec", 5)),
        "cooldown_time_min": int(config.get("cooldown_time_min", 10)),
        "rain_event": int(latent["rain_event"]),
        "recent_feeding_level": latent["recent_feeding_level"],
        "biofilter_maturity": latent["biofilter_maturity"],
        "water_turbidity_ntu": latent["water_turbidity_ntu"],
        "pump_response_factor": latent["pump_response_factor"],
        "operator_measurement_error": latent["operator_measurement_error"],
    }

    record["water_quality_status"] = water_quality_label(record, rng)
    record["recommended_action"] = dosing_label(record, rng)
    record["dosing_action"] = record["recommended_action"]
    record["dosing_type"] = dosing_type_from_action(str(record["dosing_action"]))
    record["pump_duration_sec"] = pump_duration(str(record["dosing_action"]), rng)
    record.update(apply_safety_rule(record))
    record["ph_after"] = simulate_ph_after(record, rng)
    record["delta_ph"] = round(float(record["ph_after"]) - float(record["ph_before"]), 2)
    record["dosing_success"] = dosing_success(record)
    record["overdosing_risk"] = overdosing_risk(record)
    record["manual_override"] = bool(str(record["dosing_action"]) == "manual_check" and rng.random() < 0.42)
    record["dominant_feature"] = dominant_feature(record)
    record["dashboard_alert"] = dashboard_alert(record)
    return record


def add_distribution_shift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add a mild time drift so time-based split is more realistic than random split."""
    split_index = int(len(df) * 0.80)
    test_idx = df.index[split_index:]
    drift_mask = rng.random(len(test_idx)) < 0.38
    affected = test_idx[drift_mask]
    df.loc[affected, "rain_event"] = 1
    df.loc[affected, "ph"] = (df.loc[affected, "ph"] - rng.uniform(0.03, 0.14, len(affected))).clip(4.5, 9.5).round(2)
    df.loc[affected, "water_level_pct"] = (df.loc[affected, "water_level_pct"] + rng.uniform(1, 6, len(affected))).clip(0, 100).round(1)

    # Recompute labels after the simulated end-period rain drift.
    for idx in affected:
        row = df.loc[idx].to_dict()
        df.at[idx, "water_quality_status"] = water_quality_label(row, rng)
        df.at[idx, "recommended_action"] = dosing_label(row, rng)
        df.at[idx, "dosing_action"] = df.at[idx, "recommended_action"]
        df.at[idx, "dosing_type"] = dosing_type_from_action(str(df.at[idx, "dosing_action"]))
        df.at[idx, "pump_duration_sec"] = pump_duration(str(df.at[idx, "dosing_action"]), rng)
        safety = apply_safety_rule(df.loc[idx].to_dict())
        for key, value in safety.items():
            df.at[idx, key] = value
        df.at[idx, "ph_before"] = df.at[idx, "ph"]
        df.at[idx, "ph_after"] = simulate_ph_after(df.loc[idx].to_dict(), rng)
        df.at[idx, "delta_ph"] = round(float(df.at[idx, "ph_after"]) - float(df.at[idx, "ph_before"]), 2)
        df.at[idx, "dosing_success"] = dosing_success(df.loc[idx].to_dict())
        df.at[idx, "overdosing_risk"] = overdosing_risk(df.loc[idx].to_dict())
        df.at[idx, "dominant_feature"] = dominant_feature(df.loc[idx].to_dict())
        df.at[idx, "dashboard_alert"] = dashboard_alert(df.loc[idx].to_dict())
    return df


def generate_dataset() -> pd.DataFrame:
    """Generate v3 synthetic-realistic Bogor aquaponic dataset and time split."""
    ensure_project_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    timestamps = pd.date_range("2026-03-01 00:00:00", periods=N_ROWS, freq="15min")
    scenarios = build_scenarios(N_ROWS, rng)
    records = [make_record(i, str(scenarios[i]), timestamps[i], rng) for i in range(N_ROWS)]
    df = pd.DataFrame(records)
    df = add_distribution_shift(df, rng)
    df = df[OUTPUT_COLUMNS]

    split_index = int(len(df) * 0.80)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    df.to_csv(OUTPUT_DATASET, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    return df


def main() -> None:
    df = generate_dataset()
    print("Generated v3.1 Bogor realistic balanced dataset.")
    print(f"Dataset: {OUTPUT_DATASET}")
    print(f"Rows: {len(df):,}")
    print("\nwater_quality_status:")
    print(df["water_quality_status"].value_counts().to_string())
    print("\ndosing_action:")
    print(df["dosing_action"].value_counts().to_string())
    print(f"\nTrain: {TRAIN_PATH}")
    print(f"Test: {TEST_PATH}")


if __name__ == "__main__":
    main()
