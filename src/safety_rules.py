from __future__ import annotations

from typing import Any


def apply_safety_rule(row: dict[str, Any]) -> dict[str, str]:
    """Apply the final safety controller before any pump activation."""
    dosing_action = row["dosing_action"]
    sensor_status = row["sensor_status"]
    water_level_pct = float(row["water_level_pct"])
    pump_duration_sec = float(row["pump_duration_sec"])
    max_pump_duration_sec = float(row["max_pump_duration_sec"])
    time_since_last_dosing_min = float(row["time_since_last_dosing_min"])
    cooldown_time_min = float(row["cooldown_time_min"])
    dosing_cycle_count = int(row["dosing_cycle_count"])
    confidence_score = float(row["confidence_score"])

    safety_status = "safety_pass"
    safety_reason = "safety_pass_all_checks"

    if sensor_status != "valid":
        safety_status = "safety_fail"
        safety_reason = "sensor_error"
    elif water_level_pct < 50:
        safety_status = "safety_fail"
        safety_reason = "water_level_low"
    elif pump_duration_sec > max_pump_duration_sec:
        safety_status = "safety_fail"
        safety_reason = "max_pump_duration_exceeded"
    elif time_since_last_dosing_min < cooldown_time_min:
        safety_status = "safety_fail"
        safety_reason = "cooldown_not_met"
    elif dosing_cycle_count >= 3:
        safety_status = "safety_fail"
        safety_reason = "max_cycle_exceeded"
    elif confidence_score < 0.70:
        safety_status = "safety_fail"
        safety_reason = "low_confidence"
    elif dosing_action == "manual_check":
        safety_status = "safety_fail"
        safety_reason = "manual_check_required"
    elif dosing_action == "no_action":
        safety_status = "safety_pass"
        safety_reason = "no_action_required"

    if dosing_action == "no_action":
        pump_status = "off"
    elif safety_status == "safety_pass":
        pump_status = "on"
    else:
        pump_status = "blocked"

    return {
        "safety_status": safety_status,
        "safety_reason": safety_reason,
        "pump_status": pump_status,
    }
