from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from generate_dataset import MANDATORY_COLUMNS
from utils import (
    PROJECT_ROOT,
    ensure_project_dirs,
    load_config,
    markdown_table_from_dataframe,
    markdown_table_from_series,
)


def collect_validation_results(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Run rule and schema checks over the generated dataset."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [column for column in MANDATORY_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing mandatory columns: {', '.join(missing_columns)}")

    if missing_columns:
        return errors, warnings

    missing_counts = df[MANDATORY_COLUMNS].isna().sum()
    missing_with_values = missing_counts[missing_counts > 0]
    if not missing_with_values.empty:
        errors.append(f"Missing values found: {missing_with_values.to_dict()}")

    range_checks = {
        "ph": df["ph"].between(4.5, 9.5),
        "temperature_c": df["temperature_c"].between(15, 40),
        "water_level_pct": df["water_level_pct"].between(0, 100),
        "ammonia_ppm": df["ammonia_ppm"] >= 0,
        "nitrite_ppm": df["nitrite_ppm"] >= 0,
        "nitrate_ppm": df["nitrate_ppm"] >= 0,
    }
    for column, valid_mask in range_checks.items():
        invalid_count = int((~valid_mask).sum())
        if invalid_count:
            errors.append(f"{column} has {invalid_count} out-of-range values.")

    duration_invalid = int((df["pump_duration_sec"] > df["max_pump_duration_sec"]).sum())
    if duration_invalid:
        errors.append(f"pump_duration_sec exceeds max_pump_duration_sec in {duration_invalid} rows.")

    safety_invalid = int(((df["safety_status"] == "safety_fail") & (df["pump_status"] == "on")).sum())
    if safety_invalid:
        errors.append(f"safety_fail rows have pump_status == on in {safety_invalid} rows.")

    no_action_duration_invalid = int(
        ((df["dosing_action"] == "no_action") & (df["pump_duration_sec"] != 0)).sum()
    )
    if no_action_duration_invalid:
        errors.append(f"no_action rows have pump_duration_sec != 0 in {no_action_duration_invalid} rows.")

    manual_duration_invalid = int(
        ((df["dosing_action"] == "manual_check") & (df["pump_duration_sec"] != 0)).sum()
    )
    if manual_duration_invalid:
        errors.append(f"manual_check rows have pump_duration_sec != 0 in {manual_duration_invalid} rows.")

    delta_invalid = int(
        (~np.isclose(df["delta_ph"], df["ph_after"] - df["ph_before"], atol=0.011)).sum()
    )
    if delta_invalid:
        errors.append(f"delta_ph is inconsistent with ph_after - ph_before in {delta_invalid} rows.")

    water_quality_counts = df["water_quality_status"].value_counts()
    dosing_action_counts = df["dosing_action"].value_counts()
    if len(water_quality_counts) < 4:
        warnings.append("Not all water_quality_status classes are represented.")
    if len(dosing_action_counts) < 6:
        warnings.append("Not all dosing_action classes are represented.")

    return errors, warnings


def range_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create min/max range summary for core numeric parameters."""
    columns = [
        "ph",
        "temperature_c",
        "water_level_pct",
        "ammonia_ppm",
        "nitrite_ppm",
        "nitrate_ppm",
        "pump_duration_sec",
        "ph_after",
        "delta_ph",
    ]
    return df[columns].agg(["min", "max", "mean", "std"]).T.round(3)


def write_quality_report(df: pd.DataFrame, errors: list[str], warnings: list[str]) -> None:
    """Write the academic-style dataset quality report."""
    ensure_project_dirs()
    report_path = PROJECT_ROOT / "reports" / "dataset_quality_report.md"
    missing_counts = df.isna().sum()
    missing_nonzero = missing_counts[missing_counts > 0]
    range_df = range_summary(df)

    report = [
        "# Dataset Quality Report",
        "",
        "## 1. Ringkasan Dataset",
        f"- Jumlah baris: {len(df):,}",
        f"- Jumlah kolom: {len(df.columns):,}",
        f"- Periode awal timestamp: {df['timestamp'].min()}",
        f"- Periode akhir timestamp: {df['timestamp'].max()}",
        "- Sumber data: synthetic",
        "",
        "## 2. Ringkasan Statistik Numerik",
        markdown_table_from_dataframe(range_df),
        "",
        "## 3. Distribusi water_quality_status",
        markdown_table_from_series(df["water_quality_status"].value_counts(), "count"),
        "",
        "## 4. Distribusi dosing_action",
        markdown_table_from_series(df["dosing_action"].value_counts(), "count"),
        "",
        "## 5. Jumlah Data per scenario_type",
        markdown_table_from_series(df["scenario_type"].value_counts(), "count"),
        "",
        "## 6. Pemeriksaan Missing Value",
    ]

    if missing_nonzero.empty:
        report.append("Tidak ada missing value pada dataset.")
    else:
        report.append(markdown_table_from_dataframe(missing_nonzero.to_frame("missing_count")))

    report.extend(
        [
            "",
            "## 7. Pemeriksaan Range Parameter",
            "- pH berada pada batas validasi 4.5 sampai 9.5.",
            "- temperature_c berada pada batas validasi 15 sampai 40 derajat Celsius.",
            "- water_level_pct berada pada batas validasi 0 sampai 100 persen.",
            "- ammonia_ppm, nitrite_ppm, dan nitrate_ppm tidak bernilai negatif.",
            "- pump_duration_sec tidak melebihi max_pump_duration_sec.",
            "",
            "## 8. Pemeriksaan Konsistensi Safety Rule",
            "- Baris safety_fail tidak mengaktifkan pompa.",
            "- no_action dan manual_check selalu memiliki pump_duration_sec = 0.",
            "- delta_ph konsisten dengan ph_after - ph_before pada toleransi pembulatan 0.011.",
            "",
            "## 9. Status Validasi",
        ]
    )

    if errors:
        report.extend([f"- ERROR: {error}" for error in errors])
    else:
        report.append("- Tidak ditemukan error validasi.")

    if warnings:
        report.extend([f"- WARNING: {warning}" for warning in warnings])
    else:
        report.append("- Tidak ditemukan warning distribusi kelas utama.")

    report.extend(
        [
            "",
            "## 10. Catatan Keterbatasan Synthetic Dataset",
            "Dataset ini dibuat dari aturan dan simulasi terkontrol. Nilai sensor, parameter nitrogen, dan respons pH setelah dosing belum berasal dari pengujian lapangan jangka panjang. Karena itu, dataset cocok untuk pengembangan awal pipeline machine learning, eksplorasi fitur, dan demonstrasi ML-assisted automation, tetapi belum boleh dianggap mewakili seluruh dinamika biologis sistem aquaponic nyata.",
            "",
            "## 11. Rekomendasi Penggunaan Berikutnya",
            "- Gunakan dataset ini untuk baseline Water Quality Classifier dan Dosing Action Classifier.",
            "- Hindari penggunaan ph_after, delta_ph, dosing_success, safety_status, pump_status, dan dashboard_alert sebagai fitur model awal karena kolom tersebut adalah hasil keputusan atau hasil proses sistem.",
            "- Validasi ulang threshold dan distribusi data menggunakan data lapangan sebelum sistem digunakan untuk keputusan operasional nyata.",
            "- Pertahankan Safety Rule Controller sebagai lapisan wajib sebelum pompa fisik aktif.",
            "",
        ]
    )

    report_path.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    load_config()
    dataset_path = PROJECT_ROOT / "data" / "aquaponic_synthetic_dataset_v1.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    errors, warnings = collect_validation_results(df)
    write_quality_report(df, errors, warnings)

    if errors:
        print("Validation completed with errors.")
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)

    print("Validation passed.")
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")
    print(f"Report: {PROJECT_ROOT / 'reports' / 'dataset_quality_report.md'}")


if __name__ == "__main__":
    main()
