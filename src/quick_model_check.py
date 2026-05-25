from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

from utils import PROJECT_ROOT, ensure_project_dirs, load_config


WATER_QUALITY_FEATURES = [
    "ph",
    "temperature_c",
    "water_level_pct",
    "ammonia_ppm",
    "nitrite_ppm",
    "nitrate_ppm",
    "time_since_last_dosing_min",
    "dosing_cycle_count",
]

DOSING_ACTION_FEATURES = [
    "ph",
    "temperature_c",
    "water_level_pct",
    "ammonia_ppm",
    "nitrite_ppm",
    "nitrate_ppm",
    "time_since_last_dosing_min",
    "dosing_cycle_count",
    "confidence_score",
]


def get_models(random_seed: int) -> dict[str, object]:
    """Return baseline supervised classifiers for quick dataset usability checks."""
    return {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_seed, max_depth=8),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=60,
            random_state=random_seed,
            max_depth=9,
            n_jobs=1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.08,
            max_depth=3,
            random_state=random_seed,
        ),
    }


def confusion_matrix_markdown(y_true: pd.Series, y_pred, labels: list[str]) -> str:
    """Format a confusion matrix as a markdown table."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    separator = "|---" + "|---:" * len(labels) + "|"
    rows = [header, separator]
    for label, values in zip(labels, matrix):
        rows.append("| " + label + " | " + " | ".join(str(int(value)) for value in values) + " |")
    return "\n".join(rows)


def evaluate_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    features: list[str],
    random_seed: int,
) -> list[str]:
    """Train and evaluate baseline models for one supervised target."""
    labels = sorted(train_df[target].unique().tolist())
    x_train = train_df[features]
    y_train = train_df[target]
    x_test = test_df[features]
    y_test = test_df[target]

    lines = [
        f"## Target: {target}",
        "",
        f"Fitur: {', '.join(features)}",
        "",
    ]

    for model_name, model in get_models(random_seed).items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        lines.extend(
            [
                f"### {model_name}",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| accuracy | {accuracy:.4f} |",
                f"| precision_macro | {precision_macro:.4f} |",
                f"| recall_macro | {recall_macro:.4f} |",
                f"| f1_macro | {f1_macro:.4f} |",
                "",
                "Classification report:",
                "",
                "```text",
                classification_report(y_test, y_pred, labels=labels, zero_division=0),
                "```",
                "",
                "Confusion matrix:",
                "",
                confusion_matrix_markdown(y_test, y_pred, labels),
                "",
            ]
        )

    return lines


def main() -> None:
    config = load_config()
    ensure_project_dirs()
    train_path = PROJECT_ROOT / "data" / "aquaponic_train.csv"
    test_path = PROJECT_ROOT / "data" / "aquaponic_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train/test CSV files are missing. Run python src/generate_dataset.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    random_seed = int(config["random_seed"])

    report_lines = [
        "# Quick Model Result",
        "",
        "Baseline ini hanya memeriksa bahwa dataset dapat digunakan untuk supervised classification. Kolom hasil proses seperti ph_after, delta_ph, dosing_success, safety_status, pump_status, dan dashboard_alert sengaja tidak dipakai sebagai fitur utama untuk menghindari data leakage.",
        "",
        f"- Train rows: {len(train_df):,}",
        f"- Test rows: {len(test_df):,}",
        "",
    ]

    report_lines.extend(
        evaluate_target(
            train_df,
            test_df,
            "water_quality_status",
            WATER_QUALITY_FEATURES,
            random_seed,
        )
    )
    report_lines.extend(
        evaluate_target(
            train_df,
            test_df,
            "dosing_action",
            DOSING_ACTION_FEATURES,
            random_seed,
        )
    )

    report_path = PROJECT_ROOT / "reports" / "quick_model_result.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("Quick model check completed.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
