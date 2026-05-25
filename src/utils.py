from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_config.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration used by all dataset scripts."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_project_dirs() -> None:
    """Create expected output directories if they do not already exist."""
    for folder in ("data", "reports", "figures"):
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)


def format_percentage(count: int, total: int) -> str:
    """Return a compact percentage string for markdown reports."""
    if total == 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def markdown_table_from_series(series, name: str = "count") -> str:
    """Convert a pandas Series into a small markdown table."""
    total = int(series.sum()) if len(series) else 0
    lines = [f"| class | {name} | percentage |", "|---|---:|---:|"]
    for label, count in series.items():
        lines.append(f"| {label} | {int(count)} | {format_percentage(int(count), total)} |")
    return "\n".join(lines)


def markdown_table_from_dataframe(df) -> str:
    """Convert a small pandas DataFrame into a markdown table without extra dependencies."""
    if df.empty:
        return "_No rows._"
    display_df = df.reset_index()
    headers = [str(column) for column in display_df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)
