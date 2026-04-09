from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DERIVED = HERE.parent / "data" / "derived"
RAW = HERE.parent / "data" / "raw"

EMAD_COLUMNS_PATH = DERIVED / "benford_emad_columns.json"
EMAD_STATES_PATH = DERIVED / "benford_emad_states.json"
EMAD_AGGREGATE_PATH = DERIVED / "benford_emad_aggregate.json"
LEGACY_SUMMARY_PATH = DERIVED / "benford_analysis_summary.json"
MISSINGNESS_PATH = DERIVED / "missingness_mechanism.json"
SUPPORTING_DOCS_PATH = RAW / "esser-2023-supporting-documentation.xlsx"

TOTAL_ROWS = 17_705

ROMAN_NUMERALS = {
    "1": "I",
    "2": "II",
    "3": "III",
}

ACRONYMS = {
    "arp": "ARP",
    "auc": "AUC",
    "duns": "DUNS",
    "ed": "ED",
    "emad": "EMAD",
    "esser": "ESSER",
    "fte": "FTE",
    "id": "ID",
    "lea": "LEA",
    "mad": "MAD",
    "mcar": "MCAR",
    "nces": "NCES",
    "pct": "%",
    "sea": "SEA",
    "usd": "USD",
}

WORD_REPLACEMENTS = {
    "aft": "After",
    "sch": "School",
    "curr": "Current",
    "expended": "Expended",
    "expendedtotal": "Expended Total",
    "his": "Hispanic",
    "hmls": "Homeless",
    "int": "Intervention",
    "li": "Low Income",
    "mand": "Mandatory",
    "nhpi": "Native Hawaiian / Pacific Islander",
    "swd": "Students With Disabilities",
    "tmr": "Two or More Races",
    "tutoring": "Tutoring",
    "wh": "White",
}


def _read_json(path: Path) -> list[dict] | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _description_map() -> dict[str, str]:
    descriptions: dict[str, str] = {}

    if LEGACY_SUMMARY_PATH.exists():
        for row in _read_json(LEGACY_SUMMARY_PATH):
            column = row.get("column")
            description = str(row.get("description", "")).strip()
            if column and description and description != "No description available":
                descriptions[column] = description

    if SUPPORTING_DOCS_PATH.exists():
        try:
            docs = pd.read_excel(
                SUPPORTING_DOCS_PATH,
                usecols=["Variable Name", "Description"],
            )
        except Exception:
            docs = None
        if docs is not None:
            docs = docs.fillna("")
            for _, row in docs.iterrows():
                key = str(row["Variable Name"]).strip()
                description = str(row["Description"]).strip()
                if key and description:
                    descriptions.setdefault(key, description)

    return descriptions


def humanize_identifier(identifier: str) -> str:
    text = identifier.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", text)
    text = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    parts: list[str] = []
    for part in text.split():
        lower = part.lower()
        if lower == "esser" and parts:
            parts.append("ESSER")
            continue
        if lower in ACRONYMS:
            parts.append(ACRONYMS[lower])
            continue
        if lower in WORD_REPLACEMENTS:
            parts.extend(WORD_REPLACEMENTS[lower].split())
            continue
        parts.append(part.title())

    label = " ".join(parts)
    for arabic, roman in ROMAN_NUMERALS.items():
        label = re.sub(rf"\bESSER {arabic}\b", f"ESSER {roman}", label)
    label = re.sub(r"\b20\b", "20%", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def load_column_catalog() -> pd.DataFrame:
    df = pd.DataFrame(_read_json(EMAD_COLUMNS_PATH))
    if "label" in df.columns and "column" not in df.columns:
        df = df.rename(columns={"label": "column"})

    descriptions = _description_map()
    if "description" not in df.columns:
        df["description"] = ""

    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["description"] = df.apply(
        lambda row: row["description"] or descriptions.get(row["column"], ""),
        axis=1,
    )
    df["display_name"] = df.apply(
        lambda row: row["description"] or humanize_identifier(row["column"]),
        axis=1,
    )
    return df


def load_state_catalog() -> pd.DataFrame:
    df = pd.DataFrame(_read_json(EMAD_STATES_PATH))
    if "label" in df.columns and "state" not in df.columns:
        df = df.rename(columns={"label": "state"})
    return df


def load_aggregate_result() -> dict:
    if not EMAD_AGGREGATE_PATH.exists():
        return {}
    return _read_json(EMAD_AGGREGATE_PATH)


def load_missingness_profiles() -> pd.DataFrame:
    if not MISSINGNESS_PATH.exists():
        return pd.DataFrame()
    return pd.DataFrame(_read_json(MISSINGNESS_PATH))
