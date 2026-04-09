from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = REPO / "data" / "derived"
RAW_DIR = REPO / "data" / "raw"

EMAD_COLUMNS_PATH = DATA_DIR / "benford_emad_columns.json"
EMAD_STATES_PATH = DATA_DIR / "benford_emad_states.json"
MISSINGNESS_PATH = DATA_DIR / "missingness_mechanism.json"
LEGACY_SUMMARY_PATH = DATA_DIR / "benford_analysis_summary.json"
SUPPORTING_DOCS_PATH = RAW_DIR / "esser-2023-supporting-documentation.xlsx"

TOTAL_ROWS = 17_705

MAD_GOOD = 0.005
MAD_MARGINAL = 0.015
MIN_N_BENFORD = 1000

RISK_READY = 0.05
RISK_CAUTION = 0.20
AUC_RANDOM = 0.65

BUCKETS = {
    "ready": {
        "label": "Ready to use",
        "color": "#2ca25f",
        "icon": "[OK]",
        "blurb": "Clean Benford fit and missingness is approximately random.",
    },
    "caution": {
        "label": "Use with caution",
        "color": "#f0a500",
        "icon": "[!]",
        "blurb": "Marginal Benford or some missingness bias. Usable with care.",
    },
    "investigate": {
        "label": "Investigate first",
        "color": "#d7191c",
        "icon": "[X]",
        "blurb": "Bad Benford fit OR strongly biased missingness pattern.",
    },
}
BUCKET_ORDER = ["ready", "caution", "investigate"]


def _read_json_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_description_map() -> dict[str, str]:
    """Best-effort description lookup for EMAD columns.

    Priority:
    1. Legacy summary JSON, if it has a description for the column.
    2. Supporting documentation workbook, if available.
    """
    mapping: dict[str, str] = {}

    legacy = _read_json_frame(LEGACY_SUMMARY_PATH)
    if not legacy.empty and {"column", "description"}.issubset(legacy.columns):
        legacy["description"] = legacy["description"].fillna("").astype(str).str.strip()
        for row in legacy.itertuples(index=False):
            if getattr(row, "description", ""):
                mapping[str(row.column)] = str(row.description)

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
                desc = str(row["Description"]).strip()
                if key and desc:
                    mapping.setdefault(key, desc)

    return mapping


def infer_topic(col: str) -> str:
    c = col.lower()
    if "tutor" in c:
        return "Tutoring"
    if "aftsch" in c:
        return "After-school"
    if "summer" in c:
        return "Summer learning"
    if "mental" in c:
        return "Mental health"
    if "academic" in c:
        return "Academic recovery"
    if "physical" in c or "property" in c or "building" in c:
        return "Facilities"
    if "operational" in c:
        return "Operations"
    if "personnel" in c or "salaries" in c or "benefits" in c or "staff" in c:
        return "Personnel"
    if "enrollment" in c:
        return "Enrollment / demographics"
    if "subgrant" in c:
        return "Subgrants"
    if "sea" in c and "reserve" in c:
        return "SEA Reserve"
    if "esser1" in c:
        return "ESSER I (general)"
    if "esser2" in c:
        return "ESSER II (general)"
    if "esser3" in c:
        return "ESSER III (general)"
    return "Metadata / other"


def compute_bias_risk(missingness_auc: float | None, pct_missing: float | None) -> float:
    if (
        missingness_auc is None
        or pd.isna(missingness_auc)
        or pct_missing is None
        or pct_missing < 0.5
    ):
        return 0.0
    auc_excess = max(0.0, float(missingness_auc) - 0.5) * 2.0
    return float(auc_excess * (pct_missing / 100.0))


def compute_score(excess_mad: float, bias_risk: float) -> float:
    fit = max(0.0, 1.0 - (excess_mad / MAD_MARGINAL))
    safety = max(0.0, 1.0 - (bias_risk / 0.5))
    return 100.0 * (0.5 * fit + 0.5 * safety)


def assign_bucket(num_valid: int, excess_mad: float, bias_risk: float) -> str:
    if num_valid < MIN_N_BENFORD:
        return "investigate"
    if excess_mad > MAD_MARGINAL:
        return "investigate"
    if excess_mad <= MAD_GOOD:
        if bias_risk < RISK_READY:
            return "ready"
        if bias_risk < RISK_CAUTION:
            return "caution"
        return "investigate"
    if bias_risk < RISK_READY * 2:
        return "caution"
    return "investigate"


def normalize_emad_columns(include_missingness: bool = True) -> pd.DataFrame:
    df = _read_json_frame(EMAD_COLUMNS_PATH)
    if df.empty:
        return df

    rename_map = {
        "label": "column",
        "N": "num_valid_entries",
        "chi2": "chi_squared",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "mad" in df.columns and "mad_score" not in df.columns:
        df["mad_score"] = df["mad"]
    if "euclidean_distance" not in df.columns:
        obs = pd.DataFrame(df["observed_proportions"].tolist())
        exp = pd.DataFrame(df["expected_proportions"].tolist())
        df["euclidean_distance"] = np.sqrt(((obs - exp) ** 2).sum(axis=1))

    if include_missingness and MISSINGNESS_PATH.exists():
        miss = _read_json_frame(MISSINGNESS_PATH)
        if not miss.empty and "profile" not in miss.columns:
            miss["profile"] = None
        if not miss.empty:
            keep = [c for c in ["column", "missingness_auc", "auc_std", "n_used", "n_missing", "pct_missing", "note", "profile"] if c in miss.columns]
            df = df.merge(miss[keep], on="column", how="left")
    if "missingness_auc" not in df.columns:
        df["missingness_auc"] = None
    if "pct_missing" not in df.columns:
        df["pct_missing"] = None
    if "profile" not in df.columns:
        df["profile"] = None

    desc_map = load_description_map()
    if "description" not in df.columns:
        df["description"] = None
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["description"] = df.apply(
        lambda r: r["description"]
        if r["description"] and r["description"].lower() != "nan"
        else desc_map.get(str(r["column"]), ""),
        axis=1,
    )

    df["coverage"] = df["num_valid_entries"] / TOTAL_ROWS
    df["missingness_pct"] = 100 * (1 - df["coverage"])
    df["bias_risk"] = df.apply(
        lambda r: compute_bias_risk(r["missingness_auc"], r["pct_missing"]),
        axis=1,
    )
    df["quality_score"] = df.apply(
        lambda r: compute_score(float(r["excess_mad"]), float(r["bias_risk"])),
        axis=1,
    )
    df["bucket"] = df.apply(
        lambda r: assign_bucket(
            int(r["num_valid_entries"]),
            float(r["excess_mad"]),
            float(r["bias_risk"]),
        ),
        axis=1,
    )
    df["topic"] = df["column"].apply(infer_topic)
    df["display_name"] = (
        df["description"].fillna("").astype(str).str.strip().replace("", np.nan).fillna(df["column"])
    )
    return df


def normalize_emad_states() -> pd.DataFrame:
    df = _read_json_frame(EMAD_STATES_PATH)
    if df.empty:
        return df
    df = df.rename(columns={"label": "state", "N": "num_valid_entries", "chi2": "chi_squared"})
    if "reporting_pct" in df.columns:
        max_reporting = df["reporting_pct"].dropna().max()
        if pd.notna(max_reporting) and float(max_reporting) <= 1.5:
            df["reporting_pct_frac"] = df["reporting_pct"]
            df["reporting_pct"] = df["reporting_pct"] * 100.0
    else:
        df["reporting_pct"] = np.nan
        df["reporting_pct_frac"] = np.nan
    if "reporting_pct_frac" not in df.columns:
        df["reporting_pct_frac"] = df["reporting_pct"] / 100.0
    return df


def format_state_label(state: str, description_map: dict[str, str] | None = None) -> str:
    return state if not description_map else description_map.get(state, state)
