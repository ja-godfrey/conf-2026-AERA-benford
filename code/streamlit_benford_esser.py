"""
Canonical Streamlit triage app for the AERA 2026 ESSER audit.

This app is intentionally aligned to the corrected EMAD pipeline used by the
slide deck:

    data/derived/benford_emad_columns.json
    data/derived/benford_emad_states.json
    data/derived/benford_emad_aggregate.json

It also merges the missingness-mechanism diagnostics when available.

Run with:
    streamlit run code/streamlit_benford_triage.py
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_catalog import (
    TOTAL_ROWS,
    load_aggregate_result,
    load_column_catalog,
    load_missingness_profiles,
    load_state_catalog,
)

# ---------------------------------------------------------------------------
# Thresholds and labels
# ---------------------------------------------------------------------------

MAD_GOOD = 0.001
MAD_MARGINAL = 0.002
MIN_N_BENFORD = 500

RISK_READY = 0.05
RISK_CAUTION = 0.20
AUC_RANDOM = 0.65

BUCKETS = {
    "ready": {
        "label": "Ready",
        "color": "#2ca25f",
        "blurb": "Clean EMAD fit and low missingness bias risk.",
    },
    "caution": {
        "label": "Caution",
        "color": "#f0a500",
        "blurb": "Usable, but not without sensitivity checks.",
    },
    "investigate": {
        "label": "Investigate",
        "color": "#d7191c",
        "blurb": "High distortion, high bias risk, or too little signal.",
    },
}
BUCKET_ORDER = ["investigate", "caution", "ready"]
EMAD_NONCONFORMING = 0.002


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def infer_topic(column: str) -> str:
    value = column.lower()
    if "tutor" in value:
        return "Tutoring"
    if "aftsch" in value:
        return "After-school"
    if "summer" in value:
        return "Summer learning"
    if "mental" in value:
        return "Mental health"
    if "academic" in value:
        return "Academic recovery"
    if "physical" in value or "property" in value or "building" in value:
        return "Facilities"
    if "operational" in value:
        return "Operations"
    if "personnel" in value or "salaries" in value or "benefits" in value or "staff" in value:
        return "Personnel"
    if "enrollment" in value:
        return "Enrollment / demographics"
    if "subgrant" in value:
        return "Subgrants"
    if "sea" in value and "reserve" in value:
        return "SEA Reserve"
    if "esser1" in value:
        return "ESSER I (general)"
    if "esser2" in value:
        return "ESSER II (general)"
    if "esser3" in value:
        return "ESSER III (general)"
    return "Other"


def compute_bias_risk(missingness_auc: float | None, pct_missing: float | None) -> float:
    if missingness_auc is None or pd.isna(missingness_auc):
        return 0.0
    if pct_missing is None or pd.isna(pct_missing) or pct_missing < 0.5:
        return 0.0
    auc_excess = max(0.0, float(missingness_auc) - 0.5) * 2.0
    return float(auc_excess * (float(pct_missing) / 100.0))


def compute_score(excess_mad: float, bias_risk: float) -> float:
    fit = max(0.0, 1.0 - (float(excess_mad) / MAD_MARGINAL))
    safety = max(0.0, 1.0 - (float(bias_risk) / 0.5))
    return round(100.0 * (0.5 * fit + 0.5 * safety), 1)


def assign_bucket(num_valid: int, excess_mad: float, bias_risk: float) -> str:
    if int(num_valid) < MIN_N_BENFORD:
        return "investigate"
    if float(excess_mad) > MAD_MARGINAL:
        return "investigate"
    if float(excess_mad) <= MAD_GOOD:
        if bias_risk < RISK_READY:
            return "ready"
        if bias_risk < RISK_CAUTION:
            return "caution"
        return "investigate"
    if bias_risk < (RISK_READY * 2):
        return "caution"
    return "investigate"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = load_column_catalog()
    missingness = load_missingness_profiles()

    if not missingness.empty:
        profile_cols = [c for c in ["column", "missingness_auc", "pct_missing", "profile"] if c in missingness.columns]
        df = df.merge(missingness[profile_cols], on="column", how="left")
    else:
        df["missingness_auc"] = np.nan
        df["pct_missing"] = np.nan
        df["profile"] = None

    df["reporting_pct"] = 100 * df["N"] / TOTAL_ROWS
    df["dataset_missing_pct"] = 100 - df["reporting_pct"]
    df["bias_risk"] = df.apply(
        lambda row: compute_bias_risk(row.get("missingness_auc"), row.get("pct_missing")),
        axis=1,
    )
    df["quality_score"] = df.apply(
        lambda row: compute_score(row["excess_mad"], row["bias_risk"]),
        axis=1,
    )
    df["bucket"] = df.apply(
        lambda row: assign_bucket(row["N"], row["excess_mad"], row["bias_risk"]),
        axis=1,
    )
    df["bucket_label"] = df["bucket"].map(lambda key: BUCKETS[key]["label"])
    df["topic"] = df["column"].apply(infer_topic)
    return df.sort_values(["quality_score", "excess_mad"], ascending=[False, True]).reset_index(drop=True)


@st.cache_data
def load_states() -> pd.DataFrame:
    df = load_state_catalog().copy()
    if "reporting_pct" in df.columns:
        df["reporting_percent"] = 100 * df["reporting_pct"]
    else:
        df["reporting_percent"] = np.nan
    return df.sort_values("excess_mad", ascending=False).reset_index(drop=True)


@st.cache_data
def load_aggregate() -> dict:
    return load_aggregate_result()


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def plot_benford(row: pd.Series, title: str) -> plt.Figure:
    digits = list(range(1, 10))
    observed = row["observed_counts"]
    expected = row["expected_counts"]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))

    axis.bar(digits, observed, color="#8ecae6", edgecolor="white", linewidth=1.1, alpha=0.92)
    axis.plot(
        digits,
        expected,
        color="#d7191c",
        marker="o",
        linewidth=2.1,
        label="Expected (Benford)",
    )

    max_height = max(max(observed), max(expected)) if observed and expected else 1
    percent_diffs = []
    if "percent_differences" in row:
        percent_diffs = row["percent_differences"]
    else:
        percent_diffs = [
            ((obs - exp) / exp * 100) if exp else 0.0
            for obs, exp in zip(observed, expected)
        ]

    for digit, obs, diff in zip(digits, observed, percent_diffs):
        axis.text(
            digit,
            obs + (0.018 * max_height),
            f"{diff:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#4b5563",
        )

    axis.set_title(title, loc="left", fontsize=14, fontweight="bold")
    axis.set_xlabel("Leading digit")
    axis.set_ylabel("Count")
    axis.set_xticks(digits)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.15)
    figure.tight_layout()
    return figure


def make_triage_chart(df: pd.DataFrame) -> go.Figure:
    rng = np.random.default_rng(42)
    fig = go.Figure()

    for position, bucket in enumerate(BUCKET_ORDER):
        subset = df[df["bucket"] == bucket].sort_values("quality_score")
        if subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=subset["quality_score"],
                y=np.full(len(subset), position) + rng.uniform(-0.17, 0.17, len(subset)),
                mode="markers",
                marker=dict(
                    size=11,
                    color=BUCKETS[bucket]["color"],
                    line=dict(color="white", width=1),
                    opacity=0.9,
                ),
                customdata=subset[
                    [
                        "display_name",
                        "topic",
                        "bucket_label",
                        "quality_score",
                        "excess_mad",
                        "reporting_pct",
                        "missingness_auc",
                        "bias_risk",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Topic: %{customdata[1]}<br>"
                    "Bucket: %{customdata[2]}<br>"
                    "Score: %{customdata[3]:.1f}<br>"
                    "EMAD: %{customdata[4]:.4f}<br>"
                    "Reporting rate: %{customdata[5]:.1f}%<br>"
                    "AUC: %{customdata[6]:.2f}<br>"
                    "Bias risk: %{customdata[7]:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=390,
        margin=dict(l=30, r=20, t=10, b=45),
        plot_bgcolor="white",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(BUCKET_ORDER))),
            ticktext=[BUCKETS[bucket]["label"] for bucket in BUCKET_ORDER],
            showgrid=False,
            zeroline=False,
            title="",
        ),
        xaxis=dict(
            title="Quality score (0-100)",
            range=[-1, 101],
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
        ),
    )
    return fig


def make_state_chart(df: pd.DataFrame) -> go.Figure:
    subset = pd.concat([df.head(10), df.tail(10)]).drop_duplicates("state")
    subset = subset.sort_values("excess_mad", ascending=True)

    colors = ["#d7191c" if value >= EMAD_NONCONFORMING else "#2ca25f" for value in subset["excess_mad"]]
    fig = go.Figure(
        go.Bar(
            x=subset["excess_mad"],
            y=subset["state"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "EMAD: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=10, b=35),
        plot_bgcolor="white",
        xaxis=dict(title="Excess MAD", showgrid=True, gridcolor="rgba(148,163,184,0.18)"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    fig.add_vline(x=EMAD_NONCONFORMING, line_dash="dash", line_color="#111827")
    return fig


def _rate_color(rate: float) -> str:
    if rate >= 0.8:
        return "#2ca25f"
    if rate >= 0.6:
        return "#f0a500"
    return "#d7191c"


def _has_profile(profile: object) -> bool:
    if not isinstance(profile, dict):
        return False
    return any(isinstance(value, Iterable) and len(value) > 0 for value in profile.values())


def _profile_bar_chart(records: list[dict], label_key: str, title: str, height: int = 240, sort_by_rate: bool = True) -> go.Figure:
    if not records:
        return go.Figure()
    ordered = sorted(records, key=lambda row: row["reported"]) if sort_by_rate else list(records)
    labels = [str(row[label_key]) for row in ordered]
    rates = [100 * row["reported"] for row in ordered]
    counts = [row["count"] for row in ordered]

    figure = go.Figure(
        go.Bar(
            x=rates,
            y=labels,
            orientation="h",
            marker=dict(color=[_rate_color(row["reported"]) for row in ordered]),
            customdata=np.array(counts).reshape(-1, 1),
            text=[f"{rate:.0f}%" for rate in rates],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Reporting rate: %{x:.1f}%<br>"
                "Districts: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14)),
        height=height,
        margin=dict(l=10, r=35, t=40, b=10),
        plot_bgcolor="white",
        xaxis=dict(range=[0, 105], showticklabels=False, showgrid=False, title=""),
        yaxis=dict(showgrid=False, title=""),
        showlegend=False,
    )
    return figure


# ---------------------------------------------------------------------------
# Narrative helpers
# ---------------------------------------------------------------------------

def missingness_status(row: pd.Series) -> str:
    auc = row.get("missingness_auc")
    if auc is None or pd.isna(auc):
        return "Missingness model not yet regenerated for this column."
    if auc < AUC_RANDOM:
        return "Missingness looks approximately random."
    bias = row.get("bias_risk", 0.0)
    if bias < RISK_READY:
        return "Missingness has some structure, but bias risk is still low."
    if bias < RISK_CAUTION:
        return "Missingness is meaningfully patterned. Use sensitivity checks."
    return "Missingness is strongly patterned and likely to bias comparisons."


def compliance_rows(row: pd.Series) -> list[tuple[bool, str, str]]:
    auc = row.get("missingness_auc")
    bias = row.get("bias_risk", 0.0)
    rows = [
        (
            row["N"] >= MIN_N_BENFORD,
            f"Sample size: {int(row['N']):,} usable values",
            f"Benford diagnostics are being run on {int(row['N']):,} non-zero observations.",
        ),
        (
            row["excess_mad"] <= MAD_GOOD,
            f"EMAD: {row['excess_mad']:.4f}",
            "Close conformity."
            if row["excess_mad"] <= MAD_GOOD
            else (
                "Marginal conformity."
                if row["excess_mad"] <= MAD_MARGINAL
                else "Nonconforming under the corrected EMAD threshold."
            ),
        ),
        (
            auc is not None and not pd.isna(auc) and auc < AUC_RANDOM,
            f"Missingness AUC: {auc:.2f}" if auc is not None and not pd.isna(auc) else "Missingness AUC: unavailable",
            missingness_status(row),
        ),
        (
            bias < RISK_READY,
            f"Bias risk: {bias:.2f}",
            "Low estimated bias risk."
            if bias < RISK_READY
            else ("Moderate estimated bias risk." if bias < RISK_CAUTION else "High estimated bias risk."),
        ),
    ]
    return rows


def bucket_rank_caption(row: pd.Series, bucket_df: pd.DataFrame) -> str:
    ordered = bucket_df.sort_values("quality_score", ascending=False).reset_index(drop=True)
    rank = int(ordered.index[ordered["column"] == row["column"]][0]) + 1
    return f"{row['display_name']} is ranked {rank} of {len(ordered)} inside the {BUCKETS[row['bucket']]['label'].lower()} bucket."


def reporting_narrative(row: pd.Series) -> str:
    auc = row.get("missingness_auc")
    if auc is None or pd.isna(auc):
        return (
            "This column does not yet have a regenerated missingness profile in the current EMAD-aligned universe. "
            "You can still inspect Benford fit and reporting rate, but the missingness mechanism should be rerun before treating it as fully triaged."
        )

    parts = [
        f"Overall reporting rate is {row['reporting_pct']:.1f}% of the full ESSER file.",
        missingness_status(row),
    ]
    profile = row.get("profile")
    if _has_profile(profile) and profile["state"]:
        state_rates = sorted(profile["state"], key=lambda item: item["reported"])
        low = state_rates[0]
        high = state_rates[-1]
        gap = (high["reported"] - low["reported"]) * 100
        if gap >= 10:
            parts.append(
                f"State reporting varies sharply, from {low['stateCode']} at {low['reported'] * 100:.0f}% to {high['stateCode']} at {high['reported'] * 100:.0f}%."
            )
    if _has_profile(profile) and profile["enrollment_quintile"]:
        ordered = sorted(profile["enrollment_quintile"], key=lambda item: item["enrollment_quintile"])
        smallest = ordered[0]["reported"]
        largest = ordered[-1]["reported"]
        gap = (largest - smallest) * 100
        if abs(gap) >= 8:
            direction = "more often" if gap > 0 else "less often"
            parts.append(
                f"The largest districts report this {direction} than the smallest ({largest * 100:.0f}% vs {smallest * 100:.0f}%)."
            )
    return " ".join(parts)


def selectbox_label(row: pd.Series) -> str:
    return (
        f"{BUCKETS[row['bucket']]['label']} | {row['quality_score']:>5.1f} | "
        f"{row['display_name']} ({row['column']})"
    )


def state_selectbox_label(row: pd.Series) -> str:
    return f"{row['state']} | EMAD {row['excess_mad']:+.4f} | N {int(row['N']):,}"


def filtered_variables(df: pd.DataFrame, topic: str, buckets: list[str], query: str) -> pd.DataFrame:
    view = df.copy()
    if topic != "All topics":
        view = view[view["topic"] == topic]
    if buckets:
        view = view[view["bucket"].isin(buckets)]
    if query:
        query_lower = query.lower()
        view = view[
            view["display_name"].str.lower().str.contains(query_lower)
            | view["column"].str.lower().str.contains(query_lower)
        ]
    return view.sort_values(["quality_score", "excess_mad"], ascending=[False, True])


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ESSER Variable Triage",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stMetricLabel"] { font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f3f4f6;
        border-radius: 10px 10px 0 0;
        font-weight: 700;
        padding: 12px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #111827 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()
states = load_states()
aggregate = load_aggregate()

state_nonconforming = int((states["excess_mad"] >= EMAD_NONCONFORMING).sum()) if not states.empty else 0
missingness_coverage = int(df["missingness_auc"].notna().sum())

st.title("ESSER variable triage")
st.caption(
    "Canonical research-facing app for the corrected EMAD pipeline. "
    "Use it to decide which ESSER variables are ready to analyze, which need sensitivity checks, "
    "and which should be treated as unstable."
)

if missingness_coverage < len(df):
    st.info(
        f"Missingness diagnostics are currently available for {missingness_coverage} of {len(df)} EMAD-tested columns. "
        "Regenerate `data/derived/missingness_mechanism.json` to refresh the full EMAD-aligned profile set."
    )

metric_cols = st.columns(5)
metric_cols[0].metric("Columns tested", f"{len(df):,}")
metric_cols[1].metric("Ready", int((df["bucket"] == "ready").sum()))
metric_cols[2].metric("Caution", int((df["bucket"] == "caution").sum()))
metric_cols[3].metric("Investigate", int((df["bucket"] == "investigate").sum()))
metric_cols[4].metric("Aggregate EMAD", f"{aggregate.get('excess_mad', 0.0):+.4f}")

topics = ["All topics"] + sorted(df["topic"].unique().tolist())
bucket_options = ["investigate", "caution", "ready"]

tab_overview, tab_detail, tab_reporting, tab_states = st.tabs(
    ["Overview", "Variable detail", "Who reports this?", "States"]
)

with tab_overview:
    controls = st.columns([1.2, 1.1, 1.7])
    with controls[0]:
        overview_topic = st.selectbox("Topic", topics, key="overview_topic")
    with controls[1]:
        overview_buckets = st.multiselect(
            "Buckets",
            bucket_options,
            default=list(reversed(bucket_options)),
            format_func=lambda key: BUCKETS[key]["label"],
            key="overview_buckets",
        )
    with controls[2]:
        overview_query = st.text_input("Search", placeholder="Search display name or column name", key="overview_query")

    overview_df = filtered_variables(df, overview_topic, overview_buckets, overview_query)
    if overview_df.empty:
        st.warning("No variables match the current filters.")
    else:
        st.plotly_chart(make_triage_chart(overview_df), use_container_width=True)
        st.caption(
            f"Showing {len(overview_df)} of {len(df)} variables. Score blends corrected EMAD fit with missingness safety."
        )
        st.dataframe(
            overview_df[
                [
                    "display_name",
                    "topic",
                    "bucket_label",
                    "quality_score",
                    "excess_mad",
                    "reporting_pct",
                    "missingness_auc",
                    "bias_risk",
                ]
            ].rename(
                columns={
                    "display_name": "Variable",
                    "topic": "Topic",
                    "bucket_label": "Bucket",
                    "quality_score": "Score",
                    "excess_mad": "EMAD",
                    "reporting_pct": "Reporting %",
                    "missingness_auc": "AUC",
                    "bias_risk": "Bias risk",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("What is this app showing?"):
        st.markdown(
            f"""
This app triages ESSER variables on **two independent dimensions**:

1. **Benford fit via corrected EMAD.** The corrected AERA 2026 pipeline uses package-based EMAD and treats
   **{EMAD_NONCONFORMING:.3f}** as the nonconformity threshold.
2. **Missingness mechanism.** A per-column logistic model estimates how predictable missingness is from state,
   year, district size, locale, and number of schools. Random-looking missingness is safer than patterned missingness.

The resulting **quality score** is half Benford fit and half missingness safety:

`score = 50 * benford_fit + 50 * missingness_safety`

- **Ready**: close EMAD conformity and low missingness bias risk.
- **Caution**: usable, but only with sensitivity checks.
- **Investigate**: high distortion, high bias risk, or too little usable signal.
            """
        )

with tab_detail:
    detail_controls = st.columns([1.1, 2.1])
    with detail_controls[0]:
        detail_topic = st.selectbox("Topic", topics, key="detail_topic")
    detail_pool = df if detail_topic == "All topics" else df[df["topic"] == detail_topic]
    detail_pool = detail_pool.sort_values(["quality_score", "excess_mad"], ascending=[False, True])

    option_map = {selectbox_label(row): row["column"] for _, row in detail_pool.iterrows()}
    with detail_controls[1]:
        detail_choice = st.selectbox("Variable", list(option_map), key="detail_variable")

    detail_row = detail_pool[detail_pool["column"] == option_map[detail_choice]].iloc[0]
    detail_bucket_df = df[df["bucket"] == detail_row["bucket"]]

    st.subheader(detail_row["display_name"])
    st.caption(
        f"`{detail_row['column']}` • {detail_row['topic']} • {BUCKETS[detail_row['bucket']]['label']} bucket"
    )

    detail_metrics = st.columns(4)
    detail_metrics[0].metric("Quality score", f"{detail_row['quality_score']:.1f}")
    detail_metrics[1].metric("EMAD", f"{detail_row['excess_mad']:+.4f}")
    detail_metrics[2].metric("Reporting rate", f"{detail_row['reporting_pct']:.1f}%")
    detail_metrics[3].metric(
        "Missingness AUC",
        "n/a" if pd.isna(detail_row["missingness_auc"]) else f"{detail_row['missingness_auc']:.2f}",
    )

    chart_col, notes_col = st.columns([3, 2])
    with chart_col:
        st.pyplot(plot_benford(detail_row, detail_row["display_name"]))
        st.progress(max(0.0, min(detail_row["quality_score"] / 100.0, 1.0)))
        st.caption(bucket_rank_caption(detail_row, detail_bucket_df))

    with notes_col:
        st.markdown(f"**Research recommendation**")
        st.markdown(BUCKETS[detail_row["bucket"]]["blurb"])
        st.markdown("**Checks**")
        for ok, headline, explanation in compliance_rows(detail_row):
            icon = "PASS" if ok else "FLAG"
            st.markdown(f"**{icon}** {headline}")
            st.caption(explanation)

    with st.expander("Raw metrics and interpretation"):
        st.markdown(
            f"""
- **N**: {int(detail_row['N']):,}
- **MAD**: {detail_row['mad']:.4f}
- **Expected MAD**: {detail_row['expected_mad']:.4f}
- **EMAD**: {detail_row['excess_mad']:+.4f}
- **Chi-squared**: {detail_row['chi2']:.2f}
- **p-value**: {detail_row['p_value']:.3e}
- **Reporting rate in full ESSER file**: {detail_row['reporting_pct']:.1f}%
- **Missingness model coverage**: {"available" if not pd.isna(detail_row['missingness_auc']) else "not yet regenerated"}
- **Bias risk**: {detail_row['bias_risk']:.2f}
            """
        )

with tab_reporting:
    reporting_controls = st.columns([1.1, 2.1])
    with reporting_controls[0]:
        reporting_topic = st.selectbox("Topic", topics, key="reporting_topic")
    reporting_pool = df if reporting_topic == "All topics" else df[df["topic"] == reporting_topic]
    reporting_pool = reporting_pool.sort_values(["quality_score", "excess_mad"], ascending=[False, True])
    reporting_option_map = {selectbox_label(row): row["column"] for _, row in reporting_pool.iterrows()}
    with reporting_controls[1]:
        reporting_choice = st.selectbox("Variable", list(reporting_option_map), key="reporting_variable")

    reporting_row = reporting_pool[reporting_pool["column"] == reporting_option_map[reporting_choice]].iloc[0]
    st.subheader(reporting_row["display_name"])
    st.caption(f"`{reporting_row['column']}` • {reporting_row['topic']}")

    reporting_metrics = st.columns(4)
    reporting_metrics[0].metric("Reporting rate", f"{reporting_row['reporting_pct']:.1f}%")
    reporting_metrics[1].metric("Dataset missing", f"{reporting_row['dataset_missing_pct']:.1f}%")
    reporting_metrics[2].metric(
        "Predictability AUC",
        "n/a" if pd.isna(reporting_row["missingness_auc"]) else f"{reporting_row['missingness_auc']:.2f}",
    )
    reporting_metrics[3].metric("Bias risk", f"{reporting_row['bias_risk']:.2f}")

    st.markdown(
        f"""
<div style="background:#f8fafc;border-left:4px solid {BUCKETS[reporting_row['bucket']]['color']};
padding:12px 16px;margin:8px 0 16px 0;border-radius:4px;">
{reporting_narrative(reporting_row)}
</div>
        """,
        unsafe_allow_html=True,
    )

    profile = reporting_row.get("profile")
    if not _has_profile(profile):
        st.info(
            "This column does not yet have a regenerated reporting profile in the current EMAD-aligned missingness file."
        )
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _profile_bar_chart(
                    profile["state"],
                    label_key="stateCode",
                    title="By state",
                    height=max(280, 16 * len(profile["state"]) + 60),
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with right:
            enrollment_records = []
            for row in profile["enrollment_quintile"]:
                label = {
                    1: "Q1 smallest",
                    2: "Q2",
                    3: "Q3",
                    4: "Q4",
                    5: "Q5 largest",
                }.get(int(row["enrollment_quintile"]), f"Q{row['enrollment_quintile']}")
                enrollment_records.append(
                    {
                        "label": label,
                        "reported": row["reported"],
                        "count": row["count"],
                    }
                )
            st.plotly_chart(
                _profile_bar_chart(
                    enrollment_records,
                    label_key="label",
                    title="By enrollment quintile",
                    height=230,
                    sort_by_rate=False,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.plotly_chart(
                _profile_bar_chart(
                    profile["locale"],
                    label_key="locale_name",
                    title="By urban-rural locale",
                    height=320,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

with tab_states:
    st.markdown(
        "State results pool all scale-free ESSER dollar columns inside each state, then rerun the corrected EMAD analysis."
    )

    state_metrics = st.columns(4)
    state_metrics[0].metric("States tested", f"{len(states):,}")
    state_metrics[1].metric("Nonconforming", f"{state_nonconforming:,}")
    state_metrics[2].metric("Worst EMAD", f"{states.iloc[0]['state']} {states.iloc[0]['excess_mad']:+.4f}")
    state_metrics[3].metric("Cleanest EMAD", f"{states.iloc[-1]['state']} {states.iloc[-1]['excess_mad']:+.4f}")

    st.plotly_chart(make_state_chart(states), use_container_width=True)

    state_option_map = {state_selectbox_label(row): row["state"] for _, row in states.iterrows()}
    state_choice = st.selectbox("State detail", list(state_option_map), key="state_detail")
    state_row = states[states["state"] == state_option_map[state_choice]].iloc[0]

    detail_metrics = st.columns(4)
    detail_metrics[0].metric("State", state_row["state"])
    detail_metrics[1].metric("EMAD", f"{state_row['excess_mad']:+.4f}")
    detail_metrics[2].metric("N", f"{int(state_row['N']):,}")
    detail_metrics[3].metric(
        "Reporting completeness",
        "n/a" if pd.isna(state_row["reporting_percent"]) else f"{state_row['reporting_percent']:.1f}%",
    )

    st.pyplot(plot_benford(state_row, f"State {state_row['state']}"))
    st.dataframe(
        states[["state", "excess_mad", "mad", "N", "reporting_percent"]]
        .rename(
            columns={
                "state": "State",
                "excess_mad": "EMAD",
                "mad": "MAD",
                "N": "N",
                "reporting_percent": "Reporting %",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
