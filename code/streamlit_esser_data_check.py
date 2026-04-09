"""
ESSER variable triage dashboard.

A bucketed-gradient redesign of the original Power BI Benford dashboard.
Every variable is assigned a bucket (Ready / Caution / Investigate / Too sparse)
AND a continuous 0-100 quality score, so users can see where in a bucket each
variable sits and how close it is to the next bucket over.

Designed for two audiences at once:
  * Middle-school reader: traffic-light buckets + plain-English names/explanations
  * Research scoping: transparent score formula + topic filter + raw metrics +
    "position within bucket" thermometer + selectable detail view.

Run with:  streamlit run code/streamlit_esser_data_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "derived" / "benford_analysis_summary.json"
MISSINGNESS_PATH = HERE.parent / "data" / "derived" / "missingness_mechanism.json"

# Total rows in the ESSER expenditures dataset — used for missingness.
# Matches the constant baked into the original Power BI Python visual.
TOTAL_ROWS = 17_705

# Benford thresholds follow Nigrini (2012):
#   excess MAD <= 0.005  -> close conformity
#   excess MAD <= 0.015  -> marginal conformity (upper bound of "usable")
#   excess MAD  > 0.015  -> nonconformity
MAD_GOOD = 0.005
MAD_MARGINAL = 0.015

# Minimum absolute sample size for Benford's Law analysis to be meaningful.
# Nigrini (2012) recommends N >= 1000 observations; below that, sampling noise
# in the MAD/chi-squared statistics swamps the signal regardless of the point
# estimate. This is the empirical basis for the "sparse" disqualifier.
MIN_N_BENFORD = 1000

# Missingness-mechanism thresholds.
#
# bias_risk = 2*max(0, AUC - 0.5) * (pct_missing / 100)
# Interpretation: approximately the fraction of the dataset whose presence/
# absence is *systematically* tied to observed structure. AUC=0.5 (random)
# always yields zero risk regardless of how much is missing -- which is the
# whole point. The 0.5 factor is so AUC=1.0 maps to "1 unit of suspicion
# per missing row" instead of "2 units".
#
# Cutoffs are calibrated against the realized distribution in the ESSER data:
# the median bias_risk is ~0.03 and the 75th percentile ~0.28.
RISK_READY = 0.05      # at most 5% of dataset systematically biased
RISK_CAUTION = 0.20    # at most 20% systematically biased
AUC_RANDOM = 0.65      # below this, missingness considered random

BUCKETS = {
    "ready": {
        "label": "Ready to use",
        "color": "#2ca25f",      # green
        "icon": "[OK]",
        "blurb": "Clean Benford fit and missingness is approximately random.",
    },
    "caution": {
        "label": "Use with caution",
        "color": "#f0a500",      # amber
        "icon": "[!]",
        "blurb": "Marginal Benford or some missingness bias. Usable with care.",
    },
    "investigate": {
        "label": "Investigate first",
        "color": "#d7191c",      # red
        "icon": "[X]",
        "blurb": "Bad Benford fit OR strongly biased missingness pattern.",
    },
}
BUCKET_ORDER = ["ready", "caution", "investigate"]


# ---------------------------------------------------------------------------
# Topic inference (from column name prefix)
# ---------------------------------------------------------------------------

def infer_topic(col: str) -> str:
    c = col.lower()
    # most specific first
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


# ---------------------------------------------------------------------------
# Quality score + bucket assignment
# ---------------------------------------------------------------------------

def compute_bias_risk(missingness_auc: float | None, pct_missing: float | None) -> float:
    """Approximate fraction of the dataset whose presence/absence is
    systematically tied to observed district characteristics.

    Returns 0 when AUC is at or below chance (0.5) -- i.e. when missingness
    is empirically random, the variable carries no bias risk regardless of
    how much is missing. This is the empirical version of "if missingness
    is random, the variable is still usable".
    """
    if (missingness_auc is None or pd.isna(missingness_auc)
            or pct_missing is None or pct_missing < 0.5):
        return 0.0
    auc_excess = max(0.0, float(missingness_auc) - 0.5) * 2.0  # 0..1
    return float(auc_excess * (pct_missing / 100.0))


def compute_score(excess_mad: float, bias_risk: float) -> float:
    """0-100 quality score combining Benford fit and missingness safety.

    Half is Benford shape match; half is "fraction of the dataset that
    isn't systematically biased". Coverage no longer enters the score
    directly because bias_risk subsumes it: 80% missing with random
    missingness is fine, 20% missing with systematic missingness is not.
    """
    fit = max(0.0, 1.0 - (excess_mad / MAD_MARGINAL))
    safety = max(0.0, 1.0 - (bias_risk / 0.5))  # bias_risk 0.5 -> 0 safety
    return 100.0 * (0.5 * fit + 0.5 * safety)


def assign_bucket(num_valid: int, excess_mad: float, bias_risk: float) -> str:
    """Two-dimensional bucketing: Benford fit + empirical missingness mechanism.

    - Investigate: Benford clearly fails OR missingness is so systematic
      and so substantial that >20% of the dataset is biased. Also catches
      pathologically small N (< Nigrini's 1000 minimum) defensively, though
      no ESSER column hits that case (min N = 1,851).
    - Ready: clean Benford fit AND missingness affects <5% of the dataset
      in a biased way (or is empirically random).
    - Caution: marginal Benford OR moderate missingness bias.
    """
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
    # MAD_GOOD < excess_mad <= MAD_MARGINAL  (marginal Benford fit)
    if bias_risk < RISK_READY * 2:  # be a little more forgiving here
        return "caution"
    return "investigate"


# ---------------------------------------------------------------------------
# Data load + enrichment
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    with open(DATA_PATH) as f:
        df = pd.DataFrame(json.load(f))
    # Merge in the precomputed missingness mechanism (run
    # `python code/missingness_mechanism.py` to regenerate).
    if MISSINGNESS_PATH.exists():
        miss = pd.DataFrame(json.loads(MISSINGNESS_PATH.read_text()))
        # Ensure the profile column exists even in older JSON files that
        # predate the reporting-profile precompute.
        if "profile" not in miss.columns:
            miss["profile"] = None
        df = df.merge(
            miss[["column", "missingness_auc", "pct_missing", "profile"]],
            on="column",
            how="left",
        )
    else:
        df["missingness_auc"] = None
        df["pct_missing"] = None
        df["profile"] = None
    df["coverage"] = df["num_valid_entries"] / TOTAL_ROWS
    df["missingness_pct"] = 100 * (1 - df["coverage"])
    df["bias_risk"] = df.apply(
        lambda r: compute_bias_risk(r["missingness_auc"], r["pct_missing"]),
        axis=1,
    )
    df["quality_score"] = df.apply(
        lambda r: compute_score(r["excess_mad"], r["bias_risk"]),
        axis=1,
    )
    df["bucket"] = df.apply(
        lambda r: assign_bucket(
            r["num_valid_entries"], r["excess_mad"], r["bias_risk"]
        ),
        axis=1,
    )
    df["topic"] = df["column"].apply(infer_topic)
    # Prefer the description; fall back to the technical column name.
    df["display_name"] = (
        df["description"].fillna("").str.strip().replace("", np.nan).fillna(df["column"])
    )
    return df


# ---------------------------------------------------------------------------
# Triage strip chart (main visual)
# ---------------------------------------------------------------------------

def make_triage_chart(df: pd.DataFrame) -> go.Figure:
    """Monotonic staircase: buckets stack bottom-to-top in quality order, and
    within each bucket items are placed left-to-right by score inside a
    reserved x-band. The result is a strictly ascending layout: every red dot
    sits left-and-below every yellow dot, every yellow dot left-and-below
    every green one.
    """
    # Worst-to-best going up the y-axis, and left-to-right on the x-axis.
    # Only render lanes that actually contain variables -- the sparse bucket
    # is typically empty because the ESSER columns all exceed N >= 1000.
    all_lanes = list(reversed(BUCKET_ORDER))  # sparse, investigate, caution, ready
    present = set(df["bucket"].unique())
    lanes = [b for b in all_lanes if b in present]
    n_lanes = len(lanes)
    band_w = 100 / n_lanes  # each bucket owns an equal slice of the x-axis
    bands = {b: (i * band_w, (i + 1) * band_w) for i, b in enumerate(lanes)}

    rng = np.random.default_rng(42)
    fig = go.Figure()

    # Soft colored background for each lane's reserved band.
    for i, b in enumerate(lanes):
        lo, hi = bands[b]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=lo, x1=hi, y0=i - 0.48, y1=i + 0.48,
            fillcolor=BUCKETS[b]["color"],
            opacity=0.10,
            line_width=0,
            layer="below",
        )
        # Thin vertical divider between bands.
        if i > 0:
            fig.add_shape(
                type="line", xref="x", yref="paper",
                x0=lo, x1=lo, y0=0, y1=1,
                line=dict(color="#e5e7eb", width=1, dash="dot"),
                layer="below",
            )

    for i, b in enumerate(lanes):
        sub = (
            df[df["bucket"] == b]
            .sort_values("quality_score", ascending=True)
            .reset_index(drop=True)
        )
        if sub.empty:
            continue
        lo, hi = bands[b]
        n = len(sub)
        # Distribute evenly inside the band with a small inner margin.
        inner_lo = lo + 0.5
        inner_hi = hi - 0.5
        if n == 1:
            xs = np.array([(inner_lo + inner_hi) / 2])
        else:
            xs = np.linspace(inner_lo, inner_hi, n)
        ys = i + rng.uniform(-0.28, 0.28, n)

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=12,
                color=BUCKETS[b]["color"],
                line=dict(width=1.0, color="white"),
                opacity=0.92,
            ),
            customdata=np.stack([
                sub["display_name"].values,
                sub["topic"].values,
                sub["quality_score"].values,
                sub["missingness_pct"].values,
                sub["excess_mad"].values,
                sub["missingness_auc"].fillna(0.5).values,
                sub["bias_risk"].values,
            ], axis=-1),
            hovertemplate=(
                "<b style='font-size:15px;color:#111'>%{customdata[0]}</b><br>"
                "<span style='color:#374151;font-weight:600'>%{customdata[1]}</span><br>"
                "<b style='font-size:16px;color:#111'>Score %{customdata[2]:.0f} / 100</b><br>"
                "<span style='color:#111'>"
                "Missing: %{customdata[3]:.0f}%  &nbsp; "
                "MAD: %{customdata[4]:.3f}<br>"
                "Missingness AUC: %{customdata[5]:.2f}  &nbsp; "
                "Bias risk: %{customdata[6]:.2f}"
                "</span>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_lanes)),
            ticktext=[
                f"{BUCKETS[b]['icon']} {BUCKETS[b]['label']}" for b in lanes
            ],
            range=[-0.6, n_lanes - 0.4],
            showgrid=False,
            zeroline=False,
        ),
        xaxis=dict(
            title="Research-readiness (low to high)",
            range=[-1, 101],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        height=420,
        margin=dict(l=180, r=30, t=20, b=55),
        plot_bgcolor="white",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#1f2937",
            font=dict(size=14, family="Segoe UI, Helvetica, Arial, sans-serif",
                      color="#111"),
            align="left",
            namelength=-1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Detail: Benford bar/line chart
# ---------------------------------------------------------------------------

def plot_benford_detail(row: pd.Series) -> plt.Figure:
    digits = list(range(1, 10))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(digits, row["observed_counts"], color="#8ecae6",
           label="Observed", alpha=0.9)
    ax.plot(digits, row["expected_counts"], color="#d7191c",
            marker="o", label="Expected (Benford)", linewidth=2)

    # Percent difference labels (like the original pbix)
    max_h = max(row["observed_counts"]) if max(row["observed_counts"]) > 0 else 1
    for d, obs, diff in zip(digits, row["observed_counts"], row["percent_differences"]):
        ax.text(d, obs + 0.01 * max_h, f"{diff:+.0f}%",
                ha="center", va="bottom", fontsize=9, color="#555")

    ax.set_xlabel("Leading digit")
    ax.set_ylabel("Count")
    ax.set_xticks(digits)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Detail: plain-language checklist
# ---------------------------------------------------------------------------

def compliance_rows(row: pd.Series) -> list[tuple[bool, str, str]]:
    """Return (pass?, headline, explanation) for each check."""
    n_valid = row["num_valid_entries"]
    mad = row["excess_mad"]
    p_val = row["p_value"]
    chi2 = row["chi_squared"]
    ed = row["euclidean_distance"]
    auc = row.get("missingness_auc")
    bias = row.get("bias_risk", 0.0)

    miss_pct = 100 * (1 - n_valid / TOTAL_ROWS)

    checks = []
    # 1. Sample size
    checks.append((
        n_valid >= MIN_N_BENFORD,
        f"Sample size: {n_valid:,} of {TOTAL_ROWS:,} rows ({miss_pct:.1f}% missing)",
        f"At least {MIN_N_BENFORD:,} valid entries -- enough for Benford's Law "
        "to be statistically meaningful (Nigrini 2012)."
        if n_valid >= MIN_N_BENFORD else
        f"Fewer than {MIN_N_BENFORD:,} valid entries. Benford results below "
        "are noisy and should not be trusted.",
    ))
    # 2. Missingness mechanism (the new dimension)
    if auc is None or pd.isna(auc) or miss_pct < 0.5:
        checks.append((
            True,
            "Missingness mechanism: not applicable",
            "No meaningful missingness, so there's nothing to bias.",
        ))
    else:
        if auc < AUC_RANDOM:
            verdict = ("Missingness looks empirically random (MCAR-like). "
                       "The missing rows are not predictable from district "
                       "state, size, or locale -- so the visible rows are a "
                       "fair sample.")
            ok = True
        elif bias < RISK_READY:
            verdict = (f"Missingness has structure but only {bias*100:.1f}% "
                       "of the dataset is biased -- small enough to ignore "
                       "for most purposes.")
            ok = True
        elif bias < RISK_CAUTION:
            verdict = (f"Roughly {bias*100:.0f}% of the dataset has "
                       "systematic missingness. Use with care: results may "
                       "be skewed toward whichever districts do report.")
            ok = False
        else:
            verdict = (f"Roughly {bias*100:.0f}% of the dataset is "
                       "systematically missing -- the missing rows differ "
                       "from the visible rows in predictable ways. "
                       "Conclusions drawn from this column will likely be "
                       "biased.")
            ok = False
        checks.append((
            ok,
            f"Missingness mechanism: AUC = {auc:.2f}, bias risk = {bias:.2f}",
            verdict,
        ))
    # 3. Benford shape match
    checks.append((
        mad <= MAD_GOOD,
        f"Benford shape match (excess MAD = {mad:.4f})",
        "Leading digits closely match Benford's Law -- this looks like real-world data."
        if mad <= MAD_GOOD else
        ("Close-ish to Benford but not perfect. OK with care."
         if mad <= MAD_MARGINAL else
         "Leading digits deviate a lot. Something non-natural is going on "
         "(rounding, caps, defaults, data-entry habits, fraud)."),
    ))
    # 4. Chi-squared p-value
    checks.append((
        p_val >= 0.001,
        f"Chi-squared p-value = {p_val:.4f}",
        "p >= 0.001, so the deviation isn't extreme."
        if p_val >= 0.001 else
        "Tiny p-value: with 17k+ rows, even small shape mismatches "
        "become 'significant'. Excess MAD is a better guide here.",
    ))
    # 5. Euclidean distance
    checks.append((
        ed <= 0.040,
        f"Euclidean distance = {ed:.4f}",
        "Overall distance to Benford is small."
        if ed <= 0.040 else
        "Overall distance to Benford is large.",
    ))
    return checks


# ---------------------------------------------------------------------------
# Detail: dense one-line "where in the bucket" widget (HTML)
# ---------------------------------------------------------------------------

def bucket_position_html(
    row: pd.Series, bucket_df: pd.DataFrame, meta: dict
) -> str:
    """Single-line horizontal widget: rank text on the left, mini gradient
    bar on the right, everything on one row."""
    here = float(row["quality_score"])
    lo = float(bucket_df["quality_score"].min())
    hi = float(bucket_df["quality_score"].max())
    if hi - lo < 0.5:  # degenerate single-member bucket
        lo, hi = max(0.0, here - 1.0), min(100.0, here + 1.0)
    # Rank within bucket: 1 = best.
    rank = int((bucket_df["quality_score"] > here).sum() + 1)
    total = len(bucket_df)
    color = meta["color"]
    return (
        f"<div style='display:flex;align-items:center;gap:16px;"
        f"margin:4px 0 14px 0;font-size:13px;color:#1f2937'>"
        f"<div style='flex:0 0 auto;white-space:nowrap'>"
        f"<b style='color:{color}'>Rank {rank} of {total}</b> "
        f"in {meta['label']} &middot; score <b>{here:.0f}</b> "
        f"&middot; bucket range {lo:.0f}&ndash;{hi:.0f}"
        f"</div>"
        f"<div style='flex:1;position:relative;height:10px;"
        f"background:#f1f3f4;border-radius:5px;min-width:140px'>"
        f"<div style='position:absolute;left:{lo}%;"
        f"width:{max(hi-lo,0.5)}%;top:0;bottom:0;"
        f"background:{color};opacity:0.30;border-radius:5px'></div>"
        f"<div style='position:absolute;left:calc({here}% - 2px);top:-3px;"
        f"width:4px;height:16px;background:{color};border-radius:2px'></div>"
        f"</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# "Who reports this?" tab helpers
# ---------------------------------------------------------------------------

def _rate_color(reported: float) -> str:
    """Continuous green-amber-red mapping for reporting rate (0..1)."""
    if reported >= 0.5:
        return "#2ca25f"
    if reported >= 0.25:
        return "#f0a500"
    return "#d7191c"


def _has_profile(profile) -> bool:
    """Robust check: profile may be None, NaN (after a pandas merge), or a
    dict. Only a real dict counts as 'has a profile'."""
    return isinstance(profile, dict) and bool(profile)


def reporting_narrative(row: pd.Series) -> str:
    """Generate a 2-3 sentence plain-language summary of who reports this
    variable, driven by the precomputed profile + AUC."""
    profile = row.get("profile")
    pct_miss = row.get("pct_missing")
    auc = row.get("missingness_auc")

    # 0% missing: the simple happy case -- there's nothing to explore.
    if pd.notna(pct_miss) and float(pct_miss) < 0.5:
        return (f"Almost every district reports this column "
                f"({float(pct_miss):.1f}% missing). "
                "The visible sample is the entire population.")

    # We have meaningful missingness but no profile to drill into --
    # could be an all-missing edge case or precompute skipped it.
    if not _has_profile(profile):
        if pd.notna(pct_miss):
            return (f"About {100 - float(pct_miss):.0f}% of districts "
                    f"report this column. Group-level reporting profile "
                    "isn't available -- re-run "
                    "`code/missingness_mechanism.py` to regenerate it.")
        return ("Reporting-profile data isn't available for this column. "
                "Re-run `code/missingness_mechanism.py` to regenerate it.")

    pct_miss_f = float(pct_miss)
    parts = []

    # Overall framing
    if auc is None or pd.isna(auc) or float(auc) < AUC_RANDOM:
        parts.append(
            f"About {100 - pct_miss_f:.0f}% of districts report this column, "
            "and missingness looks empirically unpredictable from district "
            "structure -- the visible sample is roughly representative."
        )
        return " ".join(parts)

    parts.append(
        f"About {100 - pct_miss_f:.0f}% of districts report this column. "
        f"Missingness is predictable (AUC = {float(auc):.2f}), meaning the "
        "visible sample is not a random slice of the country."
    )

    # State signal
    states = profile.get("state", [])
    if states:
        top_states = states[:3]
        bottom_states = [s for s in reversed(states) if s["count"] >= 50][:3]
        if top_states and bottom_states:
            top_str = ", ".join(
                f"{s['stateCode']} ({s['reported']*100:.0f}%)" for s in top_states
            )
            bot_str = ", ".join(
                f"{s['stateCode']} ({s['reported']*100:.0f}%)" for s in bottom_states
            )
            parts.append(
                f"Reporting rate is highest in {top_str} and lowest in "
                f"{bot_str}."
            )

    # Size signal
    eq = profile.get("enrollment_quintile", [])
    if len(eq) == 5:
        smallest = eq[0]["reported"]
        largest = eq[-1]["reported"]
        gap = largest - smallest
        if abs(gap) >= 0.05:
            direction = "more" if gap > 0 else "less"
            parts.append(
                f"The largest districts report this {direction} often than "
                f"the smallest "
                f"({largest*100:.0f}% vs {smallest*100:.0f}%)."
            )

    return " ".join(parts)


def _profile_bar_chart(
    records: list[dict],
    label_key: str,
    title: str,
    height: int = 240,
    sort_by_rate: bool = True,
) -> go.Figure:
    """Horizontal bar chart of reporting rate per group."""
    if not records:
        return go.Figure()
    if sort_by_rate:
        recs = sorted(records, key=lambda r: r["reported"])
    else:
        recs = list(records)
    labels = [str(r[label_key]) for r in recs]
    rates = [r["reported"] * 100 for r in recs]
    counts = [r["count"] for r in recs]
    colors = [_rate_color(r["reported"]) for r in recs]

    fig = go.Figure(
        go.Bar(
            x=rates,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            customdata=np.array(counts).reshape(-1, 1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Reporting rate: %{x:.1f}%<br>"
                "Districts in group: %{customdata[0]:,}"
                "<extra></extra>"
            ),
            text=[f"{r:.0f}%" for r in rates],
            textposition="outside",
            textfont=dict(size=11, color="#1f2937"),
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left",
                   font=dict(size=14, color="#111")),
        xaxis=dict(range=[0, 105], showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=11, color="#1f2937")),
        height=height,
        margin=dict(l=10, r=40, t=40, b=10),
        plot_bgcolor="white",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", bordercolor="#1f2937",
                        font=dict(size=13, color="#111")),
    )
    return fig


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ESSER Variable Triage",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Make the tabs prominent -- Streamlit's default tab styling is easy to miss.
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 17px !important;
        font-weight: 700 !important;
        padding: 12px 26px !important;
        background-color: #f3f4f6;
        border-radius: 8px 8px 0 0;
        color: #4b5563;
    }
    .stTabs [aria-selected="true"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        box-shadow: 0 -2px 0 0 #111827 inset;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()

# --- Title + dense headline counts -----------------------------------------
st.markdown(
    "<h1 style='margin:0 0 2px 0;font-size:28px'>ESSER variable triage</h1>"
    "<p style='color:#666;margin:0 0 10px 0;font-size:13px'>"
    "Which of the 220 ESSER expenditure variables are trustworthy enough to "
    "use in research? Each one gets a bucket AND a score inside it."
    "</p>",
    unsafe_allow_html=True,
)

counts = df["bucket"].value_counts().to_dict()
total = len(df)
segments = []
for b in BUCKET_ORDER:  # best-first, left-to-right
    meta = BUCKETS[b]
    n = counts.get(b, 0)
    pct = 100 * n / total if total else 0
    segments.append(
        f"<div title='{meta['blurb']}' style='flex:1;"
        f"background:{meta['color']};color:white;padding:8px 12px;"
        f"border-right:1px solid rgba(255,255,255,0.25)'>"
        f"<div style='font-size:10px;text-transform:uppercase;"
        f"letter-spacing:0.5px;opacity:0.95;white-space:nowrap'>"
        f"{meta['icon']} {meta['label']}</div>"
        f"<div style='font-size:22px;font-weight:700;line-height:1.15'>"
        f"{n}<span style='font-size:11px;opacity:0.85;"
        f"margin-left:5px;font-weight:400'>({pct:.0f}%)</span></div>"
        f"</div>"
    )
st.markdown(
    "<div style='display:flex;border-radius:5px;overflow:hidden;"
    "margin-bottom:14px;font-family:sans-serif'>"
    + "".join(segments)
    + "</div>",
    unsafe_allow_html=True,
)

# --- Tabs: overview / detail / who-reports-this ----------------------------
topics = ["All topics"] + sorted(df["topic"].unique().tolist())

tab_overview, tab_detail, tab_who = st.tabs(
    ["Overview", "Variable detail", "Who reports this?"]
)

with tab_overview:
    topic = st.selectbox(
        "Filter by topic",
        topics,
        index=0,
        key="overview_topic",
    )
    view = df.copy() if topic == "All topics" else df[df["topic"] == topic]
    if view.empty:
        st.warning("No variables match that topic.")
    else:
        st.plotly_chart(make_triage_chart(view), use_container_width=True)
        st.caption(
            f"Showing **{len(view)}** of {len(df)} variables. "
            "Lanes stack bottom-to-top from worst to best bucket; within a "
            "lane, items are ordered left-to-right by quality score. Hover "
            "a dot for its name, topic, and score."
        )
        with st.expander("What is this actually showing?"):
            st.markdown(
                f"""
This dashboard rates each ESSER variable on **two independent dimensions**
of data quality:

1. **Benford's Law fit** -- in naturally occurring numeric data the digit
   **1** appears first about 30% of the time, **2** about 18%, down to
   **9** under 5%. When leading digits *don't* follow that curve, it
   suggests rounding, caps, defaults, data-entry shortcuts, or fabrication.
   We measure the deviation with **excess MAD** (Nigrini 2012).

2. **Missingness mechanism** -- *why* values are missing matters more than
   *how many*. We fit a per-column logistic regression that tries to predict
   `is_missing` from district state, size, locale, and reporting year, then
   take the cross-validated **AUC**. AUC near 0.5 means missingness is
   empirically unpredictable (MCAR-like, safe). AUC near 1.0 means the
   missing rows are *exactly* the kind of districts our covariates pick out
   -- a strong sign of bias.

We combine the two as a **bias risk** = `2*(AUC-0.5) * pct_missing` --
roughly the fraction of the dataset that's systematically biased.

- **Quality score** = 50% Benford fit + 50% missingness safety.
- **Buckets** use excess MAD (<=0.005 good, <=0.015 marginal) and bias_risk
  (<{RISK_READY} ready, <{RISK_CAUTION} caution).
- **Position inside a lane** shows whether a variable is barely making the
  cut or comfortably inside its bucket.
                """
            )

with tab_detail:
    # Side-by-side: topic on the left, variables-in-that-topic on the right.
    tc1, tc2 = st.columns([1, 2])
    with tc1:
        topic_d = st.selectbox(
            "Topic", topics, index=0, key="detail_topic"
        )
    pool = df if topic_d == "All topics" else df[df["topic"] == topic_d]
    pool = pool.sort_values("quality_score", ascending=False)
    var_options = pool.apply(
        lambda r: f"{BUCKETS[r['bucket']]['icon']} {r['quality_score']:.0f}/100  -  {r['display_name'][:90]}",
        axis=1,
    ).tolist()
    opt_to_col = dict(zip(var_options, pool["column"].tolist()))
    with tc2:
        picked = st.selectbox(
            "Variable (sorted best first within the topic)",
            var_options,
            index=0,
            key="detail_var",
        )

    if not var_options:
        st.warning("No variables in that topic.")
    else:
        row = df[df["column"] == opt_to_col[picked]].iloc[0]
        bucket_df = df[df["bucket"] == row["bucket"]]
        meta = BUCKETS[row["bucket"]]

        # Slim name header so you know what the chart below is showing.
        st.markdown(
            f"<div style='margin:10px 0 4px 0'>"
            f"<div style='font-size:20px;font-weight:700;line-height:1.25;"
            f"color:#111'>{row['display_name']}</div>"
            f"<div style='font-size:12px;color:#4b5563;margin-top:2px'>"
            f"<code>{row['column']}</code> &middot; {row['topic']} &middot; "
            f"<span style='color:{meta['color']};font-weight:700'>"
            f"{meta['icon']} {meta['label']}</span>"
            f" &middot; score <b>{row['quality_score']:.0f}/100</b>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # --- Chart first -------------------------------------------------
        st.pyplot(plot_benford_detail(row))

        # --- Dense single-line "where in bucket" ------------------------
        st.markdown(
            bucket_position_html(row, bucket_df, meta),
            unsafe_allow_html=True,
        )

        # --- Plain-language checklist -----------------------------------
        st.markdown("**Compliance checks**")
        for ok, headline, explain in compliance_rows(row):
            mark = "&#9989;" if ok else "&#10060;"
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<div style='font-size:14px;color:#111'>{mark} <b>{headline}</b></div>"
                f"<div style='font-size:12px;color:#4b5563;margin-left:22px'>{explain}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Score formula and methodology"):
            st.markdown(
                f"""
**Quality score (0-100):**
```
score = 50 * benford_fit + 50 * missingness_safety
benford_fit         = max(0, 1 - excess_MAD / {MAD_MARGINAL})
missingness_safety  = max(0, 1 - bias_risk / 0.5)
bias_risk           = 2 * max(0, AUC - 0.5) * (pct_missing / 100)
```
- **Excess MAD**: mean absolute deviation of observed from expected Benford
  frequencies (minus a small-sample bias correction). Sample-size independent,
  unlike chi-squared.
- **Missingness AUC**: cross-validated ROC AUC of a logistic regression that
  predicts `is_missing(column)` from `(stateCode, reportingYear,
  log_enrollment, urban_centric_locale, log_num_schools)`. AUC = 0.5 means
  empirically random missingness; AUC ~ 1 means missingness is fully
  predictable from observed structure.
- **Bias risk**: approximately the fraction of the dataset whose
  presence/absence is systematically tied to observed structure. Random
  missingness (AUC = 0.5) yields zero risk regardless of how much is missing.

**Bucket rules:**
- **Investigate** if MAD > {MAD_MARGINAL} OR (MAD <= {MAD_GOOD} AND
  bias_risk >= {RISK_CAUTION}).
- **Ready** if MAD <= {MAD_GOOD} AND bias_risk < {RISK_READY}.
- **Caution** otherwise.

*Note on sample size:* Nigrini (2012) recommends N >= {MIN_N_BENFORD:,}
observations as the minimum for Benford tests. Every ESSER column comfortably
clears this (min N = 1,851), so the dashboard does not need a separate
"too sparse" tier.

**Why this beats a coverage-only rule.** A column with 60% missing where the
missingness is empirically random is more trustworthy than a column with 20%
missing where the missing 20% are exactly the largest districts. Coverage
tells you *how much*; AUC tells you *whether it matters*. Bias risk combines
the two into a single number with a meaningful interpretation: roughly the
share of the dataset that's systematically biased.

**Why not trust the p-value?** With N ~ 17k, chi-squared is hypersensitive:
tiny shape mismatches become "significant" even when practically fine.
Excess MAD is the better guide for this dataset.

**Caveat on the missingness regression.** ~17% of rows (mostly charter LEAs
and special agencies) don't appear in the standard NCES district-
characteristics file and are dropped before fitting. AUC is computed on the
remaining ~14,300 rows. Run `python code/missingness_mechanism.py` to
regenerate.

*Sources:*
- *Nigrini, M. (2012). Benford's Law: Applications for Forensic Accounting,
  Auditing, and Fraud Detection. Wiley.*
- *Rubin, D. (1976). Inference and missing data. Biometrika 63(3).*
- *Little, R. J. A. (1988). A test of missing completely at random for
  multivariate data with missing values. JASA 83(404).*
                """
            )

with tab_who:
    # Same topic + variable layout as tab_detail.
    wc1, wc2 = st.columns([1, 2])
    with wc1:
        topic_w = st.selectbox(
            "Topic", topics, index=0, key="who_topic"
        )
    pool_w = df if topic_w == "All topics" else df[df["topic"] == topic_w]
    pool_w = pool_w.sort_values("quality_score", ascending=False)
    var_options_w = pool_w.apply(
        lambda r: f"{BUCKETS[r['bucket']]['icon']} {r['quality_score']:.0f}/100  -  {r['display_name'][:90]}",
        axis=1,
    ).tolist()
    opt_to_col_w = dict(zip(var_options_w, pool_w["column"].tolist()))
    with wc2:
        picked_w = st.selectbox(
            "Variable",
            var_options_w,
            index=0,
            key="who_var",
        )

    if not var_options_w:
        st.warning("No variables in that topic.")
    else:
        row_w = df[df["column"] == opt_to_col_w[picked_w]].iloc[0]
        meta_w = BUCKETS[row_w["bucket"]]

        # --- Slim header (matches tab_detail) -----------------------------
        st.markdown(
            f"<div style='margin:10px 0 4px 0'>"
            f"<div style='font-size:20px;font-weight:700;line-height:1.25;"
            f"color:#111'>{row_w['display_name']}</div>"
            f"<div style='font-size:12px;color:#4b5563;margin-top:2px'>"
            f"<code>{row_w['column']}</code> &middot; {row_w['topic']} &middot; "
            f"<span style='color:{meta_w['color']};font-weight:700'>"
            f"{meta_w['icon']} {meta_w['label']}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # --- Headline numbers in one row ---------------------------------
        miss_pct = float(row_w["pct_missing"]) if pd.notna(row_w["pct_missing"]) else 0.0
        auc_val = row_w.get("missingness_auc")
        auc_str = f"{auc_val:.2f}" if (auc_val is not None and pd.notna(auc_val)) else "n/a"
        bias_val = float(row_w.get("bias_risk", 0.0) or 0.0)

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Reporting rate", f"{100 - miss_pct:.0f}%")
        n2.metric("Missing", f"{miss_pct:.0f}%")
        n3.metric("Predictability AUC", auc_str,
                  help="Cross-validated AUC of a logistic regression that "
                       "predicts whether a row is missing this variable from "
                       "state, year, log-enrollment, locale, and number of "
                       "schools. 0.5 = unpredictable. 1.0 = fully predictable.")
        n4.metric("Bias risk", f"{bias_val:.2f}",
                  help="Approximate fraction of the dataset whose presence "
                       "or absence is systematically tied to district "
                       "characteristics. 2*(AUC-0.5) * pct_missing/100.")

        # --- Narrative summary -------------------------------------------
        st.markdown(
            f"<div style='background:#f8fafc;border-left:4px solid "
            f"{meta_w['color']};padding:12px 16px;margin:10px 0 18px 0;"
            f"font-size:14px;line-height:1.5;color:#1f2937;border-radius:4px'>"
            f"{reporting_narrative(row_w)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        profile = row_w.get("profile")
        if not _has_profile(profile):
            st.info(
                "No profile available for this column (either it has no "
                "missingness or it could not be analyzed)."
            )
        else:
            st.markdown(
                "**Reporting rate by district characteristic.** "
                "Bars show what fraction of each group reports a value for "
                "this variable. Wide spreads = systematic missingness; "
                "uniform bars = effectively random."
            )

            # --- Three profile mini-charts ---------------------------
            pc1, pc2 = st.columns(2)

            # State -- show all states with at least 25 districts.
            with pc1:
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

            # Enrollment quintile -- 5 bars, smallest to largest.
            with pc2:
                eq_records = []
                for r in profile["enrollment_quintile"]:
                    quintile_label = {
                        1: "Q1 smallest", 2: "Q2", 3: "Q3", 4: "Q4",
                        5: "Q5 largest",
                    }.get(int(r["enrollment_quintile"]),
                          f"Q{r['enrollment_quintile']}")
                    eq_records.append({
                        "label": quintile_label,
                        "reported": r["reported"],
                        "count": r["count"],
                    })
                st.plotly_chart(
                    _profile_bar_chart(
                        eq_records,
                        label_key="label",
                        title="By enrollment quintile",
                        height=240,
                        sort_by_rate=False,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                # Locale -- 12 NCES locale categories
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

        with st.expander("Where do these numbers come from?"):
            st.markdown(
                f"""
For each district in the merged ESSER + NCES sample (~14,300 rows after
joining to district characteristics), we mark whether it has a value for
this column. Then we group by **state**, by **enrollment quintile**, and
by **NCES urban-centric locale**, and report the share of each group that
*does* have a value.

The "predictability AUC" comes from a separate cross-validated logistic
regression that uses state, year, log-enrollment, locale, and log of the
number of schools as features. Bias risk combines AUC with the share
missing -- see the **Score formula and methodology** expander on the
**Variable detail** tab for the full formula.

*Caveat:* about 17% of ESSER rows (charter LEAs and certain special
agencies) don't appear in the NCES district-characteristics file and are
excluded from the breakdowns above. Their reporting behavior may differ.
                """
            )
