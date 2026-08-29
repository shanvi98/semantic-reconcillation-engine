"""Rendered components for the dashboard.

Each function returns or renders one self-contained block of markup that
composes from the tokens in ``theme.py``. Nothing here reaches into Streamlit's
default widgets to restyle them ad hoc — where a native widget is the right
tool (the exception table, which needs sorting and keyboard navigation) it is
used as-is and skinned through the stylesheet instead.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from .theme import CATEGORY_STYLES, SOURCE_LABELS, TIER_BLURBS, TIER_COLORS, TIER_LABELS

TIER_ORDER = ["rule", "tfidf", "faiss", "llm"]


def _esc(value) -> str:
    """Escape anything interpolated into markup.

    Reasons and record ids originate in CSV input and in model output, so they
    are untrusted text: rendering them raw inside `unsafe_allow_html` markup
    would let a crafted narration inject markup into the dashboard.
    """
    return html.escape(str(value), quote=True)


def _html(markup: str) -> str:
    """Flatten authored markup into a single line before handing it to Streamlit.

    Streamlit renders through a CommonMark parser, where a raw HTML block ends
    at the first blank line and any line indented four spaces or more becomes a
    code block. Readable, indented templates therefore render as *source text*
    the moment two of them are concatenated — the join puts a whitespace-only
    line between them, closing the HTML block and dumping everything after it
    on screen as code. Flattening here keeps the templates readable without
    letting their indentation reach the parser.

    Lines are joined with a single space because prose wraps mid-sentence in
    these templates; joining with nothing would weld the last word of one line
    onto the first tag of the next.
    """
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def _money(value: float) -> str:
    """Compact currency for display: 1.29M rather than 1,291,651.57.

    Full precision stays in the table and the CSV export — a KPI tile is for
    reading magnitude at a glance, not for reconciling to the paisa.
    """
    value = float(value)
    for cutoff, suffix in ((1e7, "Cr"), (1e5, "L"), (1e3, "K")):
        if abs(value) >= cutoff:
            return f"₹{value / cutoff:,.2f}{suffix}"
    return f"₹{value:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# Hero
# ══════════════════════════════════════════════════════════════════════════════

def hero(*, status: str) -> None:
    chain = ""
    for i, tier in enumerate(TIER_ORDER):
        if i:
            chain += '<span class="arrow">→</span>'
        chain += f'<span class="node">{TIER_LABELS[tier]}</span>'

    st.markdown(
        _html(f"""
        <div class="hero">
          <div class="hero-eyebrow"><span class="dot"></span>{_esc(status)}</div>
          <h1>Semantic Reconciliation Engine</h1>
          <p>
            Three-way finance-ops reconciliation across <strong>Bank Statement</strong>,
            <strong>Razorpay Settlement</strong> and <strong>Invoice</strong>. Each tier sees only
            what the cheaper one before it could not resolve, so the batch closes at the
            lowest cost that will settle it — and whatever survives all four becomes a
            reasoned exception rather than a forced guess.
          </p>
          <div class="chain">{chain}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Section header
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str, caption: str = "") -> None:
    body = f"<p>{_esc(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="section"><h2>{_esc(title)}</h2>{body}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stat cards
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Stat:
    label: str
    value: str
    unit: str = ""
    foot: str = ""
    color: str = "#5B8CFF"
    fill: float | None = None  # 0..1, draws a meter beneath the value


def stat_card(stat: Stat, delay_class: str = "d1") -> str:
    unit = f'<span class="unit">{_esc(stat.unit)}</span>' if stat.unit else ""
    meter = (
        f'<div class="meter"><i style="width:{max(0.0, min(1.0, stat.fill)) * 100:.1f}%"></i></div>'
        if stat.fill is not None else ""
    )
    # foot is composed internally (never user input), so it may carry <b> tags.
    foot = f'<div class="card-foot">{stat.foot}</div>' if stat.foot else ""
    return (
        f'<div class="card rise {delay_class}" style="--rail:{stat.color}">'
        f'<div class="card-label">{_esc(stat.label)}</div>'
        f'<div class="card-value">{_esc(stat.value)}{unit}</div>'
        f"{meter}{foot}</div>"
    )


def kpi_row(stats: list[Stat]) -> None:
    """Render stats as an evenly divided row of cards."""
    columns = st.columns(len(stats), gap="medium")
    for i, (column, stat) in enumerate(zip(columns, stats, strict=True)):
        with column:
            st.markdown(stat_card(stat, f"d{min(i + 1, 8)}"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Score rings
# ══════════════════════════════════════════════════════════════════════════════

def score_ring(value: float, label: str, caption: str, color: str, delay: float = 0.0) -> str:
    """A single conic progress ring, drawn from 0 to `value` on load."""
    radius, stroke = 52, 9
    circumference = 2 * 3.14159265 * radius
    value = max(0.0, min(1.0, value))
    offset = circumference * (1 - value)

    return _html(f"""
    <div class="ring-wrap" style="animation-delay:{delay:.2f}s">
      <div class="ring" style="--rail:{color}">
        <svg viewBox="0 0 118 118" width="118" height="118">
          <circle class="track" cx="59" cy="59" r="{radius}" fill="none" stroke-width="{stroke}"/>
          <circle class="val" cx="59" cy="59" r="{radius}" fill="none" stroke="{color}"
                  stroke-width="{stroke}" stroke-dashoffset="{offset:.2f}"
                  style="--circ:{circumference:.2f}px"/>
        </svg>
        <div class="mid"><b>{value * 100:.1f}%</b><span>{_esc(label)}</span></div>
      </div>
      <div class="ring-cap">{_esc(caption)}</div>
    </div>
    """)


def score_rings(rings: list[str]) -> None:
    st.markdown(f'<div class="rings">{"".join(rings)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tier funnel
# ══════════════════════════════════════════════════════════════════════════════

def tier_funnel(breakdown: dict[str, int], timings: list[dict] | None = None) -> None:
    """Where matches were resolved, as proportional bars.

    Bars are scaled against the busiest tier rather than the total: the shape
    that matters is the drop-off between tiers, and against a total the three
    escalation tiers are usually too small to read at all.
    """
    seconds = {t["tier"]: t.get("seconds", 0.0) for t in (timings or [])}
    peak = max([*breakdown.values(), 1])
    total = sum(breakdown.values()) or 1

    rows = ""
    for i, tier in enumerate(TIER_ORDER):
        count = int(breakdown.get(tier, 0))
        share = count / total * 100
        width = count / peak * 100
        elapsed = seconds.get(tier)
        timing = f"<em>{elapsed * 1000:,.0f} ms</em>" if elapsed is not None else ""
        empty = " funnel-empty" if count == 0 else ""

        rows += _html(f"""
        <div class="funnel-row{empty}" style="--rail:{TIER_COLORS[tier]}">
          <div class="funnel-name">
            <b>{TIER_LABELS[tier]}</b><span>{TIER_BLURBS[tier]}</span>
          </div>
          <div class="funnel-bar">
            <i style="width:{width:.1f}%; --delay:{0.25 + i * 0.12:.2f}s"></i>{timing}
          </div>
          <div class="funnel-count"><b>{count:,}</b><span>{share:.0f}%</span></div>
        </div>""")

    st.markdown(f'<div class="funnel">{rows}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline diagram
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_flow(tier_stats: list[dict], unresolved: int) -> None:
    """Escalation diagram: what each tier absorbed and what it passed on.

    Drawn rather than charted because the quantity is not the point — the
    *shape* is. Each tier is entered by everything the previous one declined,
    so the diagram is a staircase, and the animated dashes run along it in the
    direction records actually travel.

    Every figure here is a measurement the engine took, not one inferred from
    the match counts. An earlier version derived the entry counts by summing
    matches and working backwards, which produced a staircase that looked right
    and was wrong: it could not know that a leg exits the cascade early once one
    side is exhausted, so it overstated what reached the later tiers.
    """
    box_w, box_h, gap = 190, 74, 78
    tail_w, pad = 86, 12
    y, height = 58, 190
    # Derived, not hard-coded: a fixed viewBox silently clips the residual badge
    # the moment a tier is added or a box is resized.
    width = pad + len(TIER_ORDER) * (box_w + gap) + tail_w + pad

    by_tier = {stat.get("tier"): stat for stat in (tier_stats or [])}
    matched_by = {t: int(by_tier.get(t, {}).get("matches", 0)) for t in TIER_ORDER}
    entering_by = {t: int(by_tier.get(t, {}).get("candidates_in", 0)) for t in TIER_ORDER}

    # A metrics file written before candidate counts were recorded has nothing to
    # draw from. Falling back to the old derivation keeps the diagram legible
    # rather than showing a row of zeros; a current run never takes this path.
    if not any(entering_by.values()):
        running = sum(matched_by.values()) + unresolved
        for tier in TIER_ORDER:
            entering_by[tier] = running
            running -= matched_by[tier]

    nodes = ""
    links = ""

    for i, tier in enumerate(TIER_ORDER):
        x = 12 + i * (box_w + gap)
        color = TIER_COLORS[tier]
        matched = matched_by[tier]
        entering = entering_by[tier]
        dim = "" if matched else ' opacity="0.45"'

        nodes += _html(f"""
        <g class="node-box"{dim}>
          <rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="14"
                fill="{color}14" stroke="{color}" stroke-opacity="0.5" stroke-width="1"/>
          <text x="{x + 16}" y="{y + 27}" fill="#E8ECF6" font-size="14" font-weight="650">{TIER_LABELS[tier]}</text>
          <text x="{x + 16}" y="{y + 46}" fill="#9AA7C2" font-size="11">{TIER_BLURBS[tier]}</text>
          <text x="{x + box_w - 16}" y="{y + 33}" fill="{color}" font-size="19"
                font-weight="720" text-anchor="end">{matched:,}</text>
          <text x="{x + box_w - 16}" y="{y + 49}" fill="#64748B" font-size="10" text-anchor="end">matched</text>
        </g>
        <text x="{x + box_w / 2}" y="{y - 14}" fill="#64748B" font-size="10.5"
              text-anchor="middle">{entering:,} in</text>""")

        if i < len(TIER_ORDER) - 1:
            x1, x2 = x + box_w, x + box_w + gap
            mid = y + box_h / 2
            links += _html(f"""
            <path class="link" d="M {x1} {mid} L {x2} {mid}" stroke="{color}"
                  stroke-opacity="0.55" stroke-width="2" fill="none"
                  style="animation-delay:{i * 0.4:.1f}s"/>
            <path d="M {x2 - 9} {mid - 5} L {x2 - 1} {mid} L {x2 - 9} {mid + 5}"
                  stroke="{color}" stroke-opacity="0.75" stroke-width="1.6" fill="none"
                  stroke-linecap="round" stroke-linejoin="round"/>
            <text x="{(x1 + x2) / 2}" y="{mid - 12}" fill="#64748B" font-size="10"
                  text-anchor="middle">{entering_by[TIER_ORDER[i + 1]]:,}</text>""")

    tail_x = pad + len(TIER_ORDER) * (box_w + gap) - gap + pad
    residual_color = "#FB7185" if unresolved else "#34D399"
    tail_label = "unresolved" if unresolved else "all resolved"

    st.markdown(
        _html(f"""
        <div class="flow">
          <svg viewBox="0 0 {width} {height}" role="img"
               aria-label="Escalation: {entering_by[TIER_ORDER[0]]} record pairs enter the Rule tier and
                           cascade through TF-IDF, FAISS and LLM, leaving {unresolved} unresolved.">
            {links}
            {nodes}
            <g>
              <rect x="{tail_x}" y="{y + 12}" width="{tail_w}" height="50" rx="12"
                    fill="{residual_color}14" stroke="{residual_color}" stroke-opacity="0.5"
                    stroke-dasharray="4 4"/>
              <text x="{tail_x + tail_w / 2}" y="{y + 36}" fill="{residual_color}" font-size="17"
                    font-weight="700" text-anchor="middle">{unresolved:,}</text>
              <text x="{tail_x + tail_w / 2}" y="{y + 52}" fill="#64748B" font-size="9.5"
                    text-anchor="middle">{tail_label}</text>
            </g>
          </svg>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Exception queue
# ══════════════════════════════════════════════════════════════════════════════

def category_legend(counts: dict[str, int]) -> None:
    pills = ""
    for category, count in counts.items():
        label, color = CATEGORY_STYLES.get(category, (category.replace("_", " ").title(), "#9AA7C2"))
        pills += f'<span class="pill" style="--pc:{color}"><span class="dot"></span>{_esc(label)} · {count}</span>'
    st.markdown(f'<div class="legend">{pills}</div>', unsafe_allow_html=True)


def exception_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Filterable review queue. Returns the filtered frame for export."""
    if df.empty:
        st.markdown(
            '<div class="empty"><div class="mark">✓</div><h3>Nothing to review</h3>'
            "<p>Every record closed into a complete triangle.</p></div>",
            unsafe_allow_html=True,
        )
        return df

    display = df.copy()
    display["source"] = display["source"].map(lambda s: SOURCE_LABELS.get(s, s))
    display["category"] = display["category"].map(
        lambda c: CATEGORY_STYLES.get(c, (c.replace("_", " ").title(), ""))[0]
    )

    filters = st.columns([2, 2, 3], gap="medium")
    sources = sorted(display["source"].dropna().unique().tolist())
    categories = sorted(display["category"].dropna().unique().tolist())

    picked_sources = filters[0].multiselect("Source", sources, default=sources)
    picked_categories = filters[1].multiselect("Category", categories, default=categories)
    query = filters[2].text_input("Search", placeholder="record id, related record, or reason…")

    filtered = display[
        display["source"].isin(picked_sources) & display["category"].isin(picked_categories)
    ]
    if query:
        haystack = (
            filtered["reason"].fillna("")
            + " " + filtered["record_id"].astype(str)
            + " " + filtered["best_candidate_id"].fillna("").astype(str)
        )
        filtered = filtered[haystack.str.contains(query, case=False, na=False, regex=False)]

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
        column_config={
            "record_id": st.column_config.TextColumn("Record", width="small"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "category": st.column_config.TextColumn("Category", width="small"),
            "reason": st.column_config.TextColumn("Why it did not close", width="large"),
            "amount": st.column_config.NumberColumn("At risk", format="₹%.2f", width="small"),
            "best_candidate_id": st.column_config.TextColumn("Related record", width="small"),
            "best_candidate_score": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f", width="small"
            ),
        },
    )

    at_risk = float(filtered["amount"].sum())
    st.markdown(
        f'<div class="note">Showing <b>{len(filtered):,}</b> of <b>{len(display):,}</b> exceptions · '
        f"<b>{_money(at_risk)}</b> of value in view. Every record the engine could not close "
        "appears here with the reason it fell through all four tiers — nothing is dropped silently.</div>",
        unsafe_allow_html=True,
    )
    return filtered


__all__ = [
    "Stat",
    "category_legend",
    "exception_queue",
    "hero",
    "kpi_row",
    "pipeline_flow",
    "score_ring",
    "score_rings",
    "section",
    "stat_card",
    "tier_funnel",
]
