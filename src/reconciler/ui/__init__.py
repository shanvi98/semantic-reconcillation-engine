"""Presentation layer for the Streamlit dashboard.

Kept out of app.py so the design system (tokens, CSS, components) is a real
module rather than a wall of inline markup, and so each visual component can be
rendered and inspected on its own.
"""
from .components import (
    exception_queue,
    hero,
    kpi_row,
    pipeline_flow,
    score_ring,
    section,
    stat_card,
    tier_funnel,
)
from .theme import inject_theme

__all__ = [
    "exception_queue",
    "hero",
    "inject_theme",
    "kpi_row",
    "pipeline_flow",
    "score_ring",
    "section",
    "stat_card",
    "tier_funnel",
]
