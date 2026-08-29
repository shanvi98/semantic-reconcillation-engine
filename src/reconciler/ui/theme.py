"""Design tokens and stylesheet for the dashboard.

One place defines the whole visual language — palette, type scale, spacing,
elevation, easing — and every component composes from it. That is the
difference between a designed interface and an accumulation of one-off inline
styles: a change to the accent ramp or the corner radius lands everywhere at
once, and nothing drifts.

Motion is used to carry meaning rather than for decoration. Numbers rise into
place because they are the result of a computation that just finished; the
funnel's flow animates along the direction records actually travel through the
tiers; a card lifts on hover because it is interactive. Everything is
suppressed entirely under `prefers-reduced-motion`.
"""
from __future__ import annotations

import streamlit as st

# --- Tier identity -----------------------------------------------------------
# The ramp runs cool to warm in the direction of escalation, so "how expensive
# was this batch to resolve" is legible from colour alone before reading a
# single number. Rule is calm green; the LLM tier — the one that costs money and
# latency — is the only warm colour on the page.
TIER_COLORS = {
    "rule": "#34D399",
    "tfidf": "#38BDF8",
    "faiss": "#A78BFA",
    "llm": "#FBBF24",
}

TIER_LABELS = {"rule": "Rule", "tfidf": "TF-IDF", "faiss": "FAISS", "llm": "LLM"}

TIER_BLURBS = {
    "rule": "Exact ID / UTR overlap",
    "tfidf": "Char n-gram lexical",
    "faiss": "Dense semantic",
    "llm": "Model adjudication",
}

CATEGORY_STYLES = {
    "pending_settlement": ("Pending settlement", "#38BDF8"),
    "in_transit": ("In transit", "#A78BFA"),
    "unmatched_settlement": ("Unmatched settlement", "#FB7185"),
    "unmatched_bank_credit": ("Unidentified credit", "#FBBF24"),
    "partial_match": ("Partial match", "#F472B6"),
}

SOURCE_LABELS = {"invoice": "Invoice", "razorpay": "Settlement", "bank": "Bank"}


CSS = """
<style>
/* ══════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS
   ══════════════════════════════════════════════════════════════════════════ */
:root {
  /* Surfaces: a four-step elevation ramp, not arbitrary greys. */
  --bg:            #0A0E1A;
  --surface-1:     #101627;
  --surface-2:     #161E33;
  --surface-3:     #1D2740;
  --hairline:      rgba(148, 163, 184, 0.14);
  --hairline-soft: rgba(148, 163, 184, 0.08);

  /* Content: three weights only. More than three and hierarchy stops reading. */
  --text:      #E8ECF6;
  --text-mute: #9AA7C2;
  --text-dim:  #64748B;

  --accent:      #5B8CFF;
  --accent-soft: rgba(91, 140, 255, 0.14);
  --accent-line: rgba(91, 140, 255, 0.40);

  --ok:    #34D399;
  --warn:  #FBBF24;
  --risk:  #FB7185;
  --info:  #38BDF8;
  --deep:  #A78BFA;

  /* Type scale: 1.25 ratio from a 14px base. */
  --t-xs:  0.6875rem;
  --t-sm:  0.8125rem;
  --t-md:  0.9375rem;
  --t-lg:  1.25rem;
  --t-xl:  1.75rem;
  --t-2xl: 2.5rem;
  --t-3xl: 3.5rem;

  /* Spacing: 4px base, doubling. */
  --s-1: 0.25rem; --s-2: 0.5rem;  --s-3: 0.75rem;
  --s-4: 1rem;    --s-5: 1.5rem;  --s-6: 2rem;   --s-8: 3rem;

  --r-sm: 8px; --r-md: 14px; --r-lg: 20px; --r-full: 999px;

  --shadow-1: 0 1px 2px rgba(0,0,0,.30);
  --shadow-2: 0 8px 24px -8px rgba(0,0,0,.55);
  --shadow-3: 0 24px 56px -20px rgba(0,0,0,.70);
  --glow:     0 0 0 1px var(--accent-line), 0 12px 40px -12px rgba(91,140,255,.45);

  /* Easings. --ease-out for entrances, --ease-spring for anything a
     pointer touches, so interaction feels physical rather than linear. */
  --ease-out:    cubic-bezier(.22, 1, .36, 1);
  --ease-spring: cubic-bezier(.34, 1.56, .64, 1);

  --mono: ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Code", Menlo, monospace;
}

/* ══════════════════════════════════════════════════════════════════════════
   APP CHROME
   ══════════════════════════════════════════════════════════════════════════ */
.stApp {
  background:
    radial-gradient(1200px 700px at 12% -12%, rgba(91,140,255,.16), transparent 60%),
    radial-gradient(1000px 600px at 92% 4%,  rgba(167,139,250,.13), transparent 55%),
    radial-gradient(900px 500px at 50% 105%, rgba(52,211,153,.07), transparent 60%),
    var(--bg);
  background-attachment: fixed;
  color: var(--text);
}

.block-container { padding-top: var(--s-5) !important; padding-bottom: var(--s-8) !important; max-width: 1400px; }

/* Streamlit's default heading rhythm is too loose for a dense dashboard. */
h1, h2, h3 { letter-spacing: -0.022em; color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: var(--r-full); border: 2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: #2A3757; }

/* ══════════════════════════════════════════════════════════════════════════
   MOTION
   ══════════════════════════════════════════════════════════════════════════ */
@keyframes rise      { from { opacity:0; transform: translate3d(0,18px,0); filter: blur(6px); }
                       to   { opacity:1; transform: none;                  filter: blur(0);   } }
@keyframes fade      { from { opacity:0 } to { opacity:1 } }
@keyframes sweep     { from { transform: translateX(-120%) } to { transform: translateX(220%) } }
@keyframes grow      { from { transform: scaleX(0) } to { transform: scaleX(1) } }
@keyframes drift     { 0%,100% { transform: translate(0,0) scale(1) }
                       33%     { transform: translate(3%,-4%) scale(1.06) }
                       66%     { transform: translate(-3%,3%) scale(.96) } }
@keyframes flow      { to { stroke-dashoffset: -1000 } }
@keyframes pulse     { 0%,100% { opacity:.55 } 50% { opacity:1 } }
@keyframes ringdraw  { from { stroke-dashoffset: var(--circ) } }

/* Entrance choreography. Content arrives in reading order rather than all at
   once, which makes a dense page legible as it lands instead of after. */
.rise { animation: rise .7s var(--ease-out) both; }
.d1{animation-delay:.05s}.d2{animation-delay:.11s}.d3{animation-delay:.17s}
.d4{animation-delay:.23s}.d5{animation-delay:.29s}.d6{animation-delay:.35s}
.d7{animation-delay:.41s}.d8{animation-delay:.47s}

/* ══════════════════════════════════════════════════════════════════════════
   HERO
   ══════════════════════════════════════════════════════════════════════════ */
.hero {
  position: relative; overflow: hidden;
  border: 1px solid var(--hairline);
  border-radius: var(--r-lg);
  background: linear-gradient(150deg, var(--surface-2), var(--surface-1) 55%, var(--bg));
  padding: var(--s-6) var(--s-6) var(--s-5);
  margin-bottom: var(--s-5);
  box-shadow: var(--shadow-3);
  animation: rise .8s var(--ease-out) both;
}
/* Two counter-drifting colour fields. Slow enough (22s/28s) to read as ambient
   depth rather than as something demanding attention. */
.hero::before, .hero::after {
  content: ""; position: absolute; inset: -45%;
  pointer-events: none; z-index: 0;
}
.hero::before {
  background: radial-gradient(closest-side, rgba(91,140,255,.30), transparent 70%);
  animation: drift 22s ease-in-out infinite;
}
.hero::after {
  background: radial-gradient(closest-side, rgba(167,139,250,.24), transparent 70%);
  animation: drift 28s ease-in-out infinite reverse;
}
.hero > * { position: relative; z-index: 1; }

.hero-eyebrow {
  display:inline-flex; align-items:center; gap: var(--s-2);
  font-size: var(--t-xs); font-weight: 650; letter-spacing: .16em; text-transform: uppercase;
  color: var(--accent);
  background: var(--accent-soft); border: 1px solid var(--accent-line);
  padding: 5px 12px; border-radius: var(--r-full); margin-bottom: var(--s-4);
}
.hero-eyebrow .dot {
  width:6px; height:6px; border-radius:50%; background: var(--ok);
  box-shadow: 0 0 10px var(--ok); animation: pulse 2.4s ease-in-out infinite;
}
.hero h1 {
  font-size: var(--t-3xl); font-weight: 760; line-height: 1.04;
  letter-spacing: -0.035em; margin: 0 0 var(--s-3);
  background: linear-gradient(96deg, #FFFFFF 8%, #C7D6FF 46%, #9BB4FF 76%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero p { color: var(--text-mute); font-size: var(--t-md); max-width: 68ch; margin: 0; line-height: 1.6; }
.hero strong { color: var(--text); font-weight: 620; }

.chain {
  display:flex; align-items:center; flex-wrap: wrap; gap: var(--s-2);
  margin-top: var(--s-5); font-size: var(--t-sm);
}
.chain .node {
  padding: 7px 14px; border-radius: var(--r-full);
  background: rgba(255,255,255,.05); border: 1px solid var(--hairline);
  color: var(--text); font-weight: 560;
  transition: transform .3s var(--ease-spring), border-color .3s, background .3s;
}
.chain .node:hover { transform: translateY(-2px); border-color: var(--accent-line); background: var(--accent-soft); }
.chain .arrow { color: var(--text-dim); font-size: var(--t-sm); }

/* ══════════════════════════════════════════════════════════════════════════
   CARDS
   ══════════════════════════════════════════════════════════════════════════ */
.card {
  position: relative; overflow: hidden;
  background: linear-gradient(168deg, var(--surface-2), var(--surface-1));
  border: 1px solid var(--hairline);
  border-radius: var(--r-md);
  padding: var(--s-5) var(--s-4) var(--s-4);
  box-shadow: var(--shadow-2);
  height: 100%;
  transition: transform .35s var(--ease-spring), border-color .35s, box-shadow .35s;
}
.card:hover { transform: translateY(-4px); border-color: var(--accent-line); box-shadow: var(--glow); }
/* A light sweep that only runs on hover — a persistent shimmer on every card
   would compete with the data for attention. */
.card::after {
  content:""; position:absolute; top:0; bottom:0; width: 40%;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.07), transparent);
  transform: translateX(-120%); pointer-events:none;
}
.card:hover::after { animation: sweep 1.1s var(--ease-out); }
/* Accent rail: the card's semantic colour, drawn on the leading edge. */
.card::before {
  content:""; position:absolute; left:0; top:0; bottom:0; width: 3px;
  background: var(--rail, var(--accent));
  transform: scaleY(0); transform-origin: top;
  animation: grow .6s var(--ease-out) .25s both;
  animation-name: railgrow;
}
@keyframes railgrow { to { transform: scaleY(1) } }

.card-label {
  display:flex; align-items:center; gap: var(--s-2);
  font-size: var(--t-xs); font-weight: 640; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text-dim); margin-bottom: var(--s-3);
}
/* Tabular figures so digits stay column-aligned as values change. */
.card-value {
  font-size: var(--t-xl); font-weight: 700; line-height: 1;
  letter-spacing: -0.03em; color: var(--text);
  font-variant-numeric: tabular-nums;
  animation: rise .75s var(--ease-out) .18s both;
}
.card-value .unit { font-size: var(--t-md); font-weight: 560; color: var(--text-mute); margin-left: 3px; }
.card-foot { margin-top: var(--s-3); font-size: var(--t-xs); color: var(--text-dim); line-height: 1.5; }
.card-foot b { color: var(--text-mute); font-weight: 600; }

/* Meter: fills from zero to its real value, so the bar itself reads as the
   measurement being taken rather than as a static decoration. */
.meter { height: 4px; border-radius: var(--r-full); background: rgba(255,255,255,.07); margin-top: var(--s-3); overflow:hidden; }
.meter > i {
  display:block; height:100%; border-radius: inherit;
  background: linear-gradient(90deg, var(--rail, var(--accent)), color-mix(in srgb, var(--rail, var(--accent)) 55%, white));
  transform-origin: left; animation: grow .9s var(--ease-out) .3s both;
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════════════════════════════════════════ */
.section { margin: var(--s-8) 0 var(--s-4); animation: rise .6s var(--ease-out) both; }
.section h2 {
  font-size: var(--t-lg); font-weight: 680; margin: 0 0 var(--s-2);
  display:flex; align-items:center; gap: var(--s-3);
}
.section h2::after { content:""; flex:1; height:1px; background: linear-gradient(90deg, var(--hairline), transparent); }
.section p { color: var(--text-mute); font-size: var(--t-sm); margin: 0; max-width: 78ch; line-height: 1.6; }

/* ══════════════════════════════════════════════════════════════════════════
   SCORE RING
   ══════════════════════════════════════════════════════════════════════════ */
.rings { display:flex; gap: var(--s-5); flex-wrap: wrap; }
.ring-wrap { display:flex; flex-direction:column; align-items:center; gap: var(--s-2); animation: rise .7s var(--ease-out) both; }
.ring { position: relative; width: 118px; height: 118px; }
.ring svg { transform: rotate(-90deg); overflow: visible; }
.ring .track { stroke: rgba(255,255,255,.07); }
.ring .val {
  stroke-linecap: round;
  stroke-dasharray: var(--circ);
  animation: ringdraw 1.5s var(--ease-out) .3s both;
  filter: drop-shadow(0 0 8px var(--rail));
}
.ring .mid {
  position:absolute; inset:0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:1px;
}
.ring .mid b { font-size: 1.5rem; font-weight: 720; letter-spacing:-.03em; font-variant-numeric: tabular-nums; }
.ring .mid span { font-size: var(--t-xs); color: var(--text-dim); text-transform:uppercase; letter-spacing:.1em; }
.ring-cap { font-size: var(--t-xs); color: var(--text-mute); text-align:center; max-width: 20ch; line-height:1.45; }

/* ══════════════════════════════════════════════════════════════════════════
   TIER FUNNEL
   ══════════════════════════════════════════════════════════════════════════ */
.funnel {
  background: linear-gradient(168deg, var(--surface-2), var(--surface-1));
  border: 1px solid var(--hairline); border-radius: var(--r-md);
  padding: var(--s-5); box-shadow: var(--shadow-2);
  animation: rise .7s var(--ease-out) both;
}
.funnel-row { display:grid; grid-template-columns: 132px 1fr 90px; gap: var(--s-4); align-items:center; padding: var(--s-3) 0; }
.funnel-row + .funnel-row { border-top: 1px solid var(--hairline-soft); }
.funnel-name { display:flex; flex-direction:column; gap:2px; }
.funnel-name b { font-size: var(--t-sm); font-weight: 640; color: var(--text); }
.funnel-name span { font-size: var(--t-xs); color: var(--text-dim); }
.funnel-bar { height: 30px; border-radius: var(--r-sm); background: rgba(255,255,255,.04); position:relative; overflow:hidden; }
.funnel-bar > i {
  position:absolute; inset:0 auto 0 0; border-radius: inherit;
  background: linear-gradient(90deg, color-mix(in srgb, var(--rail) 70%, transparent), var(--rail));
  transform-origin:left; animation: grow 1s var(--ease-out) var(--delay, .3s) both;
  box-shadow: 0 0 20px -4px var(--rail);
}
.funnel-bar em {
  position:absolute; right: 10px; top:50%; transform: translateY(-50%);
  font-style:normal; font-size: var(--t-xs); color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}
.funnel-count { text-align:right; font-variant-numeric: tabular-nums; }
.funnel-count b { font-size: var(--t-lg); font-weight: 700; color: var(--text); }
.funnel-count span { display:block; font-size: var(--t-xs); color: var(--text-dim); }
.funnel-empty { opacity: .42; }

/* ══════════════════════════════════════════════════════════════════════════
   PIPELINE DIAGRAM
   ══════════════════════════════════════════════════════════════════════════ */
.flow { padding: var(--s-4) 0; animation: rise .7s var(--ease-out) .1s both; }
.flow svg { width: 100%; height: auto; overflow: visible; }
/* Dashes travel along each connector in the direction records actually move
   through the pipeline, so the diagram animates the process it depicts. */
.flow .link { stroke-dasharray: 5 9; animation: flow 14s linear infinite; }
.flow .node-box { transition: opacity .3s; }
.flow text { font-family: inherit; }

/* ══════════════════════════════════════════════════════════════════════════
   EXCEPTION QUEUE
   ══════════════════════════════════════════════════════════════════════════ */
.pill {
  display:inline-flex; align-items:center; gap: 6px;
  padding: 3px 10px; border-radius: var(--r-full);
  font-size: var(--t-xs); font-weight: 620; letter-spacing:.01em;
  background: color-mix(in srgb, var(--pc) 15%, transparent);
  border: 1px solid color-mix(in srgb, var(--pc) 38%, transparent);
  color: var(--pc); white-space: nowrap;
}
.pill .dot { width:5px; height:5px; border-radius:50%; background: currentColor; }

.legend { display:flex; gap: var(--s-2); flex-wrap:wrap; margin-bottom: var(--s-3); }

/* Streamlit's own dataframe, restyled rather than replaced: it keeps sorting,
   resizing and keyboard navigation, which a hand-rolled table would lose. */
[data-testid="stDataFrame"] {
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-md) !important;
  overflow: hidden;
  box-shadow: var(--shadow-2);
  animation: rise .7s var(--ease-out) .1s both;
}
[data-testid="stDataFrame"] * { font-variant-numeric: tabular-nums; }

/* ══════════════════════════════════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--surface-1), var(--bg));
  border-right: 1px solid var(--hairline);
}
section[data-testid="stSidebar"] .block-container { padding-top: var(--s-5); }

.side-title { font-size: var(--t-xs); font-weight: 680; letter-spacing:.14em; text-transform:uppercase; color: var(--text-dim); margin: var(--s-5) 0 var(--s-2); }
.side-kv { display:flex; justify-content:space-between; gap: var(--s-3); padding: 7px 0; border-bottom: 1px solid var(--hairline-soft); font-size: var(--t-sm); }
.side-kv span { color: var(--text-dim); }
.side-kv b { color: var(--text); font-weight: 580; font-family: var(--mono); font-size: var(--t-xs); }

.stButton > button {
  width: 100%; border-radius: var(--r-sm) !important;
  font-weight: 640 !important; letter-spacing: .01em;
  border: 1px solid var(--accent-line) !important;
  background: linear-gradient(180deg, var(--accent), #4171E8) !important;
  color: #fff !important;
  box-shadow: 0 6px 18px -8px rgba(91,140,255,.8) !important;
  transition: transform .25s var(--ease-spring), box-shadow .25s, filter .25s !important;
}
.stButton > button:hover { transform: translateY(-2px); filter: brightness(1.08); box-shadow: 0 12px 28px -10px rgba(91,140,255,.95) !important; }
.stButton > button:active { transform: translateY(0) scale(.985); }

/* ══════════════════════════════════════════════════════════════════════════
   EMPTY STATE
   ══════════════════════════════════════════════════════════════════════════ */
.empty {
  text-align:center; padding: var(--s-8) var(--s-5);
  border: 1px dashed var(--hairline); border-radius: var(--r-lg);
  background: linear-gradient(168deg, var(--surface-1), transparent);
  animation: rise .7s var(--ease-out) both;
}
.empty .mark { font-size: 2.5rem; margin-bottom: var(--s-3); animation: pulse 3s ease-in-out infinite; }
.empty h3 { font-size: var(--t-lg); margin: 0 0 var(--s-2); font-weight: 660; }
.empty p { color: var(--text-mute); font-size: var(--t-sm); margin: 0 auto; max-width: 52ch; line-height:1.6; }
.empty code { background: var(--surface-3); padding: 2px 7px; border-radius: 5px; font-family: var(--mono); font-size: var(--t-xs); color: var(--info); }

.note {
  border-left: 2px solid var(--accent); background: var(--accent-soft);
  padding: var(--s-3) var(--s-4); border-radius: 0 var(--r-sm) var(--r-sm) 0;
  font-size: var(--t-sm); color: var(--text-mute); line-height: 1.6;
  animation: fade .6s var(--ease-out) both;
}
.note b { color: var(--text); font-weight: 620; }
/* Amber variant, for a caveat about the numbers rather than a note about them. */
.note.warn { border-left-color: var(--warn); background: rgba(251, 191, 36, 0.10); }

/* ══════════════════════════════════════════════════════════════════════════
   RESPONSIVE
   ══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 900px) {
  :root { --t-3xl: 2.25rem; --t-2xl: 1.75rem; }
  .funnel-row { grid-template-columns: 96px 1fr 64px; gap: var(--s-3); }
  .hero { padding: var(--s-5) var(--s-4); }
}

/* ══════════════════════════════════════════════════════════════════════════
   REDUCED MOTION
   Everything above is decoration over a layout that already works. Under this
   query the page is fully static — no residual drift, no partial fills.
   ══════════════════════════════════════════════════════════════════════════ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .001ms !important;
  }
  .card:hover { transform: none; }
}
</style>
"""


def inject_theme() -> None:
    """Install the stylesheet. Call once, before rendering anything else."""
    st.markdown(CSS, unsafe_allow_html=True)
