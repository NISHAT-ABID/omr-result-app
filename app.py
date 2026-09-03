"""
app.py
------
OMR Result App - main Streamlit application.

Roles:
- Student: sign up / log in with phone + password, submit OMR sheets,
  see results, test history, leaderboard, profile.
- Mentor: set answer keys, calibrate the sheet, manage students,
  edit/override results, export data, view analytics.

Run with: streamlit run app.py
"""

import random
import io
from datetime import datetime, date, time as dtime, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

# ================= FINAL OMR REVIEW BUILD =================
# Original OMR photo + full Digital OMR + immutable double-touch audit +
# compact mobile tables. Existing exam/OMR features are intentionally preserved.
OMR_REVIEW_BUILD = "2026-08-31-final"
from streamlit_image_coordinates import streamlit_image_coordinates

import omr_scanner
import sheets_helper as sh

st.set_page_config(page_title="The Med Venture — by Bushra", page_icon="🩺", layout="wide")

# =========================================================================
# Brand — The Med Venture (by Bushra)
# A small, reusable logo mark + header block used on the entry screens.
# Pure presentation: no session/state/logic lives here.
# =========================================================================

LOGO_SVG = """
<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block;">
  <rect x="2" y="2" width="60" height="60" rx="16" fill="#123C39"/>
  <path d="M32 15 L32 49 M15 32 L49 32" stroke="#F1F4F0" stroke-width="7" stroke-linecap="round"/>
  <circle cx="47" cy="47" r="6.5" fill="#C4432E"/>
  <circle cx="47" cy="47" r="2.2" fill="#F1F4F0"/>
</svg>
"""


def render_hero(eyebrow, heading_html=None, tagline=None, compact=False, pulse=True,
                 show_badge=True, show_byline=True):
    """Full hero-style entry header - mirrors the Med Venture web app's
    role-selection screen: dotted background, logo, eyebrow badge, big
    serif heading, optional tagline, and the animated pulse-line. Used on
    every entry/gate screen (password gate, student login, mentor login)
    so the whole app opens the same way the web app does. Presentation
    only - no session/state/logic lives here.

    show_badge / show_byline let a caller drop the small eyebrow pill and
    the "By Bushra" line entirely (used on the Student/Mentor login
    screens, where "Student Portal"/"Mentor Portal" doesn't apply until
    *after* the password gate, and the byline is redundant there) without
    touching the password-gate screen, which still shows both by default.
    compact=True also tightens the whole block's padding/heading size a
    step further than before, so the heading alone doesn't push the login
    card below the fold on mobile.
    """
    logo_size = 34 if compact else 52
    heading = heading_html or "The Med <span style='color:var(--mv-accent);font-style:italic;'>Venture</span>"
    badge_html = f'<div class="mv-hero-badge">{eyebrow}</div>' if show_badge else ""
    byline_html = (
        '<div style="font-family:var(--sans);font-size:11px;letter-spacing:.08em;'
        'text-transform:uppercase;color:var(--mv-muted);margin-top:2px;">By Bushra</div>'
        if show_byline else ""
    )
    tag_html = (
        f"<p style='font-family:var(--sans);color:var(--mv-muted);font-size:14px;"
        f"max-width:420px;margin:8px auto 0;line-height:1.55;'>{tagline}</p>"
        if tagline else ""
    )
    pulse_html = f"""
        <svg viewBox="0 0 400 40" preserveAspectRatio="none"
             style="width:100%;max-width:260px;height:24px;color:var(--mv-accent);opacity:.6;margin:16px auto 2px;display:block;">
            <path class="mv-hero-pulse-path" d="M0,20 L110,20 L128,4 L145,36 L162,20 L400,20" fill="none"
                  stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """ if pulse else ""
    hero_class = "mv-hero mv-hero-compact" if compact else "mv-hero"
    parts = [
        f'<div class="{hero_class}">',
        f'<div style="width:{logo_size}px;height:{logo_size}px;margin:0 auto 12px;">{LOGO_SVG}</div>',
        badge_html,
        f'<h1 style="font-family:var(--serif);font-weight:600;'
        f'font-size:{"19px" if compact else "clamp(26px,5vw,36px)"};'
        f'margin:0 0 2px;letter-spacing:-0.01em;color:var(--mv-ink);line-height:1.12;">{heading}</h1>',
        byline_html,
    ]
    if tag_html:
        parts.append(tag_html)
    if pulse_html:
        parts.append(pulse_html)
    parts.append("</div>")
    # Joined with no newlines between parts - a blank/whitespace-only line
    # inside a raw HTML block makes Streamlit's markdown parser treat what
    # follows as plain text instead of HTML (that's what was leaking a
    # literal "</div>" onto the page whenever tag_html/pulse_html were
    # empty), so this avoids the bug entirely rather than working around it.
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_boot_loading_screen(message="Connecting..."):
    """Full-screen branded loading state, shown only while the app talks to
    Google Sheets for the first time in a session. Presentation only - the
    actual init_sheets() call and the '_sheets_ready' flag it guards are
    unchanged; this just replaces the plain default st.spinner() with the
    same pulse-line hero look used on the Med Venture web app."""
    st.markdown(
        f"""
        <div style="text-align:center; padding:76px 20px 30px;">
            <div style="width:60px; height:60px; margin:0 auto 16px;">{LOGO_SVG}</div>
            <div style="font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:25px; color:#123C39; margin-bottom:2px;">
                The Med Venture
            </div>
            <div style="font-family:'IBM Plex Sans',sans-serif; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#7C8B83; margin-bottom:26px;">
                by Bushra
            </div>
            <svg class="mv-boot-pulse" viewBox="0 0 400 40" preserveAspectRatio="none"
                 style="width:100%; max-width:260px; height:26px; color:#C4432E;">
                <path d="M0,20 L110,20 L128,4 L145,36 L162,20 L400,20" fill="none"
                      stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"
                      style="stroke-dasharray:520; stroke-dashoffset:520;"/>
            </svg>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#7C8B83; margin-top:16px; letter-spacing:.05em;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# Global styling - one shared stylesheet for the whole app (mobile + desktop)
# =========================================================================

def inject_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        /* One fixed palette, always applied - deliberately NOT gated behind
           @media (prefers-color-scheme: dark) any more. The app must look
           identical no matter what light/dark mode the visitor's OS or
           browser is set to, so we hardcode this single dark, teal Med
           Venture palette as the only palette that ever exists. */
        :root {
            color-scheme: dark;
            /* Colors below were pixel-sampled directly from the reference
               design image (not eyeballed) for an exact match:
               --mv-bg / --mv-surface: page and card background sampled at
               #061112 / #0E1C1C. --mv-primary: the avatar-circle fill and
               positive metric numbers (Marks/Correct/Average) sampled at
               #26AB8C. --mv-accent: the icon's orange flourish sampled at
               #F94D10. --mv-nav-active-bg is a NEW token (didn't exist
               before) specifically for the active nav pill, which the
               reference renders as a dark, muted teal-charcoal fill
               (#142D2A) rather than a solid bright --mv-primary block -
               using --mv-primary directly there would have been too
               vivid/flat compared to the reference's more subdued look. */
            --mv-bg: #061112;
            --mv-surface: #0E1C1C;
            --mv-ink: #EAF2EF;
            --mv-primary: #26AB8C;
            --mv-primary-hover: #34C29F;
            --mv-primary-soft: rgba(38,171,140,0.20);
            --mv-nav-active-bg: #142D2A;
            --mv-accent: #F94D10;
            --mv-accent-soft: rgba(249,77,16,0.18);
            /* Extra accent colors used for varied icon-chip backgrounds
               (Profile Information rows, the stats strip) so each stat
               reads as visually distinct rather than everything sharing
               one or two tones - matches the reference design's mixed
               teal/blue/gold/purple icon palette. */
            --mv-blue: #3B82F6;
            --mv-blue-soft: rgba(59,130,246,0.20);
            --mv-purple: #8B5CF6;
            --mv-purple-soft: rgba(139,92,246,0.20);
            --mv-danger: #F2434A;
            --mv-danger-soft: rgba(242,67,74,0.16);
            --mv-muted: #9BAAA2;
            --mv-border: rgba(230,240,235,0.14);
            --mv-card-bg: rgba(230,240,235,0.05);
            --mv-dot: rgba(230,240,235,0.10);
            --mv-input-bg: #071615;
            --surface: var(--mv-surface);
            --serif: 'Fraunces', Georgia, serif;
            --sans: 'IBM Plex Sans', -apple-system, sans-serif;
            --mono: 'IBM Plex Mono', 'Courier New', monospace;
        }
        /* color-scheme tells the BROWSER (not just our own CSS) that this
           page is dark, so any native chrome we can't fully restyle with
           CSS alone - the password show/hide eye icon, date/time picker
           popups, scrollbars - renders in dark mode too. Without this,
           those native bits were following the visitor's OS light/dark
           setting instead of our app's palette, which is what was
           showing up as a stray white box with black text/icon whenever
           someone's OS/browser was set to light mode. */
        html, body { color-scheme: dark; }

        /* ---- Native form-field theming (applies in both light and dark):
           Streamlit's default input/select/date/time boxes render pure
           white regardless of app theme, which clashed hard with our
           off-white/dark surfaces. Give them our own tinted background and
           make sure the text typed inside stays readable on it. ---- */
        input[type="text"], input[type="password"], input[type="number"],
        input[type="date"], input[type="time"], textarea,
        div[data-baseweb="select"] > div, div[data-baseweb="base-input"],
        div[data-baseweb="input"] {
            background: var(--mv-input-bg) !important;
            border-color: var(--mv-border) !important;
            color: var(--mv-ink) !important;
        }
        input::placeholder, textarea::placeholder { color: var(--mv-muted) !important; opacity: .8 !important; }
        div[data-baseweb="popover"] li, div[data-baseweb="menu"] li {
            background: var(--mv-input-bg) !important;
            color: var(--mv-ink) !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="popover"] div[role="listbox"] {
            background: var(--mv-input-bg) !important;
        }

        /* ---- Password show/hide eye-icon button: it lives inside the
           text input itself rather than as a normal .stButton, so none
           of our button-recoloring rules further down ever touched it -
           left it rendering with the browser/BaseWeb's own default
           button chrome (a plain white square, dark icon) that didn't
           match the rest of the dark theme. Force it transparent with a
           theme-colored icon instead. ---- */
        div[data-testid="stTextInput"] button,
        div[data-baseweb="input"] button,
        div[data-baseweb="base-input"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        div[data-testid="stTextInput"] button svg,
        div[data-baseweb="input"] button svg,
        div[data-baseweb="base-input"] button svg {
            color: var(--mv-muted) !important;
            fill: var(--mv-muted) !important;
        }
        /* Belt-and-suspenders: some browsers (Edge in particular) add
           their OWN native reveal-password icon on top of input type=
           password fields, separate from the button above and outside
           the page's DOM - CSS can't recolor it, only hide it via these
           browser-specific pseudo-elements, so we hide it rather than
           show two overlapping eye icons. */
        input[type="password"]::-ms-reveal,
        input[type="password"]::-ms-clear { display: none !important; }

        html, body, [class*="css"] { font-family: var(--sans); }
        h1, h2, h3 { font-family: var(--serif) !important; letter-spacing: -0.01em; color: var(--mv-ink); }
        h4, h5, h6 { font-family: var(--sans) !important; color: var(--mv-ink); }
        .metric-box .value, [data-testid="stMetricValue"], .rank-badge,
        .vitals-marks, code, .stCodeBlock, .analysis-title, table.wrong-table, .omr-qnum {
            font-family: var(--mono) !important;
        }
        /* Streamlit's own metric widget renders its value in a fairly low-
           contrast gray by default, which read as near-invisible "ghost
           text" against our themed backgrounds - force a solid, readable
           color instead. */
        [data-testid="stMetricValue"] {
            color: var(--mv-ink) !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--mv-muted) !important;
            font-family: var(--sans) !important;
        }

        .mv-exam-meta-grid {
            display:grid; grid-template-columns:1fr 1fr 1.35fr; gap:10px;
            margin:10px 0 4px;
        }
        .mv-exam-meta-grid > div {
            border:1px solid var(--mv-border); border-radius:12px;
            background:var(--mv-card-bg); padding:10px 12px;
        }
        .mv-exam-meta-grid span {
            display:block; color:var(--mv-muted); font-size:10px;
            text-transform:uppercase; letter-spacing:.07em; margin-bottom:3px;
        }
        .mv-exam-meta-grid strong {
            display:block; font-family:var(--mono); color:var(--mv-ink);
            font-size:21px; line-height:1.1;
        }
        .mv-exam-meta-primary strong { color:var(--mv-primary); }
        .mv-exam-meta-secondary strong { font-size:19px; }
        @media (max-width: 640px) {
            .mv-exam-meta-grid { grid-template-columns:1fr 1fr; }
            .mv-exam-meta-secondary { grid-column:1 / -1; }
        }
        [data-testid="stCaptionContainer"], .stCaption, small {
            color: var(--mv-muted) !important;
        }

        /* ---- App-wide background (matches the web app's body color) ---- */
        .stApp { background: var(--mv-bg) !important; color: var(--mv-ink) !important; }
        [data-testid="stHeader"] { background: transparent; }

        /* ---- Hero entry screens (password gate / student login / mentor
           login) - dotted radial background + centered badge + heading,
           same look as the web app's role-selection screen. ---- */
        .mv-hero {
            text-align: center;
            padding: 34px 16px 22px;
            margin: -1rem -1rem 20px;
            background-image: radial-gradient(var(--mv-dot) 1.3px, transparent 1.3px);
            background-size: 18px 18px;
            border-radius: 0 0 22px 22px;
        }
        /* ---- Extra-tight variant used on Student/Mentor login screens:
           two classes beats the single ".mv-hero" rule (and the mobile
           media-query overrides below, which also only carry one class),
           so this wins at every breakpoint without needing its own
           @media copies - keeps the heading from pushing the login card
           below the fold on phones. ---- */
        .mv-hero.mv-hero-compact {
            padding: 14px 16px 10px;
            margin: -1rem -1rem 12px;
        }
        .mv-hero-badge {
            display: inline-flex; align-items: center; gap: 6px;
            font-family: var(--mono); font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
            color: var(--mv-primary); background: var(--mv-primary-soft);
            padding: 5px 14px; border-radius: 999px; margin-bottom: 16px;
        }

        /* ---- Small, quiet link-styled buttons used for secondary auth
           actions (Forgot Password?, Sign Up, Mentor Login, Back to Log
           In) - real st.button/st.form_submit_button widgets underneath
           (so they're properly clickable and can drive navigation), just
           stripped of the normal button chrome and rendered as plain
           text links instead.
           Selectors below deliberately repeat the ".stButton"/
           ".stFormSubmitButton" wrapper class (not just "button") so
           their specificity clearly beats the generic
           ".stButton > button:not([kind=\"primary\"])" pill-button rule
           further down this stylesheet - with equal specificity the
           generic rule (declared later) would win and put the teal
           border box back around these links. ---- */
        .st-key-forgot_pw_link { display: flex; justify-content: flex-end; align-items: center; }
        .st-key-forgot_pw_link .stFormSubmitButton > button:not([kind="primary"]) {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: var(--mv-primary) !important;
            font-size: 12.5px !important;
            font-weight: 600 !important;
            padding: 4px 0 !important;
            min-height: unset !important;
            width: auto !important;
        }
        .st-key-forgot_pw_link .stFormSubmitButton > button:not([kind="primary"]):hover {
            text-decoration: underline; transform: none !important; background: transparent !important;
        }

        .st-key-auth_bottom_links { margin-top: 16px; text-align: center; }
        .st-key-auth_bottom_links div[data-testid="stHorizontalBlock"] { gap: 6px !important; }
        .st-key-auth_signup_link .stButton > button:not([kind="primary"]),
        .st-key-auth_mentor_link .stButton > button:not([kind="primary"]) {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: var(--mv-muted) !important;
            font-size: 12.5px !important;
            font-weight: 600 !important;
            padding: 6px 4px !important;
            min-height: unset !important;
        }
        .st-key-auth_signup_link .stButton > button:not([kind="primary"]):hover,
        .st-key-auth_mentor_link .stButton > button:not([kind="primary"]):hover {
            color: var(--mv-primary) !important;
            text-decoration: underline;
            background: transparent !important;
            transform: none !important;
        }
        .st-key-back_to_login_link { margin-top: 10px; text-align: center; }
        .st-key-back_to_login_link .stButton > button:not([kind="primary"]) {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: var(--mv-primary) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            padding: 4px 0 !important;
            min-height: unset !important;
        }
        .st-key-back_to_login_link .stButton > button:not([kind="primary"]):hover {
            text-decoration: underline; transform: none !important; background: transparent !important;
        }

        /* ---- Auth card (Log In / Sign Up / Forgot Password) - a two-
           column panel: a decorative icon+welcome side and the actual
           form side, matching the reference design.
           Styled via "[class*=...]" wildcards matching real
           st.container(key="auth_card_...") / st.container(key=
           "auth_form_...") wrappers - not a raw <div> split across two
           st.markdown() calls, which (as elsewhere in this file) would
           leave the styled box empty and the real form fields rendering
           unstyled beneath it. ---- */
        .mv-auth-card, [class*="st-key-auth_card_"] {
            border: 1px solid var(--mv-border);
            border-radius: 18px;
            background: var(--mv-surface);
            padding: 6px 6px 16px;
            margin-top: 4px;
            margin-bottom: 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.14), 0 10px 30px rgba(0,0,0,0.18);
        }
        .mv-auth-side {
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 22px 18px;
            border-right: 1px solid var(--mv-border);
        }
        .mv-auth-side-icon-wrap {
            position: relative;
            width: 110px; height: 110px;
            display: flex; align-items: center; justify-content: center;
            margin-bottom: 16px;
        }
        .mv-auth-ring {
            position: absolute; border-radius: 50%;
            border: 1px solid var(--mv-border);
        }
        .mv-auth-ring.r1 { width: 110px; height: 110px; }
        .mv-auth-ring.r2 { width: 80px; height: 80px; }
        /* ---- Orbiting dots: the two small accent dots around the lock
           icon now live inside a full-size wrapper that spins slowly, so
           the dots travel in a circle around the icon instead of sitting
           static. Each dot also gets its own gentle pulse (offset in time
           from the other) layered on top of the spin, so the motion reads
           as "alive" rather than a plain mechanical rotation. ---- */
        .mv-auth-orbit {
            position: absolute;
            inset: 0;
            animation: mv-orbit-spin 9s linear infinite;
        }
        .mv-auth-dot {
            position: absolute; width: 7px; height: 7px; border-radius: 50%;
            background: var(--mv-primary); opacity: .85;
            animation: mv-dot-pulse 2.2s ease-in-out infinite;
        }
        .mv-auth-dot.d1 { top: 8px; right: 14px; }
        .mv-auth-dot.d2 { bottom: 18px; left: 4px; width: 5px; height: 5px; opacity: .5; animation-delay: .8s; }
        @keyframes mv-orbit-spin {
            from { transform: rotate(0deg); }
            to   { transform: rotate(360deg); }
        }
        @keyframes mv-dot-pulse {
            0%, 100% { transform: scale(1); opacity: .8; }
            50%      { transform: scale(1.4); opacity: 1; }
        }
        .mv-auth-icon-box {
            position: relative; z-index: 1;
            width: 54px; height: 54px; border-radius: 16px;
            background: var(--mv-primary-soft);
            border: 1px solid var(--mv-border);
            display: flex; align-items: center; justify-content: center;
            font-size: 22px;
        }
        .mv-auth-icon-box .mv-auth-icon-dot {
            position: absolute; bottom: -3px; right: -3px;
            width: 16px; height: 16px; border-radius: 50%;
            background: var(--mv-accent);
            border: 3px solid var(--mv-surface);
        }
        .mv-auth-side-title {
            font-family: var(--serif); font-weight: 600; font-size: 19px;
            color: var(--mv-ink); margin-bottom: 6px;
        }
        .mv-auth-side-text {
            font-family: var(--sans); font-size: 13px; color: var(--mv-muted);
            max-width: 220px; line-height: 1.5;
        }
        .mv-auth-form-side, [class*="st-key-auth_form_"] { padding: 18px 20px 2px; }
        @media (max-width: 900px) {
            .mv-auth-side { border-right: none; border-bottom: 1px solid var(--mv-border); padding: 18px 16px; }
            .mv-auth-form-side, [class*="st-key-auth_form_"] { padding: 14px 14px 2px; }
        }

        .mv-remember-row {
            display: flex; align-items: center; justify-content: space-between;
            margin: 2px 0 12px;
        }
        .mv-remember-row [data-testid="stCheckbox"] label p {
            font-size: 13px !important; color: var(--mv-muted) !important;
        }
        .mv-forgot-link {
            font-family: var(--sans); font-size: 12.5px; font-weight: 600;
            color: var(--mv-primary); text-align: right;
        }

        /* ---- Panel-style cards: theme-adaptive surface + soft shadow +
           hover lift, matching the web app's .panel / .exam-card look.
           The "[class*=...]" selectors below match any st.container(key=)
           whose key starts with "card_" / "acard_" - see the note by the
           ".app-card" rule further down for why containers (not raw
           markdown divs) are used for these now. ---- */
        .app-card, [class*="st-key-card_"], div[data-testid="stExpander"], div[data-testid="stForm"] {
            background: var(--mv-surface) !important;
            border: 1px solid var(--mv-border) !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 2px rgba(18,32,28,0.05), 0 6px 18px rgba(18,32,28,0.05) !important;
            transition: box-shadow .2s ease, transform .2s ease !important;
        }
        div[data-testid="stForm"] { padding: 18px 18px !important; }
        .app-card:hover, [class*="st-key-card_"]:hover {
            box-shadow: 0 2px 4px rgba(18,32,28,0.07), 0 10px 26px rgba(18,32,28,0.09) !important;
            transform: translateY(-1px);
        }

        /* ---- Buttons: rounded 10px like the web app's .btn. Secondary
           styling is scoped the same way the primary-button rule below is
           (.stButton>button / .stFormSubmitButton>button), never a bare
           "button" selector, so it can never accidentally win over a
           primary button. ---- */
        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: transform .12s ease, box-shadow .12s ease, background-color .15s ease !important;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
        }
        .stButton > button:not([kind="primary"]),
        .stFormSubmitButton > button:not([kind="primary"]),
        .stDownloadButton > button {
            border: 1.4px solid var(--mv-primary) !important;
            color: var(--mv-primary) !important;
            background: transparent !important;
        }
        .stButton > button:not([kind="primary"]):hover,
        .stFormSubmitButton > button:not([kind="primary"]):hover,
        .stDownloadButton > button:hover {
            background: var(--mv-primary-soft) !important;
        }

        /* ---- Digital OMR: proper bubble-sheet look ---- */
        .digital-omr-shell {
            background: linear-gradient(180deg, #0d2020 0%, #0a1818 100%);
            border: 1px solid rgba(38,171,140,.28);
            border-radius: 18px;
            padding: 14px;
            box-shadow: 0 12px 34px rgba(0,0,0,.18);
        }
        .digital-omr-title {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
            margin: 0 0 10px 0;
        }
        .digital-omr-title-main {
            font-family: var(--serif);
            font-size: 20px;
            font-weight: 700;
            color: var(--mv-ink);
        }
        .digital-omr-sub {
            font-size: 11px;
            color: var(--mv-muted);
            margin-top: 2px;
        }
        .digital-omr-block {
            background: rgba(6,17,18,.78);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 13px;
            padding: 8px 7px 4px;
            margin-bottom: 10px;
        }
        .digital-omr-block-title {
            color: var(--mv-primary);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .07em;
            text-transform: uppercase;
            padding: 2px 7px 8px;
        }
        /* Streamlit buttons inside a keyed question container become the
           editable OMR bubbles. */
        [class*="st-key-digital_omr_q_"] button {
            min-height: 31px !important;
            height: 31px !important;
            padding: 0 !important;
            border-radius: 999px !important;
            border: 1.5px solid rgba(125,154,145,.62) !important;
            background: rgba(255,255,255,.025) !important;
            color: #b9c9c3 !important;
            font-family: var(--mono) !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }
        [class*="st-key-digital_omr_q_"] button:hover {
            border-color: var(--mv-primary) !important;
            color: var(--mv-primary) !important;
            background: var(--mv-primary-soft) !important;
            transform: translateY(-1px);
        }
        [class*="st-key-digital_omr_q_"] button[kind="primary"] {
            border: 2px solid var(--mv-primary) !important;
            background: rgba(38,171,140,.18) !important;
            color: #dffff4 !important;
            box-shadow: 0 0 0 3px rgba(38,171,140,.08) !important;
        }
        [class*="st-key-digital_omr_q_"] button p {
            margin: 0 !important;
            line-height: 1 !important;
        }
        [class*="st-key-digital_omr_q_"] [data-testid="stMarkdownContainer"] p {
            margin: 5px 0 0 !important;
            font-family: var(--mono) !important;
            font-size: 11px !important;
            font-weight: 800 !important;
            color: var(--mv-ink) !important;
        }
        .digital-q-issue {
            border-left: 3px solid #f5b83d;
            background: rgba(245,184,61,.07);
            border-radius: 8px;
            padding: 3px 5px;
        }
        .digital-q-edited {
            border-left: 3px solid var(--mv-primary);
            background: rgba(38,171,140,.06);
            border-radius: 8px;
            padding: 3px 5px;
        }
        .omr-photo-card {
            background: #0d2020;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 18px;
            padding: 12px;
            position: sticky;
            top: 12px;
        }
        .omr-photo-label {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .07em;
            text-transform: uppercase;
            color: var(--mv-muted);
            margin-bottom: 8px;
        }

        /* ---- Tabs: pill tab-bar like the web app's .tabbar/.tabbtn ---- */
        [data-baseweb="tab-list"] {
            background: var(--mv-card-bg) !important;
            border-radius: 11px !important;
            padding: 4px !important;
            gap: 2px !important;
        }
        [data-baseweb="tab-list"] button[data-baseweb="tab"] {
            border-radius: 8px !important;
            font-family: var(--sans) !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            color: var(--mv-muted) !important;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: var(--mv-surface) !important;
            color: var(--mv-primary) !important;
            box-shadow: 0 1px 3px rgba(18,32,28,.12) !important;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"] p { color: var(--mv-primary) !important; }
        [data-baseweb="tab-highlight"] { background-color: transparent !important; }
        [data-baseweb="tab-border"] { display: none !important; }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3.0rem;
            max-width: 1180px;
        }
        /* opacity intentionally left OUT of this transition: Streamlit
           already fades stale content to low opacity while a rerun is in
           progress (e.g. right after login, when the page swaps from the
           login form to the logged-in Home page with its top nav).
           Animating that opacity change too stretches out how long that
           in-between frame stays visible, which is what made the old
           page and the new page appear to render on top of each other
           for a moment. Keeping color/background-color animated (for the
           nice hover/theme feel) but leaving opacity instant fixes that
           without losing the rest of the polish. */
        * { transition: background-color .15s ease, color .15s ease; }

        /* ---- App accent color: Med Venture deep teal (was default blue) ---- */
        button[kind="primary"], .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {
            background-color: var(--mv-primary) !important;
            border-color: var(--mv-primary) !important;
            color: #fff !important;
        }
        button[kind="primary"]:hover, .stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primary"]:hover {
            background-color: var(--mv-primary-hover) !important;
            border-color: var(--mv-primary-hover) !important;
        }
        input[type="radio"], input[type="checkbox"] { accent-color: var(--mv-primary) !important; }
        div[role="radiogroup"] label[data-baseweb="radio"] div:first-child,
        [data-testid="stRadio"] label span[data-testid] {
            border-color: var(--mv-primary) !important;
        }
        div[data-baseweb="radio"] div[aria-checked="true"] {
            border-color: var(--mv-primary) !important;
            background-color: var(--mv-primary) !important;
        }
        .stProgress > div > div > div > div { background-color: var(--mv-primary) !important; }

        /* ---- Desktop navigation ---- */
        .st-key-top_nav {
            margin-bottom: 14px;
            border: 1px solid var(--mv-border);
            border-radius: 20px;
            padding: 10px 16px;
            background: var(--mv-surface);
        }
        /* align-items: center is the Bug-2 fix - without it, the desktop
           nav row's columns default to stretch, so the 34px-tall round
           avatar button and the 44px-tall pill nav buttons ended up
           vertically misaligned (avatar sitting higher/lower than the
           pill row instead of sharing the same center line). Centering
           every column's content on this shared axis fixes that for both
           the outer (logo | nav) row and the inner (nav item | ... |
           avatar) row, regardless of any element's own height. */
        .st-key-top_nav div[data-testid="stHorizontalBlock"] {
            gap: 10px;
            align-items: center !important;
        }
        .st-key-top_nav button {
            width: 100%;
            min-height: 40px;
            border-radius: 6px !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            padding: 8px 10px !important;
            font-size: 14px !important;
            white-space: nowrap !important;
            background: transparent !important;
        }
        /* Active nav item: plain underline instead of a filled pill,
           matching the reference design's flat tab-bar look - no
           background box any more, just teal text + a teal underline. */
        .st-key-top_nav .stButton > button[kind="primary"] {
            border: none !important;
            border-bottom: 2px solid var(--mv-primary) !important;
            border-radius: 0 !important;
            background: transparent !important;
            color: var(--mv-primary) !important;
        }
        /* Inactive nav pills: white/light-gray text+icon (matching the
           reference design), NOT the app's usual teal secondary-button
           color - overridden here with a selector scoped to
           ".st-key-top_nav .stButton" specifically (matching Streamlit's
           real button-wrapper markup, ".stButton > button") so it beats
           the app-wide ".stButton > button:not([kind=\"primary\"])" teal
           rule by specificity alone, without touching secondary buttons
           anywhere else in the app. Streamlit's native icon=":material/
           ...:" rendering follows the button's own text color
           automatically, so the icon recolors along with the label with
           no extra rule needed. */
        .st-key-top_nav .stButton > button:not([kind="primary"]) {
            color: var(--mv-ink) !important;
            background: transparent !important;
        }
        .st-key-top_nav .stButton > button:not([kind="primary"]):hover {
            color: var(--mv-primary) !important;
            background: transparent !important;
        }
        /* The avatar button + its wrapping container/column always
           centered on the shared row axis too, on top of the
           align-items:center above - belt-and-suspenders so the round
           avatar never drifts even if its column ends up a different
           height than its siblings for any reason. */
        .st-key-top_nav .st-key-top_nav_avatar_btn {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
        }

        /* Nav icons use Streamlit's native st.button(icon=":material/...:")
           support (see nav_icon() in app.py) - no custom CSS needed here,
           it's rendered as real button markup. */

        /* ---- Mobile top bar + custom expandable menu.
           Rebuilt WITHOUT st.columns AND without position:absolute - the
           absolute-positioning version put the hamburger and profile/
           settings buttons exactly on top of each other. The next attempt
           (targeting ".st-key-mobile_top_bar > div[data-testid=
           'stVerticalBlock']") silently matched nothing at all - the ">"
           direct-child combinator assumed stVerticalBlock sits one level
           BELOW the st-key-* class, but Streamlit puts that testid on the
           SAME element the class is on, so the rule never applied and the
           two buttons just fell back to Streamlit's default stacked
           layout. Fixed by targeting the class itself directly (no ">"),
           with a plain descendant version alongside it as a fallback.
           IMPORTANT: "display" is set in exactly two places only - hidden
           here (!important, every screen size) and shown-as-flex inside
           the max-width:900px media query further down. Putting
           "display: flex !important" in an *unconditional* rule (as an
           earlier version of this file did) beat the "display: none"
           default at every width regardless of the media query, since
           !important always wins over a non-!important rule no matter
           which one is declared first - that's what made this bar show
           up on desktop too. Every other layout property (flex-direction,
           gap, etc.) is safe to keep unconditional since it's inert while
           display:none is in effect.
           ALSO IMPORTANT (Bug-1 fix): every row inside this bar (the
           logo+toggle header, and the drawer's own nav rows) is built
           from plain markup / plain st.button calls now - NEVER
           st.columns(). Streamlit stacks st.columns() vertically below
           its own ~640px responsive breakpoint by default, which is
           exactly what was pushing the hamburger toggle below the logo
           on real phones (which are almost always narrower than that
           breakpoint). Using direct children of this flex container
           instead (each one a plain element-container div) sidesteps
           that breakpoint entirely - a flex row has no such stacking
           behavior. ---- */
        .st-key-mobile_top_bar,
        .st-key-mobile_top_bar[data-testid="stVerticalBlock"] {
            display: none !important;
        }
        .st-key-mobile_top_bar[data-testid="stVerticalBlock"],
        .st-key-mobile_top_bar div[data-testid="stVerticalBlock"] {
            flex-direction: row !important;
            justify-content: space-between !important;
            align-items: center !important;
            gap: 8px !important;
        }
        .st-key-mobile_top_bar[data-testid="stVerticalBlock"] > div,
        .st-key-mobile_top_bar div[data-testid="stVerticalBlock"] > div {
            width: auto !important;
            flex: 0 0 auto !important;
        }
        .st-key-mobile_top_bar button {
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            min-height: 40px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 17px !important;
            border: 1px solid rgba(128,128,128,0.25) !important;
        }
        /* Matched via a wildcard too so a real st.container(key=
           "mobile_menu_...") gets the same look - see the note by the
           ".app-card" rule for why containers replaced raw split <div>s
           throughout this file.
           This is now a FIXED, SLIDE-OUT DRAWER from the right edge of
           the screen (matching the reference design) instead of an
           inline dropdown card that expanded the page content downward -
           position:fixed + a slide-in animation, covering the full
           viewport height, with its own shadow/border separating it from
           the dimmed backdrop behind it (see ".mv-drawer-backdrop"
           below, rendered right before this container in render_top_nav
           when the menu is open). */
        .mobile-menu-card, [class*="st-key-mobile_menu_"] {
            position: fixed;
            top: 0;
            right: 0;
            height: 100vh;
            width: min(82vw, 320px);
            margin: 0;
            padding: 20px 16px;
            border: none;
            border-left: 1px solid var(--mv-border);
            border-radius: 0;
            background: var(--mv-surface);
            box-shadow: -10px 0 32px rgba(0,0,0,0.5);
            z-index: 9999;
            overflow-y: auto;
            animation: mv-drawer-slide-in 0.22s ease-out;
        }
        @keyframes mv-drawer-slide-in {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        .mv-drawer-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.55);
            z-index: 9998;
            animation: mv-backdrop-fade 0.22s ease-out;
        }
        @keyframes mv-backdrop-fade {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        /* Flat, plain-link style nav items inside the drawer (no pill
           background/border by default), matching the reference's icon+
           text list - a soft highlight only appears on hover/tap. */
        .mobile-menu-card button, [class*="st-key-mobile_menu_"] button {
            border-radius: 8px !important;
            min-height: 46px !important;
            text-align: left !important;
            margin-bottom: 6px !important;
            background: transparent !important;
            border: none !important;
            font-size: 15px !important;
        }
        .mobile-menu-card button:hover, [class*="st-key-mobile_menu_"] button:hover {
            background: var(--mv-card-bg) !important;
            color: var(--mv-primary) !important;
        }
        .mobile-menu-card button:last-child, [class*="st-key-mobile_menu_"] button:last-child { margin-bottom: 0 !important; }

        /* ---- Test History table: keeps its 8 columns on one row and
           becomes horizontally swipeable on narrow screens instead of
           Streamlit's default behaviour of stacking every column into a
           tall vertical list (unreadable on a phone). ---- */
        .st-key-test_history_table {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 6px;
        }
        .st-key-test_history_table div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            gap: 6px !important;
            min-width: 620px;
        }
        .st-key-test_history_table div[data-testid="column"] {
            min-width: 0 !important;
            width: auto !important;
        }
        .st-key-test_history_table div[data-testid="column"]:nth-child(1) { flex: 0 0 165px !important; }
        .st-key-test_history_table div[data-testid="column"]:nth-child(2) { flex: 0 0 85px !important; }
        .st-key-test_history_table div[data-testid="column"]:nth-child(3),
        .st-key-test_history_table div[data-testid="column"]:nth-child(4),
        .st-key-test_history_table div[data-testid="column"]:nth-child(5),
        .st-key-test_history_table div[data-testid="column"]:nth-child(6),
        .st-key-test_history_table div[data-testid="column"]:nth-child(7) { flex: 0 0 58px !important; }
        .st-key-test_history_table div[data-testid="column"]:nth-child(8) { flex: 0 0 56px !important; }
        .st-key-test_history_table p, .st-key-test_history_table div[data-testid="stMarkdownContainer"] p {
            font-size: 13px !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .st-key-test_history_table .stButton > button {
            padding: 4px 8px !important; font-size: 12px !important; min-height: 30px !important;
        }
        @media (max-width: 640px) {
            .st-key-test_history_table div[data-testid="stHorizontalBlock"] { min-width: 560px; }
            .st-key-test_history_table div[data-testid="column"]:nth-child(1) { flex-basis: 130px !important; }
        }

        /* ---- Input focus glow (nice subtle brand touch) ---- */
        .stTextInput input:focus, .stNumberInput input:focus,
        .stDateInput input:focus, .stTextArea textarea:focus {
            border-color: var(--mv-primary) !important;
            box-shadow: 0 0 0 3px var(--mv-primary-soft) !important;
        }

        /* ---- Generic cards.
           IMPORTANT: this styling is applied via a real st.container(key=
           "card_...") wrapper now, NOT by splitting an opening/closing
           <div class='app-card'> across two separate st.markdown() calls
           like earlier versions of this file did. Streamlit renders every
           st.markdown() call as its own standalone DOM node, so widgets
           placed "between" an opening and closing markdown call never
           actually end up inside that div - the styled box rendered
           empty, and the real content rendered unstyled right below it,
           which is what was showing up as an extra plain bar above every
           card. Wrapping the real content in st.container(key="card_x")
           and styling that container directly (via the "[class*=...]"
           rule above) fixes it at the root instead of patching around it.
           Padding/margins here are intentionally tight - keeps more
           content on screen per scroll, especially on phones. ---- */
        .app-card, [class*="st-key-card_"] {
            border: 1px solid var(--mv-border);
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 10px;
            background: var(--mv-card-bg);
        }
        .app-card h4, [class*="st-key-card_"] h4,
        .app-card .stMarkdown p, [class*="st-key-card_"] .stMarkdown p { margin-top: 0; }
        [class*="st-key-card_"] div[data-testid="stVerticalBlock"] { gap: 0.4rem !important; }
        .metric-row { display: flex; gap: 8px; flex-wrap: wrap; }
        .metric-box {
            flex: 1 1 140px;
            border-radius: 12px;
            padding: 9px 12px;
            background: rgba(18,60,57,0.055);
            border: 1px solid var(--mv-border);
        }
        .metric-box .label { font-size: 11.5px; opacity: .7; margin-bottom: 1px; }
        .metric-box .value { font-size: 19px; font-weight: 700; }

        /* Same story as "card_" above, applied to the compact per-test-
           result rows used in Test History / Analysis lists (one row per
           test) - real st.container(key="acard_...") now instead of a
           split raw <div>, so a long list of results doesn't render one
           empty bar per row on top of the real (previously unstyled) row
           beneath it. */
        .analysis-test-card, [class*="st-key-acard_"] {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 12px;
            padding: 9px 12px;
            margin-bottom: 6px;
            background: rgba(127,127,127,0.025);
        }
        .analysis-subtle { opacity: .68; font-size: 12px; }
        .analysis-title { font-weight: 700; font-size: 15px; }
        .history-value { text-align:center; min-width:0; white-space:nowrap; overflow:hidden; }
        .history-value span { display:block; font-size:9px; opacity:.68; line-height:1.1; }
        .history-value b { display:block; font-size:14px; line-height:1.2; white-space:nowrap; }


        /* ---- Analysis / Test History cards on mobile: keeps the same
           "Exam name | Marks | Correct | Wrong | View" ROW layout used on
           desktop, instead of Streamlit's default behaviour of stacking
           st.columns() vertically below ~640px (which is what was making
           each metric render as one huge full-width number per line -
           a single card taking up the whole screen).
           IMPORTANT: every rule below is scoped inside the
           @media (max-width: 640px) block ONLY. Desktop (>640px) is left
           completely alone, relying entirely on Streamlit's own default
           st.columns() row layout - an earlier version of this fix
           applied some of these rules unscoped (to all screen sizes),
           which ended up shrinking/cramping the metric numbers on
           desktop too, even though desktop's layout was already fine on
           its own and never needed touching. */
        @media (max-width: 640px) {
            [class*="st-key-acard_"] div[data-testid="stMetric"] {
                text-align: center !important;
            }
            /* CSS GRID instead of flexbox for this row on mobile - an
               earlier flexbox version (fixed flex-basis px widths per
               nth-child column) still broke: flex items have a default
               min-width:auto that lets their CONTENT's natural size win
               over a fixed flex-basis, and Streamlit's metric widgets
               sit one DOM level deeper than the column div we can style,
               so the override never reached them - the Correct/Wrong
               values and the View button ended up overlapping/pushed
               off-screen instead of staying in their slots (see the
               screenshot this was reported from). CSS Grid with
               `minmax(0, 1fr)` for the name column and fixed px tracks
               for the rest doesn't have that problem: a grid track is a
               hard-capped slot regardless of what's inside it, so every
               column - including the View button - reliably stays where
               it's put. */
            [class*="st-key-acard_"] div[data-testid="stHorizontalBlock"] {
                display: grid !important;
                grid-template-columns: minmax(0, 1fr) 42px 42px 42px 46px !important;
                align-items: center !important;
                gap: 4px !important;
            }
            [class*="st-key-acard_"] div[data-testid="column"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                overflow: hidden !important;
            }
            /* Exam-name column (1st grid track, the flexible minmax(0,1fr)
               one): shrinks and ellipses rather than pushing the metrics
               off-screen - this is the one column with genuinely
               variable-length content, so it's the one that should give
               way first. */
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(1) .analysis-title {
                font-size: 12.5px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(1) .analysis-subtle {
                font-size: 10px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            /* Marks / Correct / Wrong metric columns: small label + small
               value, centered in their fixed 42px grid track. */
            [class*="st-key-acard_"] [data-testid="stMetricValue"] {
                font-size: 13px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            [class*="st-key-acard_"] [data-testid="stMetricLabel"] p {
                font-size: 9px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            /* View button column (5th, 46px grid track): compact pill
               button instead of the full-size default. */
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(5) .stButton > button {
                padding: 4px 2px !important;
                font-size: 11px !important;
                min-height: 30px !important;
                width: 100% !important;
            }
            /* The Total/Skipped/Accuracy line underneath - shrink and
               allow it to wrap onto two lines instead of overflowing. */
            [class*="st-key-acard_"] .analysis-subtle:last-child {
                font-size: 10px !important;
                margin-top: 2px;
            }
        }

        /* ---- Every other card built from real st.metric() widgets inside
           st.columns(): Home page's "Active Test" (Questions/Time Left)
           and "Last Result" (Marks/Correct/Wrong), the OMR submission
           result summary (Correct/Wrong/Skipped/Marks), and each mentor
           per-submission review row (Correct/Wrong/Skipped). Same root
           cause as the Analysis cards above (Streamlit stacks
           st.columns() vertically below ~640px), but fixed with CSS
           GRID here instead of flexbox. An earlier flexbox version of
           this rule (flex-basis:0 + min-width:0 on each column) still
           overflowed off the right edge of the screen on mobile - flex
           items have a default min-width:auto that lets their CONTENT'S
           natural size win over flex-shrink even when the flex item
           itself is told min-width:0, and Streamlit's own inner metric
           markup sits one level deeper than the column div we can style,
           so the override never reached it. CSS Grid tracks don't have
           that problem: `repeat(N, 1fr)` divides the row into exactly N
           equal, hard-capped slots regardless of content size, and
           `overflow: hidden` + ellipsis on the metric text is added as a
           second safety net in case any single number is unusually wide. ---- */
        @media (max-width: 640px) {
            .st-key-card_home_active div[data-testid="stHorizontalBlock"] {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 6px !important;
            }
            .st-key-card_home_last div[data-testid="stHorizontalBlock"],
            [class*="st-key-card_mentor_result_"] div[data-testid="stHorizontalBlock"] {
                display: grid !important;
                grid-template-columns: repeat(3, 1fr) !important;
                gap: 6px !important;
            }
            .st-key-card_submit_result div[data-testid="stHorizontalBlock"] {
                display: grid !important;
                grid-template-columns: repeat(4, 1fr) !important;
                gap: 4px !important;
            }
            .st-key-card_home_active div[data-testid="column"],
            .st-key-card_home_last div[data-testid="column"],
            .st-key-card_submit_result div[data-testid="column"],
            [class*="st-key-card_mentor_result_"] div[data-testid="column"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
            }
            .st-key-card_home_active [data-testid="stMetricValue"],
            .st-key-card_home_last [data-testid="stMetricValue"],
            [class*="st-key-card_mentor_result_"] [data-testid="stMetricValue"] {
                font-size: 15px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            .st-key-card_submit_result [data-testid="stMetricValue"] {
                font-size: 13px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            .st-key-card_home_active [data-testid="stMetricLabel"] p,
            .st-key-card_home_last [data-testid="stMetricLabel"] p,
            .st-key-card_submit_result [data-testid="stMetricLabel"] p,
            [class*="st-key-card_mentor_result_"] [data-testid="stMetricLabel"] p {
                font-size: 9.5px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
        }

        /* ---- Semantic metric-number coloring, matching the reference
           design: positive values (Marks, Correct, Average, Skipped)
           render in the app's teal, and Wrong renders in red - instead
           of the generic "everything is var(--mv-ink) white" rule set
           globally on [data-testid="stMetricValue"] further up. Targeted
           with nth-child position since each card's columns always
           appear in the same fixed order, and applied at EVERY screen
           size (not just mobile) since the reference shows this
           coloring at desktop width too. */
        .st-key-card_home_last div[data-testid="column"]:nth-child(1) [data-testid="stMetricValue"],
        .st-key-card_home_last div[data-testid="column"]:nth-child(2) [data-testid="stMetricValue"],
        [class*="st-key-card_mentor_result_"] div[data-testid="column"]:nth-child(1) [data-testid="stMetricValue"] {
            color: var(--mv-primary) !important;
        }
        .st-key-card_home_last div[data-testid="column"]:nth-child(3) [data-testid="stMetricValue"],
        [class*="st-key-card_mentor_result_"] div[data-testid="column"]:nth-child(2) [data-testid="stMetricValue"] {
            color: #F2434A !important;
        }
        /* card_submit_result column order is Correct, Wrong, Skipped, Marks */
        .st-key-card_submit_result div[data-testid="column"]:nth-child(1) [data-testid="stMetricValue"],
        .st-key-card_submit_result div[data-testid="column"]:nth-child(4) [data-testid="stMetricValue"] {
            color: var(--mv-primary) !important;
        }
        .st-key-card_submit_result div[data-testid="column"]:nth-child(2) [data-testid="stMetricValue"] {
            color: #F2434A !important;
        }
        [class*="st-key-acard_"] div[data-testid="column"]:nth-child(2) [data-testid="stMetricValue"] {
            color: var(--mv-primary) !important;
        }
        [class*="st-key-acard_"] div[data-testid="column"]:nth-child(3) [data-testid="stMetricValue"] {
            color: #F2434A !important;
        }
        /* "Overall Progress" card's Average value (2nd metric-box, raw
           HTML not st.metric) - teal like the reference, Tests stays
           the default white. */
        .st-key-card_home_progress .metric-box:nth-child(2) .value {
            color: var(--mv-primary) !important;
        }

        .rank-badge {
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-weight: 700; font-size: 13px;
        }
        .rank-gold { background:#fde68a; color:#78350f; }
        .rank-silver { background:#e5e7eb; color:#374151; }
        .rank-bronze { background:#fbcfe8; color:#831843; }
        .rank-you { background: var(--mv-primary-soft); color: var(--mv-primary); }

        .lb-row {
            display:flex; align-items:center; gap:10px; padding:10px 12px;
            border-radius:10px; margin-bottom:6px; border:1px solid rgba(18,60,57,0.10);
            flex-wrap: wrap;
        }
        .lb-row.me { background: rgba(18,60,57,0.08); border-color: rgba(18,60,57,0.35); }

        /* ---- OMR review bubbles ---- */
        .omr-row {
            display:flex; align-items:center; gap:10px; padding:8px 4px;
            border-bottom:1px solid rgba(128,128,128,0.15);
        }
        .omr-qnum { width:44px; font-weight:700; font-size:13px; opacity:.8; }
        .omr-tag { font-size:11px; padding:2px 8px; border-radius:999px; margin-right:8px; font-weight:600; }
        .omr-tag.wrong-tag { background:#fee2e2; color:#991b1b; }
        .omr-tag.skip-tag { background:#e5e7eb; color:#374151; }
        .omr-bubble {
            width:26px; height:26px; border-radius:50%; border:2px solid rgba(128,128,128,0.35);
            display:inline-flex; align-items:center; justify-content:center;
            font-size:11px; font-weight:700; margin-right:6px; opacity:.85;
        }
        .omr-bubble.correct { background:#22c55e; border-color:#22c55e; color:#fff; opacity:1; }
        .omr-bubble.wrong { background:#ef4444; border-color:#ef4444; color:#fff; opacity:1; }
        .dt-star { color:#ef4444; font-weight:800; margin-left:2px; }
        .double-touch-note {
            margin-top:10px; padding:10px 12px; border-radius:10px;
            background: rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.35);
            font-size:13px;
        }
        .double-touch-note b { color:#ef4444; }

        .digital-omr-grid {
            display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:5px; margin-top:8px;
        }
        .digital-omr-row {
            display:grid; grid-template-columns:38px minmax(92px,1fr) auto; grid-template-rows:auto auto;
            gap:4px 7px; align-items:center; padding:6px 8px; border:1px solid var(--mv-border);
            border-radius:8px; background:rgba(255,255,255,.025); font-size:11.5px; min-width:0;
        }
        .digital-q { grid-row:1 / span 2; font-family:var(--mono); font-weight:800; color:var(--mv-ink); }
        .digital-options { display:flex; gap:3px; min-width:0; }
        .digital-bubble { width:21px; height:21px; border:1px solid rgba(148,163,184,.38); border-radius:50%;
            display:inline-flex; align-items:center; justify-content:center; font:700 9px var(--mono); color:var(--mv-muted); }
        .digital-bubble.detected { background:rgba(34,197,94,.18); border-color:rgba(34,197,94,.65); color:#86efac; }
        .digital-bubble.detected-double { background:rgba(239,68,68,.18); border-color:rgba(239,68,68,.7); color:#fda4af; }
        .digital-bubble.final { box-shadow:0 0 0 2px rgba(41,182,246,.8); color:var(--mv-ink); }
        .digital-bubble.key { text-decoration:underline; text-underline-offset:2px; }
        .digital-your, .digital-correct { color:var(--mv-muted); white-space:nowrap; }
        .digital-your b, .digital-correct b { color:var(--mv-ink); }
        .digital-correct { grid-column:2; }
        .digital-status { grid-column:3; grid-row:1 / span 2; font-size:9px; font-weight:800; padding:3px 6px; border-radius:999px; white-space:nowrap; }
        .digital-status.d-ok { background:rgba(34,197,94,.13); color:#4ade80; }
        .digital-status.d-bad { background:rgba(239,68,68,.13); color:#f87171; }
        .digital-status.d-skip { background:rgba(148,163,184,.13); color:#cbd5e1; }
        .digital-status.d-double { background:rgba(239,68,68,.18); color:#fb7185; }
        @media (max-width: 767px) {
            .digital-omr-grid { grid-template-columns:1fr; gap:4px; }
            .digital-omr-row { grid-template-columns:32px minmax(88px,1fr) auto; padding:5px 6px; font-size:10.5px; }
            .digital-bubble { width:19px; height:19px; font-size:8px; }
            .digital-status { font-size:8px; padding:2px 5px; }
        }
        @media (max-width: 767px) {
            /* OMR review must remain readable on a phone without horizontal
               scrolling. The question number, bubbles, final answer and status
               are all kept inside one compact card. */
            .digital-omr-row {
                grid-template-columns:28px minmax(76px,1fr) auto !important;
                gap:3px 5px !important;
                padding:5px 5px !important;
            }
            .digital-options { gap:2px !important; }
            .digital-bubble { width:18px !important; height:18px !important; font-size:7.5px !important; }
            .digital-your, .digital-correct { font-size:9px !important; }
            .digital-status { font-size:7.5px !important; padding:2px 4px !important; }
        }

        .strength-bar { height:6px; border-radius:4px; background:rgba(128,128,128,0.2); overflow:hidden; margin-top:4px; }
        .strength-fill { height:100%; border-radius:4px; }
        .time-row-label { font-weight:600; padding-top:6px; font-size:14px; }
        .mv-window-heading {
            display:flex; align-items:baseline; gap:10px; margin:8px 0 8px;
            color:var(--mv-ink);
        }
        .mv-window-heading span { font-size:12px; color:var(--mv-muted); font-weight:400; }
        .mv-time-card-title {
            font-size:14px; font-weight:700; color:var(--mv-ink); margin-bottom:1px;
        }
        .mv-time-card-sub {
            font-size:11px; color:var(--mv-muted); margin-bottom:6px;
        }
        [class*="st-key-mentor_start_t_time_card"],
        [class*="st-key-mentor_end_t_time_card"] {
            border:1px solid var(--mv-border); border-radius:12px;
            background:var(--mv-surface); padding:10px 12px 6px;
            box-sizing:border-box; min-width:0;
        }

        /* ---- Phone number field: a country-code st.selectbox ("+880 ⌄"
           style, matching the requested reference design) sitting beside
           a normal st.text_input, laid out with plain CSS flexbox.
           Two earlier attempts at this failed: st.columns() (Streamlit's
           own column-width engine breaks on real phones) and
           position:absolute (never rendered at all). This version puts
           the selectbox and the text_input as two plain, normal children
           inside this container, and forces Streamlit's own wrapper div
           around them into a flex row - using BOTH a same-element and a
           descendant version of the selector below, because Streamlit
           puts the "stVerticalBlock" testid directly on the same element
           as the "st-key-*" class (not one level below it, which is what
           made an earlier ">"-child-combinator version of this rule
           silently match nothing). An explicit border/background/shadow
           is added on this same wrapper so the whole code+digits row
           reads as ONE clearly-bounded field against the page background,
           instead of just relying on the two child pieces' own colors to
           imply a boundary. ---- */
        div[class*="_phone_row"][data-testid="stVerticalBlock"],
        div[class*="_phone_row"] div[data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: stretch !important;
            gap: 0 !important;
            border: 1px solid var(--mv-border) !important;
            border-radius: 9px !important;
            background: var(--mv-input-bg) !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.14) !important;
        }
        div[class*="_phone_row"][data-testid="stVerticalBlock"] > div,
        div[class*="_phone_row"] div[data-testid="stVerticalBlock"] > div {
            margin: 0 !important;
        }
        div[class*="_phone_row"][data-testid="stVerticalBlock"] > div:first-child,
        div[class*="_phone_row"] div[data-testid="stVerticalBlock"] > div:first-child {
            flex: 0 0 168px !important;
            min-width: 168px !important;
        }
        div[class*="_phone_row"][data-testid="stVerticalBlock"] > div:last-child,
        div[class*="_phone_row"] div[data-testid="stVerticalBlock"] > div:last-child {
            flex: 1 1 auto !important;
            min-width: 0 !important;
        }
        /* Country-code side now matches the SAME dark input background as
           the digits side (not the teal "primary-soft" tint it used to
           have) - a thin border-right is what separates the two halves
           visually, while the whole row's own border/shadow above is what
           separates the entire field from the page background. */
        div[class*="_phone_row"] div[data-baseweb="select"] > div {
            height: 46px !important;
            border-radius: 9px 0 0 9px !important;
            border: none !important;
            border-right: 1px solid var(--mv-border) !important;
            font-weight: 600 !important;
            color: var(--mv-primary) !important;
            background: var(--mv-input-bg) !important;
        }
        div[class*="_phone_row"] input {
            height: 46px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            border: none !important;
            border-radius: 0 9px 9px 0 !important;
            background: var(--mv-input-bg) !important;
        }

        /* ---- Profile "Edit Name" row now uses plain st.columns() in
           Python (see page_profile) instead of a custom flex-row CSS
           hack - the hack made the text input invisible for this
           particular avatar+input pairing, so it was removed rather than
           further patched blind. Nothing needed here. ---- */

        /* ---- Profile page: Profile Information / Account Status / Log
           Out cards (matches the reference dashboard-style design). Rows
           inside the info card, and the status pill rows, are plain CSS -
           real markup below, styled here so it's consistent with the
           rest of the app's card look instead of a one-off. ---- */
        .mv-profile-status-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 9px 0; border-bottom: 1px solid var(--mv-border);
        }
        .mv-profile-status-row:last-child { border-bottom: none; }
        .mv-profile-status-label { font-size: 13.5px; color: var(--mv-muted); }
        .mv-profile-status-pill {
            font-size: 11px; padding: 3px 12px; border-radius: 999px; font-weight: 700;
        }

        /* ---- Profile stats strip (Tests Completed / Average Score /
           Leaderboard Rank / Days Active on the student page, and the
           equivalent mentor stats on the mentor page) - one bordered
           card, 4 plain columns, no per-column border/background so it
           reads as a single unified strip matching the reference design.
           The "View X →" / "Keep Going!" line under each stat is a real
           st.button styled as a plain teal link (student side) or a
           static caption (mentor side / the 4th "Days Active" stat) -
           scoped narrowly to this card so it never affects any other
           button in the app. ---- */
        .st-key-card_profile_stats { padding: 18px 20px !important; }
        .st-key-card_profile_stats div[data-testid="column"] { text-align: center; }
        .st-key-card_profile_stats .stButton { display: flex; justify-content: center; }
        .st-key-card_profile_stats .stButton > button:not([kind="primary"]) {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: var(--mv-primary) !important;
            font-size: 12.5px !important;
            font-weight: 600 !important;
            padding: 2px 0 !important;
            margin-top: 2px !important;
            min-height: unset !important;
            width: auto !important;
        }
        .st-key-card_profile_stats .stButton > button:not([kind="primary"]):hover {
            text-decoration: underline; background: transparent !important; transform: none !important;
        }

        /* ---- Small down-chevron next to the top-nav avatar (student and
           mentor) - purely decorative, matching the reference design's
           "avatar + caret" combo. The avatar button itself still does all
           the actual navigation (click anywhere on the circle -> Profile);
           the chevron is a plain, non-interactive span sitting in its own
           narrow column right next to it. ---- */
        .mv-nav-avatar-chevron {
            display: flex; align-items: center; justify-content: center;
            height: 34px; color: var(--mv-muted); font-size: 13px;
        }

        /* ---- Neon-glow profile cards: every card on the Profile page
           (Profile Information / Account Status / Log Out / stats strip /
           "Are you a mentor?") gets a soft teal border-glow on top of the
           app's normal card shadow, matching the reference design's
           slightly "lit up" card edges. Scoped to "card_profile_*" keys
           only (via the [class*=...] wildcard) so this never touches
           unrelated cards elsewhere in the app (Home, Tests, Analysis,
           etc. keep the plain look). Strengthens a little further on
           hover for a subtle interactive feel. ---- */
        [class*="st-key-card_profile_"] {
            border-color: rgba(38,171,140,0.30) !important;
            box-shadow: 0 0 0 1px rgba(38,171,140,0.10), 0 0 22px rgba(38,171,140,0.12),
                        0 1px 2px rgba(18,32,28,0.05), 0 6px 18px rgba(18,32,28,0.05) !important;
        }
        [class*="st-key-card_profile_"]:hover {
            border-color: rgba(38,171,140,0.45) !important;
            box-shadow: 0 0 0 1px rgba(38,171,140,0.18), 0 0 28px rgba(38,171,140,0.18),
                        0 2px 4px rgba(18,32,28,0.07), 0 10px 26px rgba(18,32,28,0.09) !important;
        }
        /* Neon ring around the big avatar on the Profile page header -
           a soft glowing halo (teal for students, accent-orange for the
           mentor) instead of a plain flat circle. Applied via a small
           wrapper span (see _profile_hero_avatar_html() in app.py) so it
           never affects the small avatars used elsewhere (leaderboard
           rows, student-management list, top-nav). */
        .mv-avatar-glow-ring {
            display: inline-flex; border-radius: 50%; padding: 3px;
        }

        /* ---- "Are you a mentor?" card: icon chip + title/description on
           the left, a compact Mentor Login button on the right - matches
           the reference design instead of the earlier plain "text above
           a full-width button" layout. ---- */
        .mv-mentor-cta-icon {
            width: 42px; height: 42px; border-radius: 11px;
            background: var(--mv-accent-soft); color: var(--mv-accent);
            display: flex; align-items: center; justify-content: center;
            font-size: 19px; flex-shrink: 0;
        }
        .st-key-card_profile_mentor div[data-testid="stHorizontalBlock"] { align-items: center !important; }

        /* ---- Account Status card: a large, very faint shield watermark
           in the corner (matching the reference design's decorative
           background icon), implemented as a CSS-only ::after pseudo-
           element with a negative z-index - this paints behind the
           card's real content automatically (no extra markup, no
           fiddling with z-index on the actual status rows, so there's
           no risk of accidentally covering real content). ---- */
        .st-key-card_profile_status { position: relative; overflow: hidden; }
        .st-key-card_profile_status::after {
            content: "🛡️";
            position: absolute; right: -14px; bottom: -22px;
            font-size: 118px; line-height: 1; opacity: 0.05;
            z-index: -1; pointer-events: none;
        }

        /* ---- Log Out card: a destructive-red icon chip and a red-
           outlined button instead of the app's usual teal, since signing
           out is a distinct, deliberate action - matches the reference
           design's red accent on this one card only. Scoped tightly to
           ".st-key-card_profile_logout" (the same container key is
           reused by both the student and mentor Profile pages, so this
           one rule covers both) so no other button in the app is
           affected. ---- */
        .mv-logout-icon {
            width: 34px; height: 34px; border-radius: 9px;
            background: var(--mv-danger-soft); color: var(--mv-danger);
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; flex-shrink: 0;
        }
        .st-key-card_profile_logout .stButton > button:not([kind="primary"]) {
            border-color: var(--mv-danger) !important;
            color: var(--mv-danger) !important;
        }
        .st-key-card_profile_logout .stButton > button:not([kind="primary"]):hover {
            background: var(--mv-danger-soft) !important;
        }

        /* ---- Change Password: a clickable title row (icon-less plain-
           text button, left-aligned, bold) + a caption line underneath
           that's always visible - replaces the old st.expander so the
           description text ("Update your password regularly...") shows
           even before it's opened, matching the reference design. The
           actual toggle-open/closed behavior is a normal session_state
           flag (same pattern as "Update Profile" elsewhere on this
           page), not a custom overlay/hack, so it's exactly as reliable
           as every other toggle in the app. ---- */
        .st-key-card_profile_changepw .stButton > button:not([kind="primary"]) {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: var(--mv-ink) !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 4px 0 !important;
            min-height: unset !important;
        }
        .st-key-card_profile_changepw .stButton > button:not([kind="primary"]):hover {
            color: var(--mv-primary) !important;
            background: transparent !important;
            transform: none !important;
        }

        /* ---- Student per-submission calibration ---- */
        .calib-step-badge {
            display:inline-block; padding:4px 12px; border-radius:999px;
            background: var(--mv-primary-soft); color: var(--mv-primary); font-weight:700; font-size:13px;
        }
        .calib-point-chip {
            display:inline-block; padding:4px 10px; border-radius:999px;
            background:rgba(34,197,94,0.15); color:#15803d; font-weight:600;
            font-size:12px; margin:2px 4px 2px 0;
        }

        @media (max-width: 900px) {
            .block-container { max-width: 100%; padding-left: 1rem; padding-right: 1rem; }
            .st-key-card_profile_mentor div[data-testid="stHorizontalBlock"],
            .st-key-card_profile_mentor div[data-testid="column"] {
                min-width: 0 !important;
            }
            .st-key-top_nav { display: none !important; }
            .st-key-mobile_top_bar,
            .st-key-mobile_top_bar[data-testid="stVerticalBlock"],
            .st-key-mobile_top_bar div[data-testid="stVerticalBlock"] {
                display: flex !important;
            }
            .mv-hero { margin: -1rem -1rem 16px; padding: 26px 12px 18px; }
        }
        @media (max-width: 640px) {
            .metric-box { flex: 1 1 45%; }
            .lb-row { font-size: 13px; }
            .analysis-test-card, [class*="st-key-acard_"] { padding: 8px 10px; }
            .mv-hero { padding: 22px 10px 14px; border-radius: 0 0 16px 16px; }
            .app-card, [class*="st-key-card_"], div[data-testid="stForm"] { padding: 10px 12px !important; }
        }
        @media (min-width: 1400px) {
            .block-container { max-width: 1280px; }
        }

        /* ---- Mobile layout hardening ----
           Keep content inside the viewport and turn the reusable metric /
           leaderboard rows into explicit grids. Streamlit's own responsive
           column stacking is useful for some page sections, but it is not
           reliable for compact cards: long labels can force a column wider
           than the phone and make neighbouring values overlap. */
        .mv-mobile-hide-instruction {
            margin: 4px 0 10px;
            color: var(--mv-muted);
            font-size: 12px;
            line-height: 1.5;
        }

        .metric-row {
            min-width: 0;
        }
        .metric-box {
            min-width: 0;
            overflow: hidden;
            box-sizing: border-box;
        }

        .lb-row {
            min-width: 0;
            box-sizing: border-box;
            overflow: hidden;
        }
        .lb-row > * {
            min-width: 0;
        }

        [class*="st-key-card_"] {
            min-width: 0 !important;
            box-sizing: border-box !important;
        }
        [class*="st-key-card_"] div[data-testid="column"] {
            min-width: 0 !important;
        }

        @media (max-width: 767px) {
            .mv-window-heading { flex-wrap:wrap; gap:2px 8px; }
            .mv-time-card-title { font-size:13.5px; }
            .mv-time-card-sub { margin-bottom:5px; }
            .mv-mobile-hide-instruction {
                display: none !important;
            }

            .metric-row {
                display: grid !important;
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
                gap: 8px !important;
            }
            .metric-box {
                width: 100% !important;
                flex: none !important;
                text-align: center;
                padding: 10px 8px !important;
            }
            .metric-box .label,
            .metric-box .value {
                min-width: 0 !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
            .metric-box .label {
                font-size: 10.5px !important;
                line-height: 1.25 !important;
            }
            .metric-box .value {
                font-size: 17px !important;
            }

            .st-key-card_profile_stats div[data-testid="stHorizontalBlock"] {
                display: grid !important;
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
                gap: 12px 4px !important;
            }
            .st-key-card_profile_stats div[data-testid="column"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                text-align: center !important;
            }

            /* Test history table: preserve the desktop columns semantically, but
               tighten them enough to fit a phone without horizontal scrolling. */
            .st-key-test_history_table div[data-testid="stHorizontalBlock"] {
                /* Eight columns, but sized to the actual phone content width.
                   Exam gets the flexible space; numeric columns stay compact. */
                display:grid !important;
                grid-template-columns:minmax(0,1fr) 48px 28px 34px 32px 38px 42px 38px !important;
                gap:2px !important; width:100% !important; min-width:0 !important;
            }
            .st-key-test_history_table div[data-testid="column"] { min-width:0 !important; width:auto !important; overflow:hidden !important; }
            .st-key-test_history_table p, .st-key-test_history_table button { font-size:9px !important; line-height:1.15 !important; }
            .st-key-test_history_table .stButton > button { padding:3px 3px !important; min-height:28px !important; }

            /* Leaderboard: do NOT squeeze the desktop row into a tiny phone.
               Keep the same one-line proportions as desktop and let the user
               swipe horizontally. This is much easier to read than stacked
               or compressed columns and keeps the desktop/mobile UI visually
               consistent. */
            .st-key-leaderboard_table_student,
            .st-key-leaderboard_table_mentor {
                width: 100% !important;
                max-width: 100% !important;
                overflow-x: auto !important;
                overflow-y: visible !important;
                -webkit-overflow-scrolling: touch !important;
                scrollbar-width: thin;
                padding-bottom: 4px !important;
            }
            .st-key-leaderboard_table_student .lb-row,
            .st-key-leaderboard_table_mentor .lb-row {
                display: flex !important;
                flex-wrap: nowrap !important;
                width: 720px !important;
                min-width: 720px !important;
                max-width: none !important;
                box-sizing: border-box !important;
                align-items: center !important;
                gap: 10px !important;
                overflow: visible !important;
                white-space: nowrap !important;
            }
            .st-key-leaderboard_table_student .lb-row > span,
            .st-key-leaderboard_table_mentor .lb-row > span {
                flex: 0 0 auto !important;
                min-width: 0 !important;
                white-space: nowrap !important;
                overflow: visible !important;
                text-overflow: clip !important;
            }
            /* rank */
            .st-key-leaderboard_table_student .lb-row > span:nth-child(1),
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(1) {
                width: 54px !important;
                text-align: center !important;
            }
            /* student name */
            .st-key-leaderboard_table_student .lb-row > span:nth-child(2),
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(2) {
                width: 205px !important;
                flex: 0 0 205px !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
            /* Overall: Tests / Best / Avg / Acc / Trend */
            .st-key-leaderboard_table_student .lb-row > span:nth-child(3),
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(3) { width: 82px !important; }
            .st-key-leaderboard_table_student .lb-row > span:nth-child(4),
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(4) { width: 82px !important; }
            .st-key-leaderboard_table_student .lb-row > span:nth-child(5),
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(5) { width: 82px !important; }
            .st-key-leaderboard_table_student .lb-row > span:nth-child(6),
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(6) { width: 82px !important; }
            .st-key-leaderboard_table_student .lb-row > span:nth-child(7),
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(7) { width: 82px !important; text-align: right !important; }

            /* Test-wise has only Score + Accuracy. Keep those columns
               comfortably spaced instead of stretching them across the phone. */
            .st-key-leaderboard_table_student .lb-row > span:nth-child(3):last-child,
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(3):last-child { width: 105px !important; }
            .st-key-leaderboard_table_student .lb-row > span:nth-child(4):last-child,
            .st-key-leaderboard_table_mentor .lb-row > span:nth-child(4):last-child { width: 115px !important; }
        }

        /* ---- Analysis / exam-history rows on mobile -------------------
           No horizontal scroll: keep the same desktop information but pack it
           into a two-line card that fits a narrow phone. */
        /* Mobile Analysis / Test History
           Keep every statistic in a fixed-width cell so values like 40, 100,
           40.5 can NEVER break into separate digits. */
        @media (max-width: 767px) {
            [class*="st-key-acard_"] {
                width:100% !important; max-width:100% !important; box-sizing:border-box !important;
                padding:10px 9px !important; overflow:hidden !important;
                border-radius:12px !important;
            }
            [class*="st-key-acard_"] div[data-testid="stHorizontalBlock"] {
                display:grid !important;
                grid-template-columns:minmax(0,1fr) 54px 54px 54px 54px !important;
                width:100% !important; min-width:0 !important; max-width:100% !important;
                gap:5px !important; align-items:center !important;
            }
            [class*="st-key-acard_"] div[data-testid="column"] {
                min-width:0 !important; width:auto !important; max-width:100% !important;
                overflow:hidden !important;
            }
            [class*="st-key-acard_"] .analysis-title {
                font-size:12px !important; line-height:1.2 !important;
                white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
            }
            [class*="st-key-acard_"] .analysis-subtle {
                font-size:9px !important; line-height:1.15 !important;
                white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
            }
            [class*="st-key-acard_"] .history-value {
                width:54px !important; min-width:54px !important; max-width:54px !important;
                text-align:center !important; white-space:nowrap !important; overflow:hidden !important;
            }
            [class*="st-key-acard_"] .history-value span {
                display:block !important; font-size:8px !important; line-height:1 !important;
                white-space:nowrap !important;
            }
            [class*="st-key-acard_"] .history-value b {
                display:block !important; font-size:14px !important; line-height:1.25 !important;
                white-space:nowrap !important; word-break:keep-all !important; overflow:visible !important;
            }
            [class*="st-key-acard_"] .stButton > button {
                min-width:54px !important; width:54px !important; padding:5px 3px !important;
                font-size:10px !important; white-space:nowrap !important;
            }
            [class*="st-key-acard_"] > div[data-testid="stVerticalBlock"] {
                min-width:0 !important; width:100% !important;
            }
        }

        /* Very narrow phones: stack the title above the four fixed stats.
           This is intentionally a card, not a horizontally scrolling table. */
        @media (max-width: 380px) {
            [class*="st-key-acard_"] div[data-testid="stHorizontalBlock"] {
                grid-template-columns:minmax(0,1fr) 52px 52px 52px 52px !important;
                gap:3px !important;
            }
            [class*="st-key-acard_"] .history-value,
            [class*="st-key-acard_"] .stButton > button {
                width:52px !important; min-width:52px !important; max-width:52px !important;
            }
        }

        /* ---- Themed spinner (recolors Streamlit's built-in spinner to
           match Med Venture instead of the generic default) ---- */
        [data-testid="stSpinner"] { color: var(--mv-primary) !important; }
        [data-testid="stSpinner"] svg { color: var(--mv-primary) !important; fill: var(--mv-primary) !important; }
        [data-testid="stSpinner"] p, [data-testid="stSpinner"] div {
            font-family: var(--mono) !important; color: var(--mv-muted) !important; font-size: 13px !important;
        }

        /* ---- Boot loading screen pulse-line animation ---- */
        @keyframes mv-pulse-draw {
            0% { stroke-dashoffset: 520; }
            55% { stroke-dashoffset: 0; }
            100% { stroke-dashoffset: 0; opacity: 0; }
        }
        .mv-boot-pulse path { animation: mv-pulse-draw 1.8s ease-in-out infinite; }

        /* ---- Entry-screen (password gate / student login / mentor login)
           pulse-line: unlike the one-shot boot pulse above (which draws once
           then fades out), this one keeps a short "traveling" dash segment
           looping across the line forever - so the heartbeat line on the
           first screen keeps animating continuously instead of freezing
           after the first pass. ---- */
        .mv-hero-pulse-path {
            stroke-dasharray: 90 400;
            animation: mv-hero-pulse-travel 3.2s ease-in-out infinite;
        }
        @keyframes mv-hero-pulse-travel {
            0%   { stroke-dashoffset: 490; opacity: 0; }
            10%  { opacity: 1; }
            70%  { opacity: 1; }
            92%  { stroke-dashoffset: -400; opacity: 0; }
            100% { stroke-dashoffset: -400; opacity: 0; }
        }

        /* ---- Responsive two-panel / compact table system ---- */
        .mv-compact-row { border:1px solid var(--mv-border); border-radius:12px; background:var(--mv-surface); padding:7px 9px; margin:4px 0; }
        .mv-review-dot { font-size:13px; text-align:center; }
        @media (max-width: 800px) {
            .stApp [data-testid="stHorizontalBlock"] { gap: 6px !important; }
            .stApp [data-testid="stColumn"] { min-width: 0 !important; }
            /* OMR panels become one-column on narrow screens; no horizontal overflow. */
            .stApp div[data-testid="column"] { min-width: 0 !important; }
            .stApp .stButton > button { padding: 6px 8px !important; min-height: 36px !important; font-size: 12px !important; }
            .mv-compact-row { padding: 5px 6px; }
            .mv-compact-row [data-testid="stMetricValue"] { font-size: 18px !important; }
            .mv-exam-meta-grid { gap:6px !important; }
        }
        @media (max-width: 560px) {
            .mv-mobile-stack > div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
            .stApp .stTextInput input, .stApp .stSelectbox, .stApp textarea { font-size: 13px !important; }
            .stApp [data-testid="stMarkdownContainer"] p { overflow-wrap:anywhere; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


MOTIVATIONS = [
    "Small daily progress adds up to big results. Keep going!",
    "Every test you take makes you sharper for the real one.",
    "Mistakes today are lessons you won't repeat tomorrow.",
    "Consistency beats intensity. Show up and practice.",
    "Your best score is still ahead of you.",
    "Review your wrong answers - that's where growth happens.",
    "Discipline now, results later. You're on the right track.",
]


def motivation_for(student_id):
    seed = f"{student_id}-{date.today().isoformat()}"
    rnd = random.Random(seed)
    return rnd.choice(MOTIVATIONS)


# A small fixed palette of theme-friendly colors for avatars - picked to
# stay readable with white initials on top and to feel at home next to
# the app's teal/coral Med Venture palette rather than clashing with it.
AVATAR_COLORS = [
    "#26AB8C", "#F94D10", "#7C9CE6", "#C77DE0", "#E0A23D",
    "#5FBF6B", "#E0637D", "#4FA0E6", "#B0A23D", "#8B7DE0",
]

# Options offered on the Profile page's "Update Profile" form. Birth date
# is intentionally NOT collected at signup (see phone_field()/signup view
# below) - name + phone + password is all that's needed to create an
# account - but a student can add/update it any time afterwards here.
GENDER_OPTIONS = ["Not specified", "Male", "Female", "Other"]


def _avatar_initials(name):
    """Up to 2 letters: first letter of the first two words, or the first
    2 letters of a single word - falls back to '?' for an empty name."""
    parts = [p for p in str(name or "").strip().split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _avatar_color(student_id, name=None):
    """Same deterministic color-pick used by render_avatar(), pulled out
    on its own so callers that need just the color (e.g. styling a real
    st.button to look like an avatar, where the button's label has to be
    plain text/emoji rather than the HTML span render_avatar() returns)
    don't have to duplicate the seeding logic."""
    seed = str(student_id or name or "?")
    return random.Random(seed).choice(AVATAR_COLORS)


def render_avatar(student_id, name, size=40, font_size=None):
    """
    Deterministic, no-network avatar: a solid-color circle with the
    student's initials, colored by hashing their student_id (NOT their
    name, so two students who happen to share a first name still get
    visually distinct avatars) into AVATAR_COLORS. Same student_id always
    gets the same color and initials, every time it's rendered anywhere
    in the app - no external avatar service, image upload, or network
    call needed, so it can never fail to load or add latency.

    Returns an HTML string (a single inline <span>) meant to be embedded
    inside a larger st.markdown(..., unsafe_allow_html=True) call, or
    rendered on its own.
    """
    color = _avatar_color(student_id, name)
    initials = _avatar_initials(name)
    if font_size is None:
        font_size = max(11, int(size * 0.42))
    return (
        f"<span style='display:inline-flex; align-items:center; justify-content:center; "
        f"width:{size}px; height:{size}px; min-width:{size}px; border-radius:50%; "
        f"background:{color}; color:#fff; font-family:var(--sans); "
        f"font-weight:700; font-size:{font_size}px; letter-spacing:.02em; "
        f"line-height:1; vertical-align:middle;'>{initials}</span>"
    )


def render_student_header(student_id, name, heading_level=3):
    """Shared "[avatar] Name" header used at the top of every student-
    analysis-style page (a student's own Analysis page, and the mentor's
    per-student Analysis/drilldown page) so the same avatar + name +
    Student ID block doesn't get copy-pasted three times."""
    tag = f"h{heading_level}"
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>"
        f"{render_avatar(student_id, name, size=40)}"
        f"<{tag} style='margin:0;'>{name}</{tag}>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Student ID: **{student_id}**")


# =========================================================================
# Cached reads (keeps the app snappy - avoids hitting Google Sheets on
# every single rerun/click, which is what causes "hang" in Streamlit apps)
# =========================================================================

@st.cache_data(ttl=30, show_spinner=False)
def cached_answer_keys():
    return sh.get_all_answer_keys()


@st.cache_data(ttl=20, show_spinner=False)
def cached_results():
    return sh.get_all_results_df()


@st.cache_data(ttl=30, show_spinner=False)
def cached_students():
    return sh.get_all_students_df()


@st.cache_data(ttl=10, show_spinner=False)
def cached_active_answer_key():
    return sh.get_active_answer_key()


@st.cache_data(ttl=20, show_spinner=False)
def cached_upcoming_answer_key():
    return sh.get_upcoming_answer_key()


@st.cache_data(ttl=30, show_spinner=False)
def cached_calibration():
    return sh.load_calibration()


@st.cache_data(ttl=10, show_spinner=False)
def cached_session_version(student_id):
    return sh.get_session_version(student_id)


@st.cache_data(ttl=20, show_spinner=False)
def cached_rank(student_id, key_id=None):
    return sh.get_rank_for_student(student_id, key_id)


def clear_all_caches():
    cached_answer_keys.clear()
    cached_results.clear()
    cached_students.clear()
    cached_active_answer_key.clear()
    cached_upcoming_answer_key.clear()
    cached_calibration.clear()
    cached_session_version.clear()
    cached_rank.clear()
    sh.clear_data_caches()


# =========================================================================
# Routing helpers
# =========================================================================

def go_to(page, **params):
    st.session_state["page"] = page
    for k, v in params.items():
        st.session_state[k] = v

    # Close mobile menus after navigation and clear stale detail views.
    st.session_state["student_mobile_menu_open"] = False
    st.session_state["mentor_mobile_menu_open"] = False
    if page != "analysis":
        st.session_state.pop("analysis_view_key_id", None)
    if page != "mentor_student_analysis":
        st.session_state.pop("mentor_analysis_view_key_id", None)

    st.query_params["page"] = page
    st.rerun()


def restore_page_from_url():
    if "page" not in st.session_state:
        st.session_state["page"] = st.query_params.get("page", "home")


# =========================================================================
# App-level password gate (keeps the whole app private)
# =========================================================================

def check_app_password():
    if st.session_state.get("authed"):
        return True

    render_hero("Medical Admission Prep", tagline="Enter your access password to continue")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form(key="app_pw_form", clear_on_submit=False):
            pw = st.text_input("Password", type="password", label_visibility="collapsed",
                                placeholder="Access password")
            submitted = st.form_submit_button("Continue", use_container_width=True, type="primary")
            if submitted:
                if pw == st.secrets.get("APP_PASSWORD", ""):
                    st.session_state["authed"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    return False


# =========================================================================
# Student auth: sign up / log in / forgot password
# =========================================================================

SECURITY_QUESTIONS = [
    "What is your favorite subject?",
    "What is your mother's first name?",
    "What was the name of your first school?",
    "What is your favorite color?",
]

# Full worldwide country list for the phone field's country-code dropdown,
# shown as "Country (+code)" labels so a user can click the dropdown and
# type a country name (e.g. "Bangla") to filter down to it - st.selectbox
# has built-in typeahead search, so no extra search widget is needed.
#
# IMPORTANT: the actual login/signup/reset backend (sh.validate_bd_phone_digits
# and everything downstream of it in sheets_helper.py) only understands
# Bangladeshi 10-digit numbers - it has no concept of other countries'
# number formats. Rather than silently pretending other codes work (which
# would just fail validation in a confusing way, or worse, store a
# malformed number), phone_field() below still returns the selected dial
# code alongside the digits so each caller can gate on `code == "+880"`
# before calling into that BD-specific validation, and show a plain "only
# +880 is supported right now" message otherwise. Extending real validation
# to other countries would need changes in sheets_helper.py, which is out
# of scope here - this list only widens what's *shown*, not what's *accepted*.
COUNTRY_CODES = [
    ("Afghanistan", "+93"), ("Albania", "+355"), ("Algeria", "+213"),
    ("Argentina", "+54"), ("Australia", "+61"), ("Austria", "+43"),
    ("Bahrain", "+973"), ("Bangladesh", "+880"), ("Belgium", "+32"),
    ("Bhutan", "+975"), ("Brazil", "+55"), ("Brunei", "+673"),
    ("Cambodia", "+855"), ("Canada", "+1"), ("China", "+86"),
    ("Denmark", "+45"), ("Egypt", "+20"), ("Finland", "+358"),
    ("France", "+33"), ("Germany", "+49"), ("Greece", "+30"),
    ("Hong Kong", "+852"), ("India", "+91"), ("Indonesia", "+62"),
    ("Iran", "+98"), ("Iraq", "+964"), ("Ireland", "+353"),
    ("Italy", "+39"), ("Japan", "+81"), ("Jordan", "+962"),
    ("Kuwait", "+965"), ("Malaysia", "+60"), ("Maldives", "+960"),
    ("Mexico", "+52"), ("Myanmar", "+95"), ("Nepal", "+977"),
    ("Netherlands", "+31"), ("New Zealand", "+64"), ("Norway", "+47"),
    ("Oman", "+968"), ("Pakistan", "+92"), ("Philippines", "+63"),
    ("Poland", "+48"), ("Portugal", "+351"), ("Qatar", "+974"),
    ("Russia", "+7"), ("Saudi Arabia", "+966"), ("Singapore", "+65"),
    ("South Africa", "+27"), ("South Korea", "+82"), ("Spain", "+34"),
    ("Sri Lanka", "+94"), ("Sweden", "+46"), ("Switzerland", "+41"),
    ("Thailand", "+66"), ("Turkey", "+90"), ("UAE", "+971"),
    ("UK", "+44"), ("USA", "+1"), ("Vietnam", "+84"),
]
COUNTRY_LABELS = [f"{name} ({code})" for name, code in COUNTRY_CODES]
COUNTRY_LABEL_TO_CODE = {f"{name} ({code})": code for name, code in COUNTRY_CODES}
# Bangladesh is the default selection every time this field is rendered,
# since the vast majority of users are Bangladeshi students/mentors - this
# just saves them a search on the common case, not a functional restriction.
DEFAULT_COUNTRY_LABEL = "Bangladesh (+880)"


def phone_field(key_prefix, placeholder="1712345678"):
    """A phone number field with a searchable worldwide country selector
    (type a country name to filter, e.g. "Bangla" -> "Bangladesh (+880)")
    next to a plain digits input. Defaults to Bangladesh on first render.

    See the COUNTRY_CODES comment above for why only +880 actually
    validates right now - other countries are selectable (so the UI looks
    and behaves like a real international phone field) but will show a
    "not supported yet" caption instead of silently failing.

    Deliberately NOT built with st.columns(): Streamlit resizes columns
    using its own internal responsive rules on narrow screens, which was
    fighting our CSS overrides on phones. This instead puts the selectbox
    and the real st.text_input as two plain, normal children in this
    container, and the "_phone_row" CSS in inject_global_css forces
    Streamlit's own wrapper div around them into a simple CSS flex row -
    no column-width engine involved, just two boxes side by side like
    ordinary HTML.
    Returns (selected_code, digits) - validate digits with
    sh.validate_bd_phone_digits only when selected_code == "+880"."""
    st.markdown("**Phone number**")
    with st.container(key=f"{key_prefix}_phone_row"):
        default_idx = COUNTRY_LABELS.index(DEFAULT_COUNTRY_LABEL)
        label = st.selectbox(
            "Country", COUNTRY_LABELS, index=default_idx,
            key=f"{key_prefix}_code", label_visibility="collapsed",
        )
        code = COUNTRY_LABEL_TO_CODE[label]
        digits = st.text_input(
            "Phone number", key=f"{key_prefix}_digits", label_visibility="collapsed",
            placeholder=placeholder, max_chars=10,
        )
    if code != "+880":
        st.caption("⚠️ Only Bangladeshi (+880) numbers can be used to log in right now.")
    return code, digits


def student_session_is_valid():
    """Session security: if the password was changed (or account disabled)
    elsewhere, session_version on the sheet will differ from what we
    stored at login time - force logout."""
    sid = st.session_state.get("student_id")
    if not sid:
        return False
    live_version = cached_session_version(sid)
    if live_version is None:
        return False
    return live_version == st.session_state.get("session_version")


def _render_login_view():
    render_hero("Student Portal", heading_html="Student Login", compact=True, pulse=False,
                show_badge=False, show_byline=False)

    with st.container(key="auth_card_login"):
        side_col, form_col = st.columns([1, 1.35], gap="small")
        with side_col:
            st.markdown(
                """
                <div class='mv-auth-side'>
                    <div class='mv-auth-side-icon-wrap'>
                        <div class='mv-auth-ring r1'></div>
                        <div class='mv-auth-ring r2'></div>
                        <div class='mv-auth-orbit'>
                            <span class='mv-auth-dot d1'></span>
                            <span class='mv-auth-dot d2'></span>
                        </div>
                        <div class='mv-auth-icon-box'>🔒<span class='mv-auth-icon-dot'></span></div>
                    </div>
                    <div class='mv-auth-side-title'>Welcome back!</div>
                    <div class='mv-auth-side-text'>Login to access your student dashboard</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with form_col:
            with st.container(key="auth_form_login"):
                # Wrapped in a real st.form: this both (a) lets the browser detect
                # it as a login form for autofill / "remember password", and
                # (b) makes pressing Enter inside any field submit the form - no
                # extra click needed after autofill/paste. No live password-
                # strength feedback is needed here, so a form (which only reruns
                # on submit) doesn't cost us anything.
                #
                # "Forgot Password?" is a SECOND st.form_submit_button inside this
                # same form (Streamlit allows more than one per form) styled as a
                # plain text link via the ".st-key-forgot_pw_link" CSS - clicking
                # it submits the form too, but we check *which* button fired below
                # and route to the forgot-password view without touching the
                # phone/password validation meant for the actual Log In button.
                with st.form(key="login_form", clear_on_submit=False):
                    login_code, login_digits = phone_field("login")
                    pw = st.text_input("Password", type="password", key="login_pw")
                    rc1, rc2 = st.columns([1, 1])
                    with rc1:
                        st.checkbox("Remember me", key="login_remember_me_ui")
                    with rc2:
                        with st.container(key="forgot_pw_link"):
                            forgot_clicked = st.form_submit_button("Forgot Password?")
                    submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

    if forgot_clicked:
        st.session_state["student_auth_view"] = "forgot"
        st.rerun()

    if submitted:
        if login_code != "+880":
            st.error("Only Bangladeshi (+880) numbers can be used to log in right now.")
        else:
            ok, err, canonical_phone = sh.validate_bd_phone_digits(login_digits)
            if not ok:
                st.error(err)
            elif not pw:
                st.error("Please enter your password.")
            else:
                with st.spinner("Logging in..."):
                    try:
                        student = sh.authenticate_student(canonical_phone, pw)
                        st.session_state["student_id"] = student["student_id"]
                        st.session_state["student_name"] = student["name"]
                        st.session_state["session_version"] = sh._to_int(student.get("session_version"), 1)
                        st.session_state["role"] = "student"
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.success("Logged in!")
                        go_to("home")

    # ---- Small, quiet secondary links: Sign Up (for new students) and
    # Mentor Login - both real st.button widgets styled as plain text
    # links (see ".st-key-auth_signup_link" / ".st-key-auth_mentor_link"
    # in inject_global_css) instead of the earlier full-width pill button,
    # so they read as secondary actions rather than competing with Log In.
    with st.container(key="auth_bottom_links"):
        l1, l2 = st.columns(2)
        with l1:
            with st.container(key="auth_signup_link"):
                if st.button("New here? Sign Up", key="goto_signup_btn", use_container_width=True):
                    st.session_state["student_auth_view"] = "signup"
                    st.rerun()
        with l2:
            with st.container(key="auth_mentor_link"):
                if st.button("Are you a mentor? Mentor Login", key="goto_mentor_btn", use_container_width=True):
                    go_to("mentor")


def _render_signup_view():
    render_hero("Student Portal", heading_html="Create Account", compact=True, pulse=False,
                show_badge=False, show_byline=False)

    with st.container(key="auth_card_signup"):
        with st.container(key="auth_form_signup"):
            # NOT wrapped in st.form on purpose: a form only reruns the script
            # when its submit button is clicked, so a password-strength meter
            # inside a form only ever updates AFTER you hit submit - which is
            # exactly the confusing behaviour we're fixing here. Plain widgets
            # rerun on every keystroke, so the strength bar updates live while
            # typing, before the button is ever pressed.
            #
            # Deliberately only name + phone + password + security question
            # are asked here - no birth date / gender at signup. Those are
            # optional and can be filled in any time afterwards from the
            # Profile page (see page_profile()).
            name = st.text_input("Full name", key="su_name")
            su_code, phone_digits = phone_field("su")
            pw1 = st.text_input("Password", type="password", key="su_pw1")
            if pw1:
                score, label, _tips = sh.password_strength(pw1)
                colors = ["#ef4444", "#ef4444", "#f59e0b", "#10b981", "#059669"]
                st.markdown(
                    f"<div class='strength-bar'><div class='strength-fill' "
                    f"style='width:{(score+1)*20}%; background:{colors[score]};'></div></div>"
                    f"<small>Password strength: <b>{label}</b></small>",
                    unsafe_allow_html=True,
                )
            pw2 = st.text_input("Confirm password", type="password", key="su_pw2")
            sec_q = st.selectbox("Security question (used for password recovery)", SECURITY_QUESTIONS, key="su_secq")
            sec_a = st.text_input("Your answer", key="su_seca")

            if st.button("Create Account", type="primary", use_container_width=True, key="signup_btn"):
                ok, phone_err, canonical_phone = sh.validate_bd_phone_digits(phone_digits)
                _, _, tips = sh.password_strength(pw1)
                if not name.strip():
                    st.error("Please enter your name.")
                elif su_code != "+880":
                    st.error("Only Bangladeshi (+880) numbers can be used to sign up right now.")
                elif not ok:
                    st.error(phone_err)
                elif tips:
                    st.error("Password is too weak: " + ", ".join(tips))
                elif pw1 != pw2:
                    st.error("Passwords do not match.")
                elif not sec_a.strip():
                    st.error("Please answer the security question.")
                else:
                    with st.spinner("Creating your account..."):
                        try:
                            sh.create_student(name, canonical_phone, pw1, sec_q, sec_a)
                            clear_all_caches()
                        except ValueError as e:
                            st.error(str(e))
                        else:
                            st.success("Account created! Please log in below.")
                            st.session_state["student_auth_view"] = "login"

    with st.container(key="back_to_login_link"):
        if st.button("← Back to Log In", key="signup_back_to_login"):
            st.session_state["student_auth_view"] = "login"
            st.rerun()


def _render_forgot_view():
    render_hero("Student Portal", heading_html="Reset Password", compact=True, pulse=False,
                show_badge=False, show_byline=False)

    with st.container(key="auth_card_forgot"):
        with st.container(key="auth_form_forgot"):
            st.caption("Reset your password using the security question you set at sign up.")
            fp_code, f_phone_digits = phone_field("fp")
            ok_preview, err_preview, canonical_preview = (
                sh.validate_bd_phone_digits(f_phone_digits) if fp_code == "+880"
                else (False, "Only Bangladeshi (+880) numbers are supported right now.", None)
            )
            student_preview = sh.get_student_by_phone(canonical_preview) if ok_preview else None
            if student_preview:
                st.info(f"Security question: **{student_preview.get('security_question')}**")
                f_answer = st.text_input("Your answer", key="fp_answer")
                f_new1 = st.text_input("New password", type="password", key="fp_new1")
                if f_new1:
                    score, label, _tips = sh.password_strength(f_new1)
                    colors = ["#ef4444", "#ef4444", "#f59e0b", "#10b981", "#059669"]
                    st.markdown(
                        f"<div class='strength-bar'><div class='strength-fill' "
                        f"style='width:{(score+1)*20}%; background:{colors[score]};'></div></div>"
                        f"<small>Password strength: <b>{label}</b></small>",
                        unsafe_allow_html=True,
                    )
                f_new2 = st.text_input("Confirm new password", type="password", key="fp_new2")
                if st.button("Reset Password", type="primary", use_container_width=True, key="fp_btn"):
                    _, _, tips = sh.password_strength(f_new1)
                    if tips:
                        st.error("Password is too weak: " + ", ".join(tips))
                    elif f_new1 != f_new2:
                        st.error("Passwords do not match.")
                    else:
                        with st.spinner("Resetting..."):
                            try:
                                sh.reset_password_via_security(canonical_preview, f_answer, f_new1)
                                clear_all_caches()
                            except ValueError as e:
                                st.error(str(e))
                            else:
                                st.success("Password reset! Please log in with your new password.")
                                st.session_state["student_auth_view"] = "login"
            elif f_phone_digits.strip():
                if not ok_preview:
                    st.caption(err_preview)
                else:
                    st.warning("No account found with this phone number.")

    with st.container(key="back_to_login_link"):
        if st.button("← Back to Log In", key="forgot_back_to_login"):
            st.session_state["student_auth_view"] = "login"
            st.rerun()


def page_student_auth():
    # A small session_state-driven "sub-page" inside the student auth
    # screen (login / signup / forgot) instead of st.tabs - st.tabs has no
    # way to be switched programmatically, which is exactly what the
    # "Forgot Password?" link and the "Sign Up" link below need to do.
    view = st.session_state.get("student_auth_view", "login")
    if view == "signup":
        _render_signup_view()
    elif view == "forgot":
        _render_forgot_view()
    else:
        _render_login_view()


# =========================================================================
# Top navigation (persistent on every student page)
# =========================================================================

STUDENT_NAV = [
    ("home", "Home"),
    ("tests", "Tests & Results"),
    ("analysis", "Analysis"),
    ("leaderboard", "Leaderboard"),
    ("profile", "Profile"),
]

# Nav icons via Streamlit's own native st.button(icon=...) support -
# real, first-party rendering (not a CSS overlay or ::before hack) so it
# can never fail to show up or misalign, unlike two earlier attempts at
# drawing fully custom SVGs ourselves (an absolute-positioned overlay
# that ended up floating above the pill, then a ::before pseudo-element
# that Streamlit/BaseWeb's internal button markup didn't reliably treat
# as a flex sibling of the label). Material Symbols still gives clean,
# minimal, no-emoji line icons matching the reference design's look -
# just from Streamlit's built-in icon set rather than hand-drawn paths.
NAV_MATERIAL_ICONS = {
    "home": "home",
    "tests": "assignment",
    "analysis": "bar_chart",
    "leaderboard": "emoji_events",
    "profile": "person",
}


def nav_icon(page_key):
    """Returns the ':material/xxx:' icon string for st.button(icon=...)."""
    name = NAV_MATERIAL_ICONS.get(page_key)
    return f":material/{name}:" if name else None



def render_top_nav(current_page):
    # Desktop: logo on the far left, plain (no pill background/border)
    # nav items, and the avatar circle on the far right - matching the
    # reference design's flat, link-style nav bar instead of the earlier
    # rounded-pill-button look. Active page is shown via bold teal text
    # rather than a filled pill. The whole row (including the avatar) is
    # vertically centered via the ".st-key-top_nav div[data-testid=
    # 'stHorizontalBlock']" align-items:center rule in inject_global_css -
    # that's the Bug-2 fix (avatar not lining up with the pill buttons).
    desktop_nav_items = [item for item in STUDENT_NAV if item[0] != "profile"]
    with st.container(key="top_nav"):
        logo_col, nav_col = st.columns([1.6, 6.4])
        with logo_col:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; height:100%; padding-top:2px;'>"
                f"<div style='width:26px; height:26px; flex-shrink:0;'>{LOGO_SVG}</div>"
                f"<span style='font-family:var(--serif); font-weight:600; font-size:16px; "
                f"color:var(--mv-ink); white-space:nowrap;'>Med Venture</span></div>",
                unsafe_allow_html=True,
            )
        with nav_col:
            cols = st.columns([1] * len(desktop_nav_items) + [0.85])
            for col, (page_key, label) in zip(cols[:-1], desktop_nav_items):
                with col:
                    is_active = current_page == page_key or (page_key == "tests" and current_page == "test_detail")
                    if st.button(
                        label, key=f"nav_{page_key}", use_container_width=True,
                        type="primary" if is_active else "secondary",
                        icon=nav_icon(page_key),
                    ):
                        go_to(page_key)
            with cols[-1]:
                name = st.session_state.get("student_name", "")
                sid = st.session_state.get("student_id", "")
                initials = _avatar_initials(name)
                color = _avatar_color(sid, name)
                # A real st.button here (not just an HTML span) so the
                # avatar is clickable. Scoped with a compound class
                # selector so it reliably beats the generic plain-link nav
                # button rule and renders as a small solid circle instead.
                st.markdown(
                    f"<style>.st-key-top_nav .st-key-top_nav_avatar_btn button {{ "
                    f"background:{color} !important; border-color:{color} !important; color:#fff !important; "
                    f"border-radius:50% !important; width:34px !important; height:34px !important; "
                    f"min-height:34px !important; padding:0 !important; font-size:13px !important; "
                    f"font-weight:700 !important; }}</style>",
                    unsafe_allow_html=True,
                )
                # Avatar + a small decorative chevron beside it, matching
                # the reference design - the chevron itself is inert
                # markup (not a separate button); clicking the avatar
                # circle is what navigates to Profile, same as before.
                avatar_sub, chevron_sub = st.columns([2, 1])
                with avatar_sub:
                    with st.container(key="top_nav_avatar_btn"):
                        if st.button(initials, key="top_nav_avatar_click", help="Profile"):
                            go_to("profile")
                with chevron_sub:
                    st.markdown("<div class='mv-nav-avatar-chevron'>⌄</div>", unsafe_allow_html=True)

    # Mobile: a simplified header showing ONLY the logo (left) and the
    # hamburger/close toggle (right) - no inline avatar button any more,
    # matching the reference's collapsed mobile header exactly.
    #
    # Bug-1 fix: this used to be built with st.columns([1, 1]) for the
    # logo/toggle pair, which Streamlit stacks vertically below its own
    # ~640px responsive breakpoint (true for virtually every phone) - that
    # was exactly what pushed the hamburger button below the logo instead
    # of sitting beside it. Rebuilt below with NO st.columns: the logo
    # markdown and the toggle button (wrapped in its own small container)
    # are both plain, direct children of "mobile_top_bar", which the
    # ".st-key-mobile_top_bar" CSS in inject_global_css already forces
    # into a single flex row with the logo on the left and the toggle on
    # the right (justify-content: space-between) - the same pattern
    # already used successfully for the mentor top bar below.
    with st.container(key="mobile_top_bar"):
        is_open = st.session_state.get("student_mobile_menu_open", False)
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:6px; height:40px;'>"
            f"<div style='width:24px; height:24px; flex-shrink:0;'>{LOGO_SVG}</div>"
            f"<span style='font-family:var(--serif); font-weight:600; font-size:15px; "
            f"color:var(--mv-ink); white-space:nowrap;'>Med Venture</span></div>",
            unsafe_allow_html=True,
        )
        with st.container(key="mobile_top_bar_right"):
            if st.button("✕" if is_open else "☰", key="student_mobile_menu_toggle", help="Open menu" if not is_open else "Close menu"):
                st.session_state["student_mobile_menu_open"] = not is_open
                st.rerun()

    if st.session_state.get("student_mobile_menu_open", False):
        # Dim backdrop behind the slide-out drawer - purely visual (no JS,
        # so tapping it doesn't close the menu; the ☰/✕ toggle above is
        # the actual close control), matching the reference's dimmed
        # background around the open drawer.
        st.markdown("<div class='mv-drawer-backdrop'></div>", unsafe_allow_html=True)
        with st.container(key="mobile_menu_student"):
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:18px;'>"
                f"<div style='width:24px; height:24px;'>{LOGO_SVG}</div>"
                f"<span style='font-family:var(--serif); font-weight:600; font-size:15px; color:var(--mv-ink);'>Med Venture</span></div>",
                unsafe_allow_html=True,
            )
            for page_key, label in STUDENT_NAV:
                if st.button(label, key=f"mnav_{page_key}", use_container_width=True, icon=nav_icon(page_key)):
                    go_to(page_key)

    st.write("")


# =========================================================================
# Student: Exam Room / Question PDF
# =========================================================================

def _render_question_pdf(pdf_bytes, remaining_seconds=None, key_id=""):
    """Render the question paper inside Streamlit using PDF.js.

    This intentionally does not depend on st.pdf(). Some deployments may
    still run an older Streamlit build even after requirements changes.
    PDF.js renders the PDF pages onto canvases, so Chrome does not have to
    navigate an iframe to a PDF URL (which was the source of the
    'This page has been blocked by Chrome' problem).
    """
    import base64

    if not pdf_bytes:
        st.info("Question PDF is unavailable.")
        return

    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    safe_seconds = max(0, int(remaining_seconds)) if remaining_seconds is not None else None

    timer_block = ""
    timer_script = ""

    if safe_seconds is not None:
        timer_block = f"""
        <div style="
            position:sticky;
            top:0;
            z-index:20;
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:12px;
            margin:0 0 12px;
            padding:12px 16px;
            border:1px solid #d9e3df;
            border-radius:12px;
            background:#f8fbfa;
            font-family:system-ui,sans-serif;
            box-sizing:border-box;
        ">
          <div style="font-weight:700;">📄 Question Paper</div>
          <div id="exam-timer"
               style="font-weight:900;font-size:20px;letter-spacing:.03em;white-space:nowrap;">
            --:--:--
          </div>
        </div>
        """
        timer_script = f"""
        let remaining = {safe_seconds};
        const timerEl = document.getElementById("exam-timer");

        function updateExamTimer() {{
          if (!timerEl) return;

          if (remaining <= 0) {{
            timerEl.textContent = "00:00:00";
            try {{
              window.parent.location.reload();
            }} catch (e) {{
              try {{ window.top.location.reload(); }} catch (e2) {{}}
            }}
            return;
          }}

          const h = Math.floor(remaining / 3600);
          const m = Math.floor((remaining % 3600) / 60);
          const s = remaining % 60;

          timerEl.textContent =
            String(h).padStart(2, "0") + ":" +
            String(m).padStart(2, "0") + ":" +
            String(s).padStart(2, "0");

          remaining -= 1;
          setTimeout(updateExamTimer, 1000);
        }}

        updateExamTimer();
        """

    html = f"""
    <div id="pdf-root"
         style="font-family:system-ui,sans-serif;width:100%;box-sizing:border-box;">
      {timer_block}

      <div id="pdf-status"
           style="padding:12px 14px;border:1px solid #d9e3df;border-radius:12px;
                  background:#f8fbfa;margin-bottom:12px;">
        Loading question paper…
      </div>

      <div id="pdf-pages"
           style="display:flex;flex-direction:column;align-items:center;gap:16px;
                  width:100%;box-sizing:border-box;"></div>
    </div>

    <script>
    (async function() {{
      {timer_script}

      const pdfBase64 = "{b64}";
      const status = document.getElementById("pdf-status");
      const pages = document.getElementById("pdf-pages");

      try {{
        // PDF.js 3.x exposes the browser-friendly global `pdfjsLib`.
        const script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
        script.onload = async function() {{
          try {{
            pdfjsLib.GlobalWorkerOptions.workerSrc =
              "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

            const raw = atob(pdfBase64);
            const bytes = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) {{
              bytes[i] = raw.charCodeAt(i);
            }}

            const pdf = await pdfjsLib.getDocument({{data: bytes}}).promise;
            status.textContent = "📄 Question Paper · " + pdf.numPages + " page" +
              (pdf.numPages === 1 ? "" : "s");

            for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {{
              const page = await pdf.getPage(pageNumber);

              const wrapper = document.createElement("div");
              wrapper.style.width = "100%";
              wrapper.style.display = "flex";
              wrapper.style.justifyContent = "center";

              const canvas = document.createElement("canvas");
              canvas.style.display = "block";
              canvas.style.maxWidth = "100%";
              canvas.style.height = "auto";
              canvas.style.background = "white";
              canvas.style.borderRadius = "8px";
              canvas.style.boxShadow = "0 1px 5px rgba(0,0,0,.12)";

              wrapper.appendChild(canvas);
              pages.appendChild(wrapper);

              const baseViewport = page.getViewport({{scale: 1}});
              const maxWidth = Math.min(
                document.documentElement.clientWidth - 24,
                1100
              );
              const scale = Math.max(0.75, Math.min(1.5, maxWidth / baseViewport.width));
              const viewport = page.getViewport({{scale: scale}});

              const dpr = Math.min(window.devicePixelRatio || 1, 2);
              canvas.width = Math.floor(viewport.width * dpr);
              canvas.height = Math.floor(viewport.height * dpr);
              canvas.style.width = Math.floor(viewport.width) + "px";
              canvas.style.height = Math.floor(viewport.height) + "px";

              const ctx = canvas.getContext("2d", {{alpha: false}});
              await page.render({{
                canvasContext: ctx,
                viewport: viewport,
                transform: [dpr, 0, 0, dpr, 0, 0]
              }}).promise;
            }}
          }} catch (err) {{
            status.textContent = "Could not render the question paper.";
            status.style.color = "#b42318";
            console.error("PDF.js error:", err);
          }}
        }};

        script.onerror = function() {{
          status.textContent =
            "PDF viewer could not load. Please check the browser/network connection.";
          status.style.color = "#b42318";
        }};

        document.head.appendChild(script);
      }} catch (err) {{
        status.textContent = "Could not load the question paper.";
        status.style.color = "#b42318";
        console.error(err);
      }}
    }})();
    </script>
    """

    # The component height is large enough for the first part of the paper;
    # the inner viewer itself remains scrollable through the Streamlit page.
    components.html(html, height=900, scrolling=True)

def _parse_bd_dt(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


@st.dialog("📖 Exam Room", width="large")
def render_student_exam_room(sid, key):
    """Show the student's personal exam session in a modal exam room.

    The mentor exam window is checked only when the student STARTS. Once
    started, the persistent session expiry controls the room, so the full
    personal duration may cross the mentor exam-window end.
    """
    session = sh.get_exam_session(sid, key["key_id"])
    if not session:
        st.warning("This exam has not been opened yet.")
        if st.button("← Back", use_container_width=True):
            st.session_state.pop("exam_room_key_id", None)
            st.rerun()
        return

    expires = _parse_bd_dt(session.get("expires_at"))
    now = sh.now_bd()
    remaining = int((expires - now).total_seconds()) if expires else 0

    st.markdown(f"### {key.get('exam_name') or key['key_id']}")
    st.caption(
        f"{key['total_questions']} questions · "
        f"Personal duration: {_format_duration(key.get('duration_minutes', 0) or 0)}"
    )

    # Server-side expiry is authoritative; the browser countdown is only UI.
    if session.get("status") in ("completed", "submitted") or remaining <= 0:
        if session.get("status") == "started" and remaining <= 0:
            sh.set_exam_session_status(sid, key["key_id"], "expired")
        st.session_state["submit_key_id"] = key["key_id"]
        st.session_state.pop("exam_room_key_id", None)
        st.rerun()
        return

    pdf_id = key.get("question_pdf_file_id")
    if not pdf_id:
        st.error("This exam has no question PDF attached.")
        return

    pdf_bytes = sh.get_question_pdf_bytes(pdf_id)
    if not pdf_bytes:
        st.error("The question PDF could not be loaded.")
        return

    _render_question_pdf(pdf_bytes, remaining, key["key_id"])
    st.caption("When the timer reaches 00:00, the PDF will lock and you will be taken to OMR submission automatically.")

    if st.button("✅ Complete Exam & Go to OMR", key=f"complete_exam_{sid}_{key['key_id']}", type="primary", use_container_width=True):
        sh.set_exam_session_status(sid, key["key_id"], "completed")
        st.session_state["submit_key_id"] = key["key_id"]
        st.session_state.pop("exam_room_key_id", None)
        st.rerun()


# =========================================================================
# Student: Home
# =========================================================================

def page_home():
    sid = st.session_state["student_id"]
    name = st.session_state["student_name"]

    room_key_id = st.session_state.get("exam_room_key_id")
    if room_key_id:
        room_key = sh.get_answer_key_by_id(room_key_id)
        if room_key:
            render_student_exam_room(sid, room_key)
            return
        st.session_state.pop("exam_room_key_id", None)

    # Resume a legitimately started session even when the mentor's exam
    # window has already closed. The exam window controls START permission;
    # the student's personal duration controls the session itself.
    resume_session = sh.get_student_resume_session(sid)
    if resume_session:
        resume_key = sh.get_answer_key_by_id(resume_session.get("key_id"))
        if resume_key:
            if resume_session.get("status") == "started":
                st.session_state["exam_room_key_id"] = resume_key["key_id"]
                render_student_exam_room(sid, resume_key)
                return
            # The personal timer has ended (or the student already clicked
            # Complete). Send the student straight to the OMR page, even if
            # the global exam window is now closed.
            st.session_state["submit_key_id"] = resume_key["key_id"]
            go_to("tests")
            return

    st.markdown(f"### 👋 Welcome, {name}")

    active = cached_active_answer_key()
    # Real st.container(key=...) instead of a raw <div class='app-card'>
    # split across two st.markdown() calls - see the note above the
    # ".app-card" CSS rule for why the split version rendered an empty
    # styled bar on top of unstyled real content. Same fix applies to
    # every "card_..." container in this function.
    with st.container(key="card_home_active"):
        if active:
            already = sh.has_submitted(sid, active["key_id"])
            duration_display_min = active.get("duration_minutes") or int(
                (active["end_dt"] - active["start_dt"]).total_seconds() // 60
            )
            st.markdown(f"#### 🟢 Active Test: {active['exam_name'] or active['key_id']}")
            window_text = (
                f"{active['start_dt'].strftime('%d %b %Y, %I:%M %p')} → "
                f"{active['end_dt'].strftime('%d %b %Y, %I:%M %p')}"
                if active.get("start_dt") and active.get("end_dt") else "Exam window unavailable"
            )
            st.markdown(
                f"<div class='mv-exam-meta-grid'>"
                f"<div class='mv-exam-meta-primary'><span>Total Questions</span><strong>{active['total_questions']}</strong></div>"
                f"<div class='mv-exam-meta-primary'><span>Student Duration</span><strong>{_format_duration(duration_display_min)}</strong></div>"
                f"<div class='mv-exam-meta-secondary'><span>Exam Window</span><strong style='font-size:14px;line-height:1.35;'>{window_text}</strong></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if sh._to_bool(active.get("negative_marking", False)):
                st.caption(
                    f"Negative marking: −{float(active.get('negative_marks_value', 0.0) or 0.0):.2f} per wrong answer"
                )
            else:
                st.caption("No negative marking")
            if already:
                st.info("You already submitted this test. Check it in Tests & Results.")
            else:
                # Exams with a mentor-uploaded question PDF always use the
                # controlled PDF + personal countdown flow. Keep Quick OMR
                # Submit only as the legacy fallback for exams that have no PDF.
                if str(active.get("question_pdf_file_id", "") or "").strip():
                    if st.button("📖 Open Exam", type="primary", use_container_width=True):
                        try:
                            sh.start_exam_session(
                                sid, active["key_id"],
                                active.get("duration_minutes") or int(
                                    (active["end_dt"] - active["start_dt"]).total_seconds() // 60
                                ),
                            )
                        except ValueError as e:
                            st.error(str(e))
                        else:
                            st.session_state["exam_room_key_id"] = active["key_id"]
                            st.rerun()
                else:
                    if st.button("📤 Quick OMR Submit", type="primary", use_container_width=True):
                        go_to("tests", quick_submit=True)
        else:
            upcoming = cached_upcoming_answer_key()
            if upcoming:
                st.info(f"No test is active right now. Next up: **{upcoming['exam_name'] or upcoming['key_id']}** "
                        f"at **{upcoming['start_dt'].strftime('%Y-%m-%d %H:%M')}**.")
            else:
                st.info("No test is active or upcoming right now.")

    results = cached_results()
    my_results = results[results["student_id"] == sid] if not results.empty else results

    # Last Result + Overall Progress side by side - makes better use of
    # wide desktop screens, and costs nothing extra on mobile since
    # Streamlit already stacks columns vertically there (same scroll
    # length as two separate full-width cards would have been).
    col_last, col_progress = st.columns(2, gap="small")

    with col_last:
        with st.container(key="card_home_last"):
            st.markdown("#### 📊 Last Result")
            if my_results.empty:
                st.caption("You haven't submitted any test yet.")
            else:
                last = my_results.sort_values("timestamp", ascending=False).iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Marks", last["marks"])
                c2.metric("Correct", int(last["correct"]))
                c3.metric("Wrong", int(last["wrong_count"]))

    with col_progress:
        with st.container(key="card_home_progress"):
            st.markdown("#### 📈 Overall Progress")
            if my_results.empty:
                st.caption("Your progress will show up here after your first test.")
            else:
                tests_completed = len(my_results)
                avg_pct = round((my_results["marks"] / my_results["total"]).mean() * 100, 1)
                rank, out_of = cached_rank(sid)

                my_results_sorted = my_results.copy()
                my_results_sorted["ts"] = pd.to_datetime(my_results_sorted["timestamp"], errors="coerce")
                this_month = my_results_sorted[my_results_sorted["ts"].dt.month == date.today().month]
                last_month_num = date.today().month - 1 or 12
                last_month = my_results_sorted[my_results_sorted["ts"].dt.month == last_month_num]
                trend_html = ""
                if not this_month.empty and not last_month.empty:
                    this_avg = (this_month["marks"] / this_month["total"]).mean() * 100
                    last_avg = (last_month["marks"] / last_month["total"]).mean() * 100
                    diff = round(this_avg - last_avg, 1)
                    arrow = "↑" if diff >= 0 else "↓"
                    trend_html = f"<p style='margin:4px 0 0;'>{arrow} {abs(diff)}% {'better' if diff >= 0 else 'lower'} than last month</p>"

                rank_html = f"<span class='rank-badge rank-you'>🏆 Rank: {rank} / {out_of}</span>" if rank else ""

                st.markdown(
                    f"""
                    <div class='metric-row'>
                        <div class='metric-box'><div class='label'>Tests</div><div class='value'>{tests_completed}</div></div>
                        <div class='metric-box'><div class='label'>Average</div><div class='value'>{avg_pct}%</div></div>
                    </div>
                    <p style='margin-top:8px;'>{rank_html}</p>
                    {trend_html}
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("📊 Full Analysis", use_container_width=True, key="home_open_analysis"):
                    go_to("analysis")

    st.markdown(
        f"<div class='app-card' style='margin-top:2px;'>💡 <i>{motivation_for(sid)}</i></div>",
        unsafe_allow_html=True,
    )


# =========================================================================
# Student: Tests & Results (list + detail)
# =========================================================================

def render_omr_review(rows):
    """rows = omr_scanner.build_review_rows() output, already filtered to
    wrong + skipped only. A row with given == "MULTI" means the student
    touched 2+ bubbles on that question (double-touch) - it's always
    scored as wrong (marks deducted under negative marking) even if the
    correct answer happened to be one of the touched bubbles.
    """
    if not rows:
        st.success("🎉 No wrong or skipped answers!")
        return

    double_touch_qs = []
    html = ["<div>"]
    for row in rows:
        q, given, correct_ans, status = row["q"], row["given"], row["correct"], row["status"]
        is_double = (given == "MULTI")
        if is_double:
            double_touch_qs.append(q)

        tag = "<span class='omr-tag wrong-tag'>Wrong</span>" if status == "wrong" else "<span class='omr-tag skip-tag'>Skipped</span>"
        bubbles = ""
        for opt in ["A", "B", "C", "D"]:
            cls = "omr-bubble"
            if opt == correct_ans:
                cls += " correct"
            elif status == "wrong" and not is_double and opt == given:
                cls += " wrong"
            bubbles += f"<span class='{cls}'>{opt}</span>"

        q_label = f"Q{q}<span class='dt-star'>*</span>" if is_double else f"Q{q}"
        html.append(
            f"<div class='omr-row'><span class='omr-qnum'>{q_label}</span>{tag}<span>{bubbles}</span></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    if double_touch_qs:
        qs_text = ", ".join(f"{q}*" for q in double_touch_qs)
        word = "question" if len(double_touch_qs) == 1 else "questions"
        st.markdown(
            f"<div class='double-touch-note'>⚠️ <b>{qs_text}</b> - double touch on this {word}, "
            "marks deducted.</div>",
            unsafe_allow_html=True,
        )


def _render_readonly_digital_omr(result_row, key_row):
    """Read-only digital OMR for completed results. Mentor/student cannot edit here."""
    import json as _json
    total = int(result_row.get("total", 0) or 0)
    try:
        final = _json.loads(result_row.get("omr_final_answers_json") or "{}")
    except Exception:
        final = {}
    try:
        original = _json.loads(result_row.get("omr_original_answers_json") or "{}")
    except Exception:
        original = {}
    if not final:
        final = original
    cols = st.columns(2 if total > 40 else 1, gap="small")
    chunk = (total + 1)//2 if total > 40 else total
    ranges = [(1, chunk), (chunk+1, total)] if total > 40 else [(1,total)]
    for col,(start,end) in zip(cols,ranges):
        with col:
            for q in range(start,end+1):
                ans = final.get(str(q), final.get(q))
                ans = _normalise_answer_value(ans)
                bubbles = " ".join(f"<span class='digital-bubble {'selected' if ans == o else ''}'>{o}</span>" for o in 'ABCD')
                st.markdown(f"<div class='mv-compact-row'><b>Q{q:02d}</b>&nbsp;&nbsp;{bubbles}</div>", unsafe_allow_html=True)


def render_result_detail(result_row, key_row, mentor_mode=False):
    exam_name = key_row.get("exam_name") or result_row["key_id"]
    st.markdown(f"### {exam_name}")
    total, correct, wrong_count, skipped = (
        int(result_row["total"]), int(result_row["correct"]),
        int(result_row["wrong_count"]), int(result_row["skipped"]),
    )
    st.markdown(
        f"`Score: {correct}/{total}` &nbsp; `❌ Wrong {wrong_count}` &nbsp; "
        f"`⚪ Skipped {skipped}` &nbsp; `✅ Correct {correct}`"
    )
    import json as _json
    # Google Sheets may return the checkbox/boolean as the text "TRUE"/"FALSE".
    # Do not use bool("FALSE"), because any non-empty string is truthy in Python.
    if sh._to_bool(result_row.get("edited_by_mentor", False)):
        st.caption("ℹ️ This result was reviewed by your mentor.")

    # Mentor can review the submitted scan and record a yes/no cheating decision,
    # but cannot change the student's answer from this profile/result page.
    double_qs = []
    try:
        double_qs = _json.loads(result_row.get("omr_double_touch_json") or "[]")
    except Exception:
        double_qs = []
    if mentor_mode and double_qs:
        status = str(result_row.get("review_status", "") or "")
        st.markdown("#### ⚠️ Review Needed")
        st.caption(f"Scanner flagged Q{', Q'.join(map(str, double_qs))} for multiple marking. The student's submitted OMR is shown below for review.")
        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("YES — Cheating", type="primary", use_container_width=True, key=f"cheat_yes_{result_row['student_id']}_{result_row['key_id']}"):
                sh.set_result_review(result_row['student_id'], result_row['key_id'], "confirmed_cheating", "Mentor confirmed cheating / invalid marking.")
                clear_all_caches(); st.rerun()
        with rc2:
            if st.button("NO — Clear Review", use_container_width=True, key=f"cheat_no_{result_row['student_id']}_{result_row['key_id']}"):
                sh.set_result_review(result_row['student_id'], result_row['key_id'], "cleared", "Mentor reviewed and cleared the flag.")
                clear_all_caches(); st.rerun()
        if status:
            st.info(f"Current review status: **{status.replace('_',' ').title()}**")

    answer_string = key_row["answer_string"]

    try:
        wrong_details = _json.loads(result_row.get("wrong_details_json") or "{}")
    except Exception:
        wrong_details = {}
    try:
        skipped_qs = _json.loads(result_row.get("skipped_json") or "[]")
    except Exception:
        skipped_qs = []

    rows = []
    for q_str, detail in sorted(wrong_details.items(), key=lambda kv: int(kv[0])):
        rows.append({"q": int(q_str), "given": detail["given"], "correct": detail["correct"], "status": "wrong"})
    for q in skipped_qs:
        rows.append({"q": q, "given": None, "correct": answer_string[q - 1].upper(), "status": "skipped"})
    rows.sort(key=lambda r: r["q"])

    pdf_id = str(key_row.get("question_pdf_file_id", "") or "")
    if pdf_id:
        with st.expander("📄 View Question Paper"):
            try:
                pdf_bytes = sh.get_question_pdf_bytes(pdf_id)
                if pdf_bytes:
                    _render_question_pdf(pdf_bytes, None, result_row["key_id"])
                else:
                    st.info("Question PDF is unavailable.")
            except Exception:
                st.info("Question PDF is unavailable.")

    omr_photo_id = str(result_row.get("omr_photo_file_id", "") or "")
    if omr_photo_id:
        with st.expander("📷 View Submitted OMR Photo"):
            try:
                omr_bytes = sh.get_student_omr_image_bytes(omr_photo_id)
                if omr_bytes:
                    st.image(omr_bytes, caption="Original OMR photo submitted by you", use_container_width=True)
                else:
                    st.info("Submitted OMR photo is unavailable.")
            except Exception:
                st.info("Submitted OMR photo is unavailable.")

    st.markdown("#### 📝 Digital OMR · View Only")
    st.caption("Answers shown here are the student's submitted final answers. They cannot be changed from a student profile/result page.")
    _render_readonly_digital_omr(result_row, key_row)

    st.markdown("#### Wrong & Skipped Answers")
    render_omr_review(rows)


def _reset_submission_state():
    """Clear all transient state for one uploaded OMR photo."""
    for k in (
        "submit_file_sig", "submit_prepared_image", "submit_original_bytes",
        "submit_validation", "submit_calib_points", "submit_grid",
        "submit_detected_answers", "submit_final_answers", "submit_double_touch",
        "submit_review_ready", "submit_review_photo", "submit_review_focus_q",
        "submit_review_filter", "submit_omr_view",
    ):
        st.session_state.pop(k, None)


def _normalise_answer_value(value):
    """Normalize scanner/edit values to None, A/B/C/D or MULTI."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in ("", "NONE", "SKIP", "SKIPPED", "NULL"):
        return None
    if text in ("MULTI", "DOUBLE", "DOUBLE_TOUCH", "MULTIPLE"):
        return "MULTI"
    if text in ("A", "B", "C", "D"):
        return text
    return None


def _normalise_answers(raw, total_q):
    """Return a stable {question_number: answer} dict from scanner output."""
    out = {}
    if isinstance(raw, dict):
        for q in range(1, total_q + 1):
            value = raw.get(q, raw.get(str(q)))
            out[q] = _normalise_answer_value(value)
    elif isinstance(raw, (list, tuple)):
        for q in range(1, total_q + 1):
            value = raw[q - 1] if q - 1 < len(raw) else None
            out[q] = _normalise_answer_value(value)
    else:
        for q in range(1, total_q + 1):
            out[q] = None
    return out


def _coord_pair(value):
    """Best-effort extraction of an (x,y) point from common grid formats."""
    if isinstance(value, dict):
        for xk, yk in (("x", "y"), ("cx", "cy"), ("center_x", "center_y")):
            if xk in value and yk in value:
                try:
                    return float(value[xk]), float(value[yk])
                except Exception:
                    pass
        for key in ("center", "point", "coord", "coords", "xy"):
            if key in value:
                pt = _coord_pair(value[key])
                if pt:
                    return pt
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            if not isinstance(value[0], (list, tuple, dict)) and not isinstance(value[1], (list, tuple, dict)):
                return float(value[0]), float(value[1])
        except Exception:
            return None
    return None


def _extract_question_option_points(grid, total_q):
    """Extract Q1..Qn -> A/B/C/D centers without depending on one exact grid shape.

    The scanner module has evolved over time, so this intentionally accepts the
    common dict/list/tuple representations used by build_grid(). If a future
    scanner representation is different, the review still works in digital mode
    rather than guessing bubble locations.
    """
    labels = ["A", "B", "C", "D"]
    found = {}

    def walk(obj, q_hint=None):
        if isinstance(obj, dict):
            # Direct question keyed dictionaries: {1: {A: (x,y), ...}}
            for key, val in obj.items():
                q = None
                try:
                    if isinstance(key, int) or str(key).isdigit():
                        q = int(key)
                except Exception:
                    pass
                if q and 1 <= q <= total_q and isinstance(val, dict):
                    opts = {}
                    for opt in labels:
                        if opt in val:
                            pt = _coord_pair(val[opt])
                            if pt:
                                opts[opt] = pt
                    if len(opts) == 4:
                        found[q] = opts
                walk(val, q or q_hint)
            return

        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        if isinstance(obj, (list, tuple)):
            # A question entry represented as four coordinates.
            if len(obj) == 4:
                pts = [_coord_pair(v) for v in obj]
                if all(pts):
                    q = q_hint
                    if q and 1 <= q <= total_q:
                        found[q] = dict(zip(labels, pts))
            for idx, val in enumerate(obj):
                child_q = q_hint
                # If this is a top-level sequence of Q entries, its index is Q.
                if q_hint is None and idx < total_q:
                    child_q = idx + 1
                walk(val, child_q)

    walk(grid)

    # Last-resort flattening: if the grid is simply 4*n coordinate pairs,
    # preserve their natural question/option order.
    if len(found) < total_q:
        flat = []
        def flatten(obj):
            pt = _coord_pair(obj)
            if pt:
                flat.append(pt)
                return
            if isinstance(obj, np.ndarray):
                obj = obj.tolist()
            if isinstance(obj, dict):
                for v in obj.values():
                    flatten(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    flatten(v)
        flatten(grid)
        if len(flat) >= total_q * 4:
            candidate = {}
            for q in range(1, total_q + 1):
                start = (q - 1) * 4
                candidate[q] = dict(zip(labels, flat[start:start + 4]))
            # Only use the flattened interpretation if it gives every question.
            if len(candidate) == total_q:
                found = candidate

    return found


def _build_review_state(final_answers, original_answers):
    """Build the PRE-SUBMISSION state from the CURRENT editable answers.

    Important: issues are no longer immutable. A scanner-detected MULTI is an
    initial warning only. Once the student selects exactly one A/B/C/D answer,
    that question is resolved and disappears from Review Issues. Likewise, a
    skipped question disappears immediately after an answer is selected.
    """
    rows = []
    total_q = len(final_answers)

    for q in range(1, total_q + 1):
        answer = _normalise_answer_value(final_answers.get(q))
        original = _normalise_answer_value(original_answers.get(q))

        if answer == "MULTI":
            status = "double"
        elif answer is None:
            status = "skipped"
        else:
            status = "answered"

        rows.append({
            "q": q,
            "given": answer,
            "original_given": original,
            "status": status,
            "was_edited": answer != original,
        })
    return rows


def _digital_omr_pick_answer(q, opt, total_q):
    """Callback for one real HTML/Streamlit OMR bubble."""
    answers = dict(st.session_state.get("submit_final_answers", {}))
    for n in range(1, total_q + 1):
        answers.setdefault(n, None)

    current = _normalise_answer_value(answers.get(q))
    answers[q] = None if current == opt else opt
    st.session_state["submit_final_answers"] = answers

    # After fixing an issue, Review → Next Issue naturally moves forward.
    st.session_state["submit_review_focus_q"] = q + 1


def _render_digital_question_row(q, answer, original, status, total_q, compact=False):
    """Render one editable OMR row with real circular A/B/C/D controls."""
    wrapper_class = "digital-q-issue" if status in ("skipped", "double") else ("digital-q-edited" if original != answer else "")
    with st.container(key=f"digital_omr_q_{q}"):
        cols = st.columns([0.72, 1, 1, 1, 1], gap="small")
        with cols[0]:
            badge = ""
            if status == "skipped":
                badge = " ⏭"
            elif status == "double":
                badge = " ⚠"
            label = f"Q{q}{badge}"
            if wrapper_class:
                st.markdown(f"<div class='{wrapper_class}'><span style='font-family:var(--mono);font-size:11px;font-weight:800;'>{label}</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Q{q}**")

        for idx, opt in enumerate(("A", "B", "C", "D"), start=1):
            selected = answer == opt
            with cols[idx]:
                st.button(
                    opt,
                    key=f"digital_omr_q{q}_{opt}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                    on_click=_digital_omr_pick_answer,
                    args=(q, opt, total_q),
                    help=f"Select {opt} for Question {q}" if not selected else f"Question {q}: {opt} selected. Click again to clear.",
                )

        if not compact and status == "skipped":
            st.caption("⏭ Scanner found no reliable fill — choose the correct bubble above.")
        elif not compact and status == "double":
            st.caption("⚠ Scanner detected more than one filled option — select the single final answer above.")
        elif not compact and original != answer:
            detected = "Multiple" if original == "MULTI" else (original or "Skipped")
            st.caption(f"Scanner detected: **{detected}** · Final answer: **{answer}**")


def _issue_rows_from_review(review_rows):
    return [r for r in review_rows if r["status"] in ("skipped", "double")]


def _render_normal_omr_view(review_rows, total_q):
    """Render the editable Digital OMR in the same block pattern as the paper."""
    row_by_q = {r["q"]: r for r in review_rows}
    physical_total = 50 if total_q in (40, 50) else 100
    block_starts = list(range(1, physical_total + 1, 25))
    # For a 40-question exam, Q41-50 are physically present on the paper but
    # are intentionally not rendered because they are outside the exam.
    block_starts = [s for s in block_starts if s <= total_q]

    block_cols = st.columns(len(block_starts), gap="small")
    for col, start_q in zip(block_cols, block_starts):
        end_q = min(start_q + 24, total_q)
        with col:
            st.markdown(
                f"<div class='digital-omr-block'><div class='digital-omr-block-title'>Questions {start_q}–{end_q}</div>",
                unsafe_allow_html=True,
            )
            for q in range(start_q, end_q + 1):
                row = row_by_q[q]
                _render_digital_question_row(
                    q=q,
                    answer=row["given"],
                    original=row["original_given"],
                    status=row["status"],
                    total_q=total_q,
                    compact=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


def _render_review_issues_view(review_rows, total_q):
    """Show only unresolved questions, with live filters and Next Issue."""
    issues = _issue_rows_from_review(review_rows)
    skipped = [r for r in issues if r["status"] == "skipped"]
    doubles = [r for r in issues if r["status"] == "double"]

    if not issues:
        st.success("✅ All detected issues reviewed")
        st.caption("You can switch back to All Questions anytime to inspect the full OMR.")
        return

    st.markdown("#### 🔎 Review Issues")
    filter_options = [
        f"⚠️ All Issues ({len(issues)})",
        f"⏭ Skipped ({len(skipped)})",
        f"⚠️ Double Touch ({len(doubles)})",
    ]
    selected = st.radio(
        "Issue filter",
        filter_options,
        horizontal=True,
        label_visibility="collapsed",
        key="submit_review_filter",
    )

    if selected.startswith("⏭"):
        visible = skipped
    elif selected.startswith("⚠️ Double"):
        visible = doubles
    else:
        visible = issues

    if not visible:
        st.info("No issue in this filter.")
        return

    # Focus is a question number, not an index, so it survives live list changes.
    focus_hint = int(st.session_state.get("submit_review_focus_q", visible[0]["q"]))
    visible_qs = [r["q"] for r in visible]
    focused_q = next((q for q in visible_qs if q >= focus_hint), visible_qs[0])
    focus_pos = visible_qs.index(focused_q)

    nav1, nav2, nav3 = st.columns([1, 1.4, 1])
    with nav1:
        if st.button("← Previous Issue", use_container_width=True, disabled=len(visible) <= 1):
            st.session_state["submit_review_focus_q"] = visible_qs[(focus_pos - 1) % len(visible)]
            st.rerun()
    with nav2:
        st.markdown(f"<div style='text-align:center;padding:8px 0;font-weight:700;'>Issue {focus_pos + 1} of {len(visible)} · Q{focused_q}</div>", unsafe_allow_html=True)
    with nav3:
        if st.button("Next Issue →", type="primary", use_container_width=True, disabled=len(visible) <= 1):
            st.session_state["submit_review_focus_q"] = visible_qs[(focus_pos + 1) % len(visible)]
            st.rerun()

    # Focused issue first.
    focused = next(r for r in visible if r["q"] == focused_q)
    with st.container(key="card_review_current_issue"):
        _render_digital_question_row(
            q=focused["q"],
            answer=focused["given"],
            original=focused["original_given"],
            status=focused["status"],
            total_q=total_q,
            compact=False,
        )

    others = [r for r in visible if r["q"] != focused_q]
    if others:
        st.markdown("##### Remaining issues")
        for row in others:
            _render_digital_question_row(
                q=row["q"],
                answer=row["given"],
                original=row["original_given"],
                status=row["status"],
                total_q=total_q,
                compact=True,
            )
            st.divider()


def _render_interactive_omr_review(img_bgr, grid_points, detected_answers, final_answers, double_qs, radius):
    """Show the student's original OMR beside an editable Digital OMR."""
    total_q = len(final_answers)
    review_rows = _build_review_state(final_answers, detected_answers)

    skipped_count = sum(r["status"] == "skipped" for r in review_rows)
    double_count = sum(r["status"] == "double" for r in review_rows)
    answered_count = total_q - skipped_count - double_count
    issue_count = skipped_count + double_count

    st.markdown("""
        <div class='digital-omr-title'>
            <div>
                <div class='digital-omr-title-main'>🖥️ OMR Review</div>
                <div class='digital-omr-sub'>Your scanned sheet on the left · editable Digital OMR on the right</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Answered", answered_count)
    c2.metric("Skipped", skipped_count)
    c3.metric("Double Touch", double_count)
    c4.metric("Needs Review", issue_count)

    view = st.radio(
        "Digital OMR view",
        ["🖥️ All Questions", "🔎 Review Issues"],
        horizontal=True,
        key="submit_omr_view",
        label_visibility="collapsed",
    )

    left, right = st.columns([0.92, 1.55], gap="medium")
    with left:
        st.markdown("<div class='omr-photo-card'><div class='omr-photo-label'>📷 Original OMR</div>", unsafe_allow_html=True)
        original_bytes = st.session_state.get("submit_original_bytes")
        if original_bytes:
            try:
                original_img = ImageOps.exif_transpose(Image.open(io.BytesIO(original_bytes)).convert("RGB"))
                st.image(original_img, use_container_width=True)
            except Exception:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("This is the exact photo you submitted. The Digital OMR is what will be submitted after your corrections.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='digital-omr-shell'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:12px;color:var(--mv-muted);margin-bottom:10px;'>"
            "Tap a bubble to edit. A selected bubble is the final answer."
            "</div>",
            unsafe_allow_html=True,
        )
        if view == "🖥️ All Questions":
            _render_normal_omr_view(review_rows, total_q)
        else:
            _render_review_issues_view(review_rows, total_q)
        st.markdown("</div>", unsafe_allow_html=True)

    return review_rows

def page_tests_results():
    sid = st.session_state["student_id"]

    view_key_id = st.session_state.get("view_key_id")
    if view_key_id:
        results = cached_results()
        keys_df = cached_answer_keys()
        match = results[(results["student_id"] == sid) & (results["key_id"] == view_key_id)]
        key_match = keys_df[keys_df["key_id"] == view_key_id]
        if match.empty or key_match.empty:
            st.warning("Result not found.")
        else:
            if st.button("← Back to Tests & Results"):
                st.session_state["view_key_id"] = None
                st.session_state.pop("submit_key_id", None)
                st.rerun()
            render_result_detail(match.iloc[0], key_match.iloc[0])
        return

    st.markdown("### 📝 Submit OMR / Test History")

    requested_submit_key = st.session_state.get("submit_key_id")
    if requested_submit_key:
        requested_key = sh.get_answer_key_by_id(requested_submit_key)
        if requested_key:
            active = requested_key
            try:
                active["start_dt"] = datetime.strptime(f"{active['date']} 00:00", "%Y-%m-%d %H:%M")
                active["end_dt"] = active["start_dt"]
            except Exception:
                pass
        else:
            active = cached_active_answer_key()
    else:
        active = cached_active_answer_key()

    with st.container(key="card_submit_omr"):
        st.markdown("#### 📤 Submit Your OMR Sheet")
        if not active:
            st.info("No test is active right now.")
        elif sh.has_submitted(sid, active["key_id"]):
            st.success("✅ You've already submitted this test. Duplicate submissions aren't allowed.")
        else:
            total_q = active["total_questions"]
            st.caption(f"Active test: **{active['exam_name'] or active['key_id']}** · {total_q} questions")
            uploaded = st.file_uploader(
                "Upload a clear, straight photo of your FULL filled OMR sheet (camera or gallery). Make sure all 4 corners of the sheet are visible in the frame.",
                type=["png", "jpg", "jpeg"], key="omr_upload",
            )

            if uploaded is None:
                _reset_submission_state()
            else:
                file_sig = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.get("submit_file_sig") != file_sig:
                    _reset_submission_state()
                    st.session_state["submit_file_sig"] = file_sig

                if "submit_prepared_image" not in st.session_state:
                    pil_img = ImageOps.exif_transpose(Image.open(uploaded).convert("RGB"))
                    original_bytes = uploaded.getvalue()
                    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    ok, errors, warnings_ = omr_scanner.validate_omr_image(orig_bgr)
                    proc_bgr = omr_scanner.resize_max_dim(orig_bgr) if ok else orig_bgr
                    st.session_state["submit_prepared_image"] = proc_bgr
                    st.session_state["submit_original_bytes"] = original_bytes
                    st.session_state["submit_validation"] = (ok, errors, warnings_)

                img_bgr = st.session_state["submit_prepared_image"]
                ok, errors, warnings_ = st.session_state["submit_validation"]
                display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                display_pil = Image.fromarray(display_rgb)

                if not ok:
                    st.image(display_rgb, caption="Your uploaded sheet - full photo", use_container_width=True)
                    for e in errors:
                        st.error(e)
                else:
                    for w in warnings_:
                        st.warning(w)

                    points_info = omr_scanner.calibration_points_info(total_q)
                    calib_points = st.session_state.get("submit_calib_points", [])
                    total_points = len(points_info)

                    if not st.session_state.get("submit_review_ready"):
                        st.markdown("#### 🎯 Calibrate Your Sheet")
                        st.caption(
                            f"Tap the exact CENTER of these {total_points} bubbles on YOUR photo above, in order (a top AND bottom point for every question block, so the reading stays accurate even if the sheet is a little curved, folded, or tilted in the photo)."
                        )
                        if len(calib_points) < total_points:
                            step = points_info[len(calib_points)]
                            st.markdown(
                                f"<span class='calib-step-badge'>Step {len(calib_points) + 1} of {total_points}</span> &nbsp; Now tap: **{step['full']}**",
                                unsafe_allow_html=True,
                            )
                            coords = streamlit_image_coordinates(display_pil, key=f"submit_calib_img_{file_sig}")
                            if coords is not None:
                                pt = (coords["x"], coords["y"])
                                if not calib_points or calib_points[-1] != pt:
                                    calib_points.append(pt)
                                    st.session_state["submit_calib_points"] = calib_points
                                    st.rerun()
                        else:
                            st.success(f"✅ All {total_points} points marked!")
                            calibration = {info["key"]: pt for info, pt in zip(points_info, calib_points)}
                            grid = omr_scanner.build_grid(calibration, total_questions=total_q)
                            radius = omr_scanner.compute_bubble_radius(img_bgr)
                            detected = _normalise_answers(omr_scanner.read_answers(img_bgr, grid, radius=radius), total_q)
                            double_qs = [q for q, a in detected.items() if a == "MULTI"]
                            # Keep the raw grid in state; it is used only for the interactive overlay.
                            st.session_state["submit_grid"] = grid
                            st.session_state["submit_detected_answers"] = detected
                            st.session_state["submit_final_answers"] = dict(detected)
                            st.session_state["submit_double_touch"] = double_qs
                            st.session_state["submit_review_ready"] = True
                            st.rerun()
                    else:
                        grid = st.session_state.get("submit_grid")
                        detected = st.session_state.get("submit_detected_answers", {})
                        final_answers = st.session_state.get("submit_final_answers", dict(detected))
                        double_qs = st.session_state.get("submit_double_touch", [])
                        grid_points = _extract_question_option_points(grid, total_q)
                        radius = omr_scanner.compute_bubble_radius(img_bgr)

                        review_rows = _render_interactive_omr_review(
                            img_bgr, grid_points, detected, final_answers, double_qs, radius
                        )

                        st.divider()
                        st.markdown("#### ✅ Ready to Submit?")
                        unresolved_double = [
                            r for r in _build_review_state(final_answers, detected)
                            if r["status"] == "double"
                        ]
                        if unresolved_double:
                            st.warning(
                                f"⚠️ {len(unresolved_double)} double-touch question(s) still need a final A/B/C/D selection before submission."
                            )

                        submitting_key = f"submitting_{file_sig}"
                        is_submitting = st.session_state.get(submitting_key, False)
                        cb1, cb2 = st.columns(2)
                        with cb1:
                            if st.button("🔄 Redo Calibration Points", use_container_width=True, disabled=is_submitting):
                                _reset_submission_state()
                                st.session_state["submit_file_sig"] = file_sig
                                st.rerun()
                        with cb2:
                            submit_clicked = st.button("📤 Confirm & Submit", type="primary", use_container_width=True, disabled=is_submitting)

                        if submit_clicked and not is_submitting:
                            st.session_state[submitting_key] = True
                            try:
                                with st.spinner("Scoring your final answers and saving your OMR..."):
                                    submit_key_id = active["key_id"]
                                    active_now = sh.get_answer_key_by_id(submit_key_id)
                                    if not active_now:
                                        st.error("This exam could not be loaded. Your result can't be recorded.")
                                    elif sh.has_submitted(sid, active_now["key_id"]):
                                        st.warning("You've already submitted this test.")
                                    else:
                                        # The student's final Digital OMR choices are authoritative.
                                        # Scanner MULTI detections remain stored as audit metadata only; once
                                        # the student chooses one bubble, that issue is resolved and must not
                                        # be forcibly converted back to MULTI during scoring.
                                        scoring_answers = dict(final_answers)
                                        result = omr_scanner.score_answers(
                                            scoring_answers,
                                            active_now["answer_string"],
                                            negative_marking=active_now.get("negative_marking", False),
                                            negative_value=active_now.get("negative_marks_value", 0.0),
                                        )
                                        # Preserve the scanner's first-pass detection separately
                                        # from the student's editable final answer. This is the audit
                                        # trail that keeps double-touch negative marking enforceable.
                                        result["omr_original_answers"] = dict(detected)
                                        result["omr_final_answers"] = dict(final_answers)
                                        result["omr_double_touch"] = list(double_qs)
                                        neg_enabled = sh._to_bool(active_now.get("negative_marking", False))
                                        if neg_enabled:
                                            neg_per_wrong = max(0.0, float(active_now.get("negative_marks_value", 0.0) or 0.0))
                                            result["marks"] = round(float(result.get("correct", 0)) - float(result.get("wrong_count", 0)) * neg_per_wrong, 4)
                                            result["negative_marking"] = True
                                            result["negative_value"] = neg_per_wrong

                                        saved = sh.append_result_if_not_submitted(
                                            sid,
                                            st.session_state["student_name"],
                                            submit_key_id,
                                            result,
                                            omr_photo_bytes=st.session_state.get("submit_original_bytes"),
                                            omr_photo_name=uploaded.name,
                                        )
                                        clear_all_caches()
                                        if not saved:
                                            st.warning("You've already submitted this test (from another tab or device).")
                                        else:
                                            sh.set_exam_session_status(sid, submit_key_id, "submitted")
                                            st.session_state.pop("submit_key_id", None)
                                            _reset_submission_state()
                                            st.success("✅ Result saved!")
                                            with st.container(key="card_submit_result"):
                                                r1, r2, r3, r4 = st.columns(4)
                                                r1.metric("Correct ✅", result["correct"])
                                                r2.metric("Wrong ❌", result["wrong_count"])
                                                r3.metric("Skipped ⚪", result["skipped"])
                                                r4.metric("🏆 Marks", result["marks"])
                                                if sh._to_bool(result.get("negative_marking", False)):
                                                    st.caption(
                                                        f"Negative marking: {result['wrong_count']} wrong × {float(result.get('negative_value', 0.0)):.2f} deducted · skipped = no deduction"
                                                    )
                                            rows = omr_scanner.build_review_rows(scoring_answers, active_now["answer_string"])
                                            review_rows = [r for r in rows if r["status"] in ("wrong", "skipped")]
                                            st.markdown("#### Review")
                                            render_omr_review(review_rows)
                            except Exception as e:
                                st.error("Something went wrong while saving your result and it was NOT recorded. Please try submitting again.")
                                st.caption(f"Technical detail: {e}")
                            finally:
                                st.session_state[submitting_key] = False

    st.markdown("#### 📋 Test History")
    results = cached_results()
    keys_df = cached_answer_keys()
    my_results = results[results["student_id"] == sid] if not results.empty else results
    if my_results.empty:
        st.caption("No tests submitted yet.")
        return
    my_results = my_results.sort_values("timestamp", ascending=False)
    with st.container(key="test_history_table"):
        header_cols = st.columns([2.4, 1.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8])
        for c, label in zip(header_cols, ["Exam", "Date", "Total", "Correct", "Wrong", "Skipped", "Marks", ""]):
            c.markdown(f"**{label}**")
        for _, row in my_results.iterrows():
            key_match = keys_df[keys_df["key_id"] == row["key_id"]]
            exam_name = key_match.iloc[0]["exam_name"] if not key_match.empty and key_match.iloc[0]["exam_name"] else row["key_id"]
            cols = st.columns([2.4, 1.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8])
            cols[0].write(exam_name)
            cols[1].write(str(row["timestamp"]).split(" ")[0])
            cols[2].write(int(row["total"]))
            cols[3].write(int(row["correct"]))
            cols[4].write(int(row["wrong_count"]))
            cols[5].write(int(row["skipped"]))
            cols[6].write(row["marks"])
            if cols[7].button("View", key=f"view_{row['key_id']}"):
                st.session_state["view_key_id"] = row["key_id"]
                st.rerun()


def _exam_name_from_keys(keys_df, key_id):
    if keys_df is not None and not keys_df.empty:
        match = keys_df[keys_df["key_id"] == key_id]
        if not match.empty:
            name = match.iloc[0].get("exam_name")
            if name:
                return str(name)
    return str(key_id)


def _safe_pct(numerator, denominator, digits=1):
    try:
        n = float(numerator)
        d = float(denominator)
        if d <= 0:
            return 0.0
        return round((n / d) * 100, digits)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def render_student_analysis(sid, name, *, mentor_mode=False):
    """Shared, paginated analysis screen for a single student.

    Mentor mode only changes navigation/labels; all result filtering is still
    done by the supplied student_id so a mentor never sees mixed students.
    """
    results = cached_results()
    keys_df = cached_answer_keys()

    if results.empty:
        student_results = results
    else:
        student_results = results[results["student_id"].astype(str) == str(sid)].copy()

    render_student_header(sid, name, heading_level=3)

    if mentor_mode:
        if st.button("← Back to Students", use_container_width=False, key="back_to_students_analysis"):
            st.session_state.pop("mentor_analysis_sid", None)
            go_to("mentor", mentor_page="m_students")
    else:
        if st.button("← Back to Home", use_container_width=False, key="back_to_home_analysis"):
            go_to("home")

    if student_results.empty:
        st.info("This student hasn't submitted any test yet.")
        return

    student_results["_total_num"] = pd.to_numeric(student_results["total"], errors="coerce").fillna(0)
    student_results["_marks_num"] = pd.to_numeric(student_results["marks"], errors="coerce").fillna(0)
    student_results["_correct_num"] = pd.to_numeric(student_results["correct"], errors="coerce").fillna(0)

    tests_count = len(student_results)
    avg_pct = round((student_results["_marks_num"] / student_results["_total_num"].replace(0, np.nan)).mean() * 100, 1)
    avg_pct = 0.0 if pd.isna(avg_pct) else avg_pct
    best_score = round(float(student_results["_marks_num"].max()), 2)
    accuracy = _safe_pct(student_results["_correct_num"].sum(), student_results["_total_num"].sum())
    rank, out_of = cached_rank(sid)

    st.markdown(
        f"""
        <div class='metric-row' style='margin-bottom:14px;'>
            <div class='metric-box'><div class='label'>📝 Total Tests</div><div class='value'>{tests_count}</div></div>
            <div class='metric-box'><div class='label'>📈 Average Score</div><div class='value'>{avg_pct}%</div></div>
            <div class='metric-box'><div class='label'>🏆 Best Score</div><div class='value'>{best_score:g}</div></div>
            <div class='metric-box'><div class='label'>🎯 Accuracy</div><div class='value'>{accuracy}%</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rank:
        st.caption(f"🏅 Current rank: **#{rank} / {out_of}**")

    st.markdown("#### 📋 Test History & Analysis")
    student_results = student_results.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Pagination keeps even a 100+ test student page fast and compact.
    page_size = 15
    total_pages = max(1, (len(student_results) + page_size - 1) // page_size)
    state_key = "mentor_analysis_page" if mentor_mode else "student_analysis_page"
    current_page = int(st.session_state.get(state_key, 1))
    current_page = min(max(current_page, 1), total_pages)

    if total_pages > 1:
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.button("← Previous", disabled=current_page <= 1, key=f"{state_key}_prev", use_container_width=True):
                st.session_state[state_key] = current_page - 1
                st.rerun()
        with p2:
            st.markdown(f"<div style='text-align:center; padding-top:8px;'>Page <b>{current_page}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
        with p3:
            if st.button("Next →", disabled=current_page >= total_pages, key=f"{state_key}_next", use_container_width=True):
                st.session_state[state_key] = current_page + 1
                st.rerun()

    start_i = (current_page - 1) * page_size
    visible = student_results.iloc[start_i:start_i + page_size]

    for idx, row in visible.iterrows():
        key_id = row["key_id"]
        exam_name = _exam_name_from_keys(keys_df, key_id)
        total = int(row["_total_num"])
        correct = int(row["_correct_num"])
        wrong_value = pd.to_numeric(row.get("wrong_count", 0), errors="coerce")
        skipped_value = pd.to_numeric(row.get("skipped", 0), errors="coerce")
        wrong = int(wrong_value) if pd.notna(wrong_value) else 0
        skipped = int(skipped_value) if pd.notna(skipped_value) else 0
        marks = row["marks"]
        pct = _safe_pct(row["_marks_num"], total)
        date_text = str(row["timestamp"]).split(" ")[0]

        with st.container(key=f"acard_{'m' if mentor_mode else 's'}_{sid}_{key_id}_{idx}"):
            c1, c2, c3, c4, c5 = st.columns([2.8, 1.0, 1.0, 1.0, 0.9])
            with c1:
                st.markdown(f"<div class='analysis-title'>{exam_name}</div><div class='analysis-subtle'>{date_text} · {pct}%</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='history-value'><span>Marks</span><b>{marks}</b></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='history-value'><span>Correct</span><b>{correct}</b></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div class='history-value'><span>Wrong</span><b>{wrong}</b></div>", unsafe_allow_html=True)
            with c5:
                if st.button("View", key=f"analysis_view_{'m' if mentor_mode else 's'}_{sid}_{key_id}_{idx}", use_container_width=True):
                    if mentor_mode:
                        st.session_state["mentor_analysis_view_key_id"] = key_id
                    else:
                        st.session_state["analysis_view_key_id"] = key_id
                    st.rerun()
            st.markdown(f"<div class='analysis-subtle'>Total: {total} · Skipped: {skipped} · Accuracy: {_safe_pct(correct, total)}%</div>", unsafe_allow_html=True)


def page_student_analysis():
    sid = st.session_state["student_id"]
    name = st.session_state.get("student_name", sid)
    view_key_id = st.session_state.get("analysis_view_key_id")

    if view_key_id:
        results = cached_results()
        keys_df = cached_answer_keys()
        match = results[(results["student_id"].astype(str) == str(sid)) & (results["key_id"] == view_key_id)]
        key_match = keys_df[keys_df["key_id"] == view_key_id]
        if st.button("← Back to My Analysis", key="back_to_my_analysis"):
            st.session_state.pop("analysis_view_key_id", None)
            st.rerun()
        if match.empty or key_match.empty:
            st.warning("Result not found.")
        else:
            render_student_header(sid, name, heading_level=3)
            render_result_detail(match.iloc[0], key_match.iloc[0])
        return

    render_student_analysis(sid, name, mentor_mode=False)


def page_mentor_student_analysis():
    sid = st.session_state.get("mentor_analysis_sid")
    if not sid:
        go_to("mentor", mentor_page="m_students")
        return

    students = cached_students()
    student_match = students[students["student_id"].astype(str) == str(sid)] if not students.empty else students
    name = student_match.iloc[0]["name"] if not student_match.empty else st.session_state.get("mentor_analysis_name", sid)
    view_key_id = st.session_state.get("mentor_analysis_view_key_id")

    if view_key_id:
        results = cached_results()
        keys_df = cached_answer_keys()
        match = results[(results["student_id"].astype(str) == str(sid)) & (results["key_id"] == view_key_id)]
        key_match = keys_df[keys_df["key_id"] == view_key_id]
        if st.button("← Back to Student Analysis", key="back_to_mentor_analysis"):
            st.session_state.pop("mentor_analysis_view_key_id", None)
            st.rerun()
        if match.empty or key_match.empty:
            st.warning("Result not found.")
        else:
            render_student_header(sid, name, heading_level=3)
            render_result_detail(match.iloc[0], key_match.iloc[0], mentor_mode=True)
        return

    render_student_analysis(sid, name, mentor_mode=True)


# =========================================================================
# Leaderboard (shared renderer for both the Student page and the Mentor panel)
# =========================================================================

def _rank_class(rank):
    return {1: "rank-gold", 2: "rank-silver", 3: "rank-bronze"}.get(rank, "")


def _rank_icon(rank):
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


def render_leaderboard_stats(df, mode):
    if mode == "Overall":
        total_students = len(df)
        avg_score = round(df["avg_percent"].mean(), 1) if not df.empty else 0
        highest_score = df["best_score"].max() if not df.empty else 0
    else:
        total_students = df["student_id"].nunique()
        avg_score = round(df["marks"].mean(), 1) if not df.empty else 0
        highest_score = df["marks"].max() if not df.empty else 0
    st.markdown(
        f"""
        <div class='metric-row' style='margin-bottom:14px;'>
            <div class='metric-box'><div class='label'>👥 Total Students</div><div class='value'>{total_students}</div></div>
            <div class='metric-box'><div class='label'>📈 Average Score</div><div class='value'>{avg_score}</div></div>
            <div class='metric-box'><div class='label'>🏆 Highest Score</div><div class='value'>{highest_score}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_leaderboard_rows(df, mode, sid=None, key_suffix="student"):
    with st.container(key=f"leaderboard_table_{key_suffix}"):
        for _, row in df.head(50).iterrows():
            rank = int(row["rank"])
            is_me = sid is not None and row["student_id"] == sid
            css_class = "lb-row me" if is_me else "lb-row"
            icon = _rank_icon(rank) if rank <= 3 else f"#{rank}"
            badge_class = _rank_class(rank)
            avatar_html = render_avatar(row["student_id"], row["student"], size=26, font_size=11)
            name_html = (
                f"<span style='display:inline-flex; align-items:center; gap:7px;'>"
                f"{avatar_html}<span>{row['student']}{' (You)' if is_me else ''}</span></span>"
            )

            if mode == "Overall":
                trend = row.get("trend")
                trend_html = "<span style='opacity:.4;'>—</span>"
                if trend is not None and pd.notna(trend):
                    arrow = "↑" if trend >= 0 else "↓"
                    color = "#22c55e" if trend >= 0 else "#ef4444"
                    trend_html = f"<span style='color:{color}; font-weight:700;'>{arrow} {abs(trend)}%</span>"
                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <span class="rank-badge {badge_class}">{icon}</span>
                        <span style="flex:1.5; font-weight:{'700' if is_me else '500'};">{name_html}</span>
                        <span style="flex:0.8; opacity:.85;">Tests: <b>{int(row['exams_taken'])}</b></span>
                        <span style="flex:0.8; opacity:.85;">Best: <b>{row['best_score']}</b></span>
                        <span style="flex:0.9; opacity:.85;">Avg: <b>{row['avg_percent']}%</b></span>
                        <span style="flex:0.9; opacity:.7;">Acc: {row['accuracy']}%</span>
                        <span style="flex:0.8; text-align:right;">{trend_html}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                accuracy_val = row.get("accuracy", "-")
                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <span class="rank-badge {badge_class}">{icon}</span>
                        <span style="flex:1; font-weight:{'700' if is_me else '500'};">{name_html}</span>
                        <span>Score: <b>{row['marks']}</b></span>
                        <span style="opacity:.7;">Accuracy: {accuracy_val}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def render_leaderboard(sid=None, key_suffix="student"):
    """Shared leaderboard renderer. sid=None -> mentor view (no personal
    'Your Rank' footer); sid='S0001' -> student view."""
    mode = st.radio(
        "View", ["Overall", "Test-wise"], horizontal=True,
        label_visibility="collapsed", key=f"lb_mode_{key_suffix}",
    )

    if mode == "Test-wise":
        keys_df = cached_answer_keys()
        if keys_df.empty:
            st.info("No tests have been created yet.")
            return
        keys_df = keys_df.iloc[::-1].reset_index(drop=True)
        options = {}
        for _, row in keys_df.iterrows():
            label = f"{row.get('exam_name') or row['key_id']} | {row['date']}"
            options[label] = row["key_id"]
        choice = st.selectbox("Choose a test", list(options.keys()), key=f"lb_test_choice_{key_suffix}")
        key_id = options[choice]
        df = sh.get_leaderboard_by_key(key_id)
    else:
        df = sh.get_overall_leaderboard()
        key_id = None

    if df is None or df.empty:
        st.info("No results yet for this view.")
        return

    render_leaderboard_stats(df, mode)

    if sid is not None:
        my_rank, _ = cached_rank(sid, key_id)
        if my_rank:
            st.caption(f"Your current rank: **#{my_rank}**")

    render_leaderboard_rows(df, mode, sid=sid, key_suffix=key_suffix)

    if sid is not None and mode == "Overall":
        match = df[df["student_id"] == sid]
        if not match.empty:
            m = match.iloc[0]
            rank = int(m["rank"])
            st.markdown(
                f"""
                <div class='app-card' style='margin-top:6px; display:flex; gap:22px; flex-wrap:wrap; align-items:center;'>
                    <div>🏅 <b>Your Rank</b><br><span style='font-size:20px; font-weight:700;'>#{rank}</span></div>
                    <div>🎯 <b>Best Score</b><br><span style='font-size:20px; font-weight:700; color:#22c55e;'>{m['best_score']}</span></div>
                    <div>📈 <b>Average Score</b><br><span style='font-size:20px; font-weight:700; color:#3b82f6;'>{m['avg_percent']}%</span></div>
                    <div>✅ <b>Accuracy</b><br><span style='font-size:20px; font-weight:700;'>{m['accuracy']}%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("ℹ️ Leaderboard updates after each test submission.")


def page_leaderboard():
    sid = st.session_state["student_id"]
    st.markdown("### 🏆 Leaderboard")
    render_leaderboard(sid=sid, key_suffix="student")


# =========================================================================
# Student: Profile
# =========================================================================

def _profile_info_row(icon, label, value, icon_bg="var(--mv-primary-soft)"):
    """One row inside the Profile Information card - small icon chip +
    label + value, matching the reference dashboard design. icon_bg lets
    each row use a distinct chip color (phone/gender/birth date/role all
    read differently at a glance) instead of every row sharing one tone."""
    return (
        f"<div style='display:flex; align-items:center; gap:12px; padding:9px 0;'>"
        f"<div style='width:34px; height:34px; border-radius:9px; background:{icon_bg}; "
        f"display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0;'>{icon}</div>"
        f"<div style='min-width:0;'>"
        f"<div style='font-size:11px; color:var(--mv-muted);'>{label}</div>"
        f"<div style='font-size:14.5px; font-weight:600; color:var(--mv-ink); margin-top:1px; "
        f"overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{value}</div>"
        f"</div></div>"
    )


def _profile_status_pill_html(label, active_label, active, active_color="#26AB8C", bad_color="#F2434A"):
    color = bad_color if not active else active_color
    return (
        f"<div class='mv-profile-status-row'>"
        f"<span class='mv-profile-status-label'>{label}</span>"
        f"<span class='mv-profile-status-pill' style='background:{color}22; color:{color};'>{active_label}</span>"
        f"</div>"
    )


def _profile_stat_card_html(icon, icon_bg, number, label):
    """One stat's icon + big number + small label, centered - used inside
    the profile stats strip (see render_profile_stats_strip below). Pure
    markup only; the clickable "View X →" link (or static caption)
    underneath is rendered separately as a real st.button/st.markdown
    right after this, since a link has to be an actual Streamlit widget
    to navigate anywhere."""
    return (
        f"<div>"
        f"<div style='width:48px; height:48px; border-radius:50%; background:{icon_bg}; "
        f"display:flex; align-items:center; justify-content:center; margin:0 auto 10px; font-size:20px;'>{icon}</div>"
        f"<div style='font-family:var(--mono); font-size:25px; font-weight:700; color:var(--mv-ink); line-height:1.1;'>{number}</div>"
        f"<div style='font-size:12px; color:var(--mv-muted); margin-top:3px;'>{label}</div>"
        f"</div>"
    )


def _profile_hero_avatar_html(inner_avatar_html, glow_color):
    """Wraps a render_avatar()/manual-avatar HTML snippet in a soft glowing
    ring - used ONLY for the big avatar on the Profile page headers (both
    student and mentor), never for the small avatars used elsewhere in the
    app (leaderboard rows, student-management list, top-nav), so this
    stays a Profile-page-specific visual flourish rather than changing the
    look of avatars everywhere."""
    return (
        f"<span class='mv-avatar-glow-ring' style='box-shadow: "
        f"0 0 0 2px {glow_color}, 0 0 16px 2px {glow_color}99, 0 0 34px 6px {glow_color}40;'>"
        f"{inner_avatar_html}</span>"
    )


def render_profile_stats_strip(stats):
    """Renders the 4-stat strip card at the top of a Profile page (student
    or mentor) - one bordered card, 4 equal columns, each with an icon,
    a big number, a label, and either a clickable "View X →" link (real
    st.button, styled as plain text via the ".st-key-card_profile_stats"
    CSS) or a static caption underneath.

    `stats` is a list of exactly 4 dicts, each either:
      {"icon", "icon_bg", "number", "label", "link_text", "go_to_page": "..."}
      {"icon", "icon_bg", "number", "label", "caption": "..."}  (no link)

    go_to_page is normally a top-level app page name (passed straight to
    go_to()) - but for the mentor Profile page, where links need to land
    on a specific tab *inside* the mentor panel rather than a top-level
    page, prefix it "mentor:<mentor_page_key>" (e.g. "mentor:m_students")
    and this sets st.session_state["mentor_page"] accordingly before
    calling go_to("mentor"), same as every other mentor nav button in
    the app does.

    Kept as one small shared function (rather than copy-pasted per page)
    so the student and mentor Profile pages can never visually drift
    apart from each other.
    """
    with st.container(key="card_profile_stats"):
        cols = st.columns(4)
        for idx, (col, stat) in enumerate(zip(cols, stats)):
            with col:
                st.markdown(
                    _profile_stat_card_html(stat["icon"], stat["icon_bg"], stat["number"], stat["label"]),
                    unsafe_allow_html=True,
                )
                target_page = stat.get("go_to_page")
                if target_page:
                    # Keyed by position (idx) as well as target_page - two
                    # stats can legitimately point at the same page (e.g.
                    # "Tests Completed" and "Total Exams" both link to
                    # "tests"), and Streamlit requires every widget key on
                    # a page to be unique, so target_page alone isn't
                    # enough to guarantee that.
                    if st.button(stat["link_text"], key=f"profile_stat_{idx}_{target_page}", use_container_width=True):
                        if target_page.startswith("mentor:"):
                            st.session_state["mentor_page"] = target_page.split("mentor:", 1)[1]
                            st.session_state.pop("mentor_analysis_sid", None)
                            st.session_state.pop("mentor_analysis_view_key_id", None)
                            go_to("mentor")
                        else:
                            go_to(target_page)
                else:
                    st.markdown(
                        f"<div style='font-size:12.5px; color:var(--mv-primary); font-weight:600; "
                        f"margin-top:4px;'>{stat.get('caption', '')}</div>",
                        unsafe_allow_html=True,
                    )


def page_profile():
    sid = st.session_state["student_id"]
    student = sh.get_student_by_id(sid)
    name = student["name"]
    disabled = sh._to_bool(student.get("disabled", False))
    birth_date_val = (student.get("birth_date") or "").strip()
    gender_val = (student.get("gender") or "").strip()

    # ---- Stats used in the strip beside the header: tests completed,
    # average score, overall leaderboard rank - all derived from data
    # already being cached elsewhere in the app, so this adds no extra
    # Google Sheets calls. ----
    results = cached_results()
    my_results = results[results["student_id"] == sid] if not results.empty else results
    tests_completed = len(my_results)
    if not my_results.empty:
        avg_pct = round((my_results["marks"] / my_results["total"]).mean() * 100, 1)
    else:
        avg_pct = 0.0
    rank, _out_of = cached_rank(sid)

    header_col, stats_col = st.columns([1.3, 2.4], gap="medium")

    # ---- Header: avatar + name + role/verified badges ----
    with header_col:
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:16px; margin-bottom:14px; flex-wrap:wrap;'>
                {_profile_hero_avatar_html(render_avatar(sid, name, size=64, font_size=24), "#26AB8C")}
                <div>
                    <div style='font-family:var(--serif); font-weight:600; font-size:23px; color:var(--mv-ink); line-height:1.2;'>{name}</div>
                    <div style='display:flex; align-items:center; gap:8px; margin-top:4px;'>
                        <span style='font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--mv-muted); font-weight:700;'>Student</span>
                        <span style='font-size:11px; padding:2px 10px; border-radius:999px; background:var(--mv-primary-soft); color:var(--mv-primary); font-weight:700;'>✓ Verified</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Stats strip: Tests Completed / Average Score / Leaderboard
    # Rank / Total Exams, each with a "View X →" shortcut into the page
    # that actually shows that data. ----
    with stats_col:
        render_profile_stats_strip([
            {"icon": "📋", "icon_bg": "var(--mv-primary-soft)", "number": tests_completed,
             "label": "Tests Completed", "link_text": "View Results →", "go_to_page": "tests"},
            {"icon": "📈", "icon_bg": "var(--mv-blue-soft)", "number": f"{avg_pct}%",
             "label": "Average Score", "link_text": "View Analysis →", "go_to_page": "analysis"},
            {"icon": "🏆", "icon_bg": "var(--mv-accent-soft)", "number": (f"#{rank}" if rank else "—"),
             "label": "Leaderboard Rank", "link_text": "View Leaderboard →", "go_to_page": "leaderboard"},
            {"icon": "📚", "icon_bg": "var(--mv-purple-soft)", "number": tests_completed,
             "label": "Total Exams", "link_text": "View Results →", "go_to_page": "tests"},
        ])

    left_col, right_col = st.columns([1.7, 1], gap="medium")

    # ---- Left: Profile Information (view mode / edit mode) ----
    with left_col:
        with st.container(key="card_profile_info"):
            hcol1, hcol2 = st.columns([2.4, 1.3])
            with hcol1:
                st.markdown("##### 👤 Profile Information")
            with hcol2:
                edit_open = st.session_state.get("profile_edit_open", False)
                if st.button("✖ Cancel" if edit_open else "✏️ Update Profile",
                             key="profile_toggle_edit_btn", use_container_width=True):
                    st.session_state["profile_edit_open"] = not edit_open
                    st.rerun()

            if not st.session_state.get("profile_edit_open"):
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.markdown(_profile_info_row("📞", "Phone", sh.format_bd_phone(student["phone"]), "var(--mv-primary-soft)"), unsafe_allow_html=True)
                    st.markdown(_profile_info_row("🚻", "Gender", gender_val or "N/A", "var(--mv-blue-soft)"), unsafe_allow_html=True)
                with fcol2:
                    st.markdown(_profile_info_row("🎂", "Birth Date", birth_date_val or "N/A", "var(--mv-accent-soft)"), unsafe_allow_html=True)
                    st.markdown(_profile_info_row("🎓", "Role", "STUDENT", "var(--mv-purple-soft)"), unsafe_allow_html=True)
            else:
                # ---- Edit mode: Name / Birth Date / Gender only. Phone is
                # intentionally NOT offered here at all - it's the
                # student's login identity, so there's simply no field for
                # it in this form (nothing to explain to the student,
                # nothing for them to accidentally try to change). ----
                new_name = st.text_input("Full name", value=name, key="profile_edit_name_input")

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    want_birth_date = st.checkbox(
                        "Add / update birth date", value=bool(birth_date_val),
                        key="profile_edit_bd_toggle",
                    )
                    new_birth_date = None
                    if want_birth_date:
                        try:
                            default_bd_val = (
                                datetime.strptime(birth_date_val, "%Y-%m-%d").date()
                                if birth_date_val else date(2005, 1, 1)
                            )
                        except Exception:
                            default_bd_val = date(2005, 1, 1)
                        new_birth_date = st.date_input(
                            "Birth date", value=default_bd_val, key="profile_edit_bd_input",
                            min_value=date(1950, 1, 1), max_value=date.today(),
                        )
                with bcol2:
                    gender_idx = GENDER_OPTIONS.index(gender_val) if gender_val in GENDER_OPTIONS else 0
                    new_gender = st.selectbox(
                        "Gender", GENDER_OPTIONS, index=gender_idx, key="profile_edit_gender_input",
                    )

                st.caption("📞 Your phone number is your login ID, so it can't be changed here.")

                if st.button("💾 Save Changes", type="primary", use_container_width=True, key="profile_save_btn"):
                    cleaned_name = new_name.strip()
                    if not cleaned_name:
                        st.error("Name cannot be empty.")
                    else:
                        with st.spinner("Updating your profile..."):
                            try:
                                if cleaned_name != name:
                                    sh.update_student_name(sid, cleaned_name)
                                    st.session_state["student_name"] = cleaned_name
                                sh.update_student_extra_profile(
                                    sid,
                                    birth_date=(new_birth_date.strftime("%Y-%m-%d") if want_birth_date and new_birth_date else ""),
                                    gender=(new_gender if new_gender != "Not specified" else ""),
                                )
                                clear_all_caches()
                            except ValueError as e:
                                st.error(str(e))
                            else:
                                st.session_state["profile_edit_open"] = False
                                st.success("Profile updated!")
                                st.rerun()

        # ---- Change Password lives right under Profile Information (in
        # the same left column) instead of as its own full-width section
        # below both columns - that used to leave a large empty gap here
        # whenever the right column (Account Status + Log Out) ended up
        # taller than this one, since the two columns aren't forced to
        # match height. Putting it here fills that gap naturally. ----
        with st.container(key="card_profile_changepw"):
            pw_open = st.session_state.get("profile_changepw_open", False)
            if st.button(f"🔑  Change Password {'▾' if pw_open else '▸'}",
                         key="profile_changepw_toggle_btn", use_container_width=True):
                st.session_state["profile_changepw_open"] = not pw_open
                st.rerun()
            st.caption("Update your password regularly to keep your account secure.")

            if pw_open:
                # Plain widgets (no st.form) so the strength bar updates live while
                # typing, instead of only appearing after the button is clicked.
                cur_pw = st.text_input("Current password", type="password", key="prof_cur_pw")
                new_pw1 = st.text_input("New password", type="password", key="prof_new_pw1")
                if new_pw1:
                    score, label, _tips = sh.password_strength(new_pw1)
                    colors = ["#ef4444", "#ef4444", "#f59e0b", "#10b981", "#059669"]
                    st.markdown(
                        f"<div class='strength-bar'><div class='strength-fill' "
                        f"style='width:{(score+1)*20}%; background:{colors[score]};'></div></div>"
                        f"<small>Password strength: <b>{label}</b></small>",
                        unsafe_allow_html=True,
                    )
                new_pw2 = st.text_input("Confirm new password", type="password", key="prof_new_pw2")
                change_submitted = st.button("Update Password", type="primary")

                if change_submitted:
                    try:
                        sh.authenticate_student(student["phone"], cur_pw)
                    except ValueError:
                        st.error("Current password is incorrect.")
                    else:
                        _, _, tips = sh.password_strength(new_pw1)
                        if tips:
                            st.error("New password is too weak: " + ", ".join(tips))
                        elif new_pw1 != new_pw2:
                            st.error("New passwords don't match.")
                        else:
                            with st.spinner("Updating..."):
                                sh.change_student_password(sid, new_pw1)
                                clear_all_caches()
                            st.success("Password updated. Please log in again.")
                            for k in ("student_id", "student_name", "session_version", "role"):
                                st.session_state.pop(k, None)
                            st.rerun()

    # ---- Right: Account Status + Log Out ----
    with right_col:
        with st.container(key="card_profile_status"):
            st.markdown("##### 🛡️ Account Status")
            st.markdown(
                _profile_status_pill_html("Account Status", "Disabled" if disabled else "Active", not disabled)
                + _profile_status_pill_html("Block Status", "Blocked" if disabled else "Not Blocked", not disabled),
                unsafe_allow_html=True,
            )

        with st.container(key="card_profile_logout"):
            st.markdown(
                "<div style='display:flex; align-items:center; gap:10px;'>"
                "<div class='mv-logout-icon'>🚪</div>"
                "<span style='font-weight:700; font-size:15px; color:var(--mv-ink);'>Log Out</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.caption("Sign out from your account securely.")
            if st.button("Log Out", use_container_width=True, key="profile_logout_btn_new"):
                for k in ("student_id", "student_name", "session_version", "role"):
                    st.session_state.pop(k, None)
                go_to("home")

    # ---- Mentor Login lives here so the student nav stays a clean,
    # consistent set of items everywhere. Icon chip + title/description
    # on the left, a compact button on the right - matches the reference
    # design instead of the earlier plain text-above-full-width-button. ----
    with st.container(key="card_profile_mentor"):
        text_col, btn_col = st.columns([3.2, 1.3])
        with text_col:
            st.markdown(
                "<div style='display:flex; align-items:center; gap:14px;'>"
                "<div class='mv-mentor-cta-icon'>🎓</div>"
                "<div>"
                "<div style='font-weight:700; font-size:15px; color:var(--mv-ink);'>Are you a mentor?</div>"
                "<div style='font-size:12.5px; color:var(--mv-muted); margin-top:2px;'>"
                "Join our mentor community and help others achieve their goals.</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
        with btn_col:
            if st.button("👨‍🏫 Mentor Login", use_container_width=True, key="profile_mentor_login_btn"):
                go_to("mentor")


# =========================================================================
# Mentor: Answer Key tab (native bubble-grid input)
# =========================================================================

def _inject_bubble_grid_css():
    st.markdown(
        """
        <style>
        /* OMR answer bubbles.
           IMPORTANT: scope the row layout to the per-question wrapper,
           not to the whole answer_bubble_grid. The old selector matched the
           OUTER 2-column layout as well as every question row, so its
           first/last-child flex rules moved the second question column and
           made the A/B/C/D bubbles appear far to the right. */
        [class*="st-key-answer_row_"] div[data-testid="stRadio"] { margin-bottom: -14px; }
        [class*="st-key-answer_row_"] div[data-testid="stRadio"] > label { display: none; }
        [class*="st-key-answer_row_"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: flex-start !important;
            gap: 6px !important;
            width: max-content !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }
        [class*="st-key-answer_row_"] [data-testid="stRadio"] {
            width: max-content !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }
        [class*="st-key-answer_row_"] div[role="radiogroup"] label {
            flex: 0 0 auto !important;
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 999px;
            padding: 2px 10px 2px 6px;
            margin-right: 0 !important;
            white-space: nowrap !important;
        }
        [class*="st-key-answer_row_"] div[data-testid="stHorizontalBlock"] {
            display: grid !important;
            /* 46px is intentional: Q20/Q100 must stay as "20"/"100",
               never wrap into 2 over 0 or 10 over 0. */
            grid-template-columns: 46px minmax(0, 1fr) !important;
            align-items: center !important;
            column-gap: 4px !important;
            width: 100% !important;
            min-width: 0 !important;
        }
        [class*="st-key-answer_row_"] div[data-testid="column"] {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
        }
        .q-num-badge {
            display: block;
            width: 46px;
            min-width: 46px;
            max-width: 46px;
            font-weight: 600;
            color: var(--mv-ink);
            opacity: 0.75;
            padding-top: 2px;
            text-align: left;
            white-space: nowrap !important;
            overflow: visible !important;
        }
        /* The OUTER 1-20 | 21-40 (or 1-25 | 26-50) grid is deliberately
           left alone on desktop. On phones Streamlit may stack those two
           blocks, but each individual question remains number + options on
           one horizontal line. */
        [class*="st-key-answer_row_"] div[data-testid="stRadio"] > div {
            min-width: 0 !important;
        }
        @media (max-width: 640px) {
            [class*="st-key-answer_row_"] div[role="radiogroup"] { gap: 4px !important; }
            [class*="st-key-answer_row_"] div[role="radiogroup"] label {
                padding: 2px 6px 2px 4px !important;
                font-size: 12px !important;
            }
            [class*="st-key-answer_row_"] div[data-testid="stHorizontalBlock"] {
                grid-template-columns: 32px minmax(0, 1fr) !important;
            }
            .q-num-badge {
                width: 32px;
                min-width: 32px;
                max-width: 32px;
                font-size: 13px;
            }
        }
        @media (max-width: 360px) {
            [class*="st-key-answer_row_"] div[role="radiogroup"] {
                width: 100% !important;
                max-width: 100% !important;
                justify-content: space-between !important;
            }
            [class*="st-key-answer_row_"] [data-testid="stRadio"] {
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _answers_store():
    """Single dict in session_state holding every question's chosen answer,
    independent from any individual widget's mount/unmount lifecycle."""
    if "mentor_answers" not in st.session_state:
        st.session_state["mentor_answers"] = {}
    if "mentor_answer_widget_version" not in st.session_state:
        st.session_state["mentor_answer_widget_version"] = 0
    return st.session_state["mentor_answers"]


def _on_bubble_change(q, widget_key):
    _answers_store()[q] = st.session_state.get(widget_key)


def _count_answered(total_q):
    store = _answers_store()
    return sum(1 for q in range(1, total_q + 1) if store.get(q) is not None)


def _build_answer_string(total_q):
    store = _answers_store()
    return "".join(store.get(q) or "?" for q in range(1, total_q + 1))


def _render_bubble_block(q_start, q_end):
    store = _answers_store()
    options = ["A", "B", "C", "D"]
    widget_version = st.session_state.get("mentor_answer_widget_version", 0)
    for q in range(q_start, q_end + 1):
        # Give every question its own stable DOM scope. This is critical:
        # answer_bubble_grid also contains the OUTER two-column layout, so
        # applying row CSS directly to answer_bubble_grid accidentally styled
        # that outer layout as if it were a question row.
        #
        # The version suffix is equally important: Quick Fill replaces the
        # entire answer set. A fresh widget key guarantees Streamlit cannot
        # reuse an already-mounted radio's old browser/widget state.
        with st.container(key=f"answer_row_{q}"):
            num_col, radio_col = st.columns([0.55, 3], gap="small")
            widget_key = f"ans_q_{q}_v{widget_version}"
            with num_col:
                st.markdown(f"<div class='q-num-badge'>{q}</div>", unsafe_allow_html=True)
            with radio_col:
                current = store.get(q)
                idx = options.index(current) if current in options else None
                st.radio(
                    f"Q{q}", options=options, index=idx, horizontal=True,
                    key=widget_key, label_visibility="collapsed",
                    on_change=_on_bubble_change, args=(q, widget_key),
                )


def _go_answer_page(page_num):
    """Callback for the Next/Back buttons below the bubble grid. Using
    on_click here (instead of the old 'if st.button(...): set state;
    st.rerun()' pattern) is what fixes the multi-second lag when switching
    pages: an on_click callback runs BEFORE the script reruns, so the page
    number is already updated by the time the script body executes -
    no extra, second full script rerun is needed on top of the one
    Streamlit already triggers for the button click itself."""
    st.session_state["mentor_answer_page"] = page_num


def _time_input_12h(label, key_prefix, default_hour_24=9, default_minute=0):
    """Friendly Start/End time card. Keeps the same 12-hour AM/PM logic,
    but presents the controls as a clear three-part time selector so the
    mentor can set the exam window quickly without guessing which field is
    which. The caller can place Start and End side by side on desktop;
    Streamlit can stack the two cards naturally on a narrow phone."""
    default_period = "PM" if default_hour_24 >= 12 else "AM"
    default_hour_12 = default_hour_24 % 12
    if default_hour_12 == 0:
        default_hour_12 = 12

    with st.container(key=f"{key_prefix}_time_card"):
        st.markdown(
            f"<div class='mv-time-card-title'><span>{'🟢' if label.lower().startswith('start') else '🔴'}</span> {label}</div>"
            "<div class='mv-time-card-sub'>Choose the exact time</div>",
            unsafe_allow_html=True,
        )
        h_col, m_col, p_col = st.columns([1, 1, 1], gap="small")
        with h_col:
            hour = st.selectbox(
                "Hour", list(range(1, 13)), index=default_hour_12 - 1,
                key=f"{key_prefix}_hour", label_visibility="collapsed",
            )
        with m_col:
            minute = st.selectbox(
                "Minute", [f"{m:02d}" for m in range(60)], index=default_minute,
                key=f"{key_prefix}_min", label_visibility="collapsed",
            )
        with p_col:
            period = st.selectbox(
                "AM/PM", ["AM", "PM"],
                index=0 if default_period == "AM" else 1,
                key=f"{key_prefix}_period", label_visibility="collapsed",
            )

    hour_24 = hour % 12
    if period == "PM":
        hour_24 += 12
    return dtime(hour_24, int(minute))


# Preset exam-duration options (minutes) shown as quick-pick pills, plus a
# "Custom" option that reveals a free-entry minutes field - covers the
# common cases (30/60/90/120 min) with one tap, while still allowing any
# other length.
DURATION_PRESETS = [15, 30, 45, 60, 90, 120, 180]


def _duration_picker(key_prefix, default_minutes=60):
    """Renders a duration picker (preset pills + custom minutes field) and
    returns the chosen duration as an int number of minutes."""
    preset_labels = [f"{m} min" for m in DURATION_PRESETS] + ["Custom"]
    default_label = f"{default_minutes} min" if default_minutes in DURATION_PRESETS else "Custom"
    default_idx = preset_labels.index(default_label)
    choice = st.radio(
        "Duration", preset_labels, horizontal=True, index=default_idx,
        label_visibility="collapsed", key=f"{key_prefix}_choice",
    )
    if choice == "Custom":
        return int(st.number_input(
            "Custom duration (minutes)", min_value=1, max_value=1440,
            value=default_minutes, step=5, key=f"{key_prefix}_custom",
        ))
    return DURATION_PRESETS[preset_labels.index(choice)]


def _compute_exam_end_datetime(exam_date, start_t, end_t):
    """Combines the exam date with the mentor-picked END time, then rolls
    that date forward by one day if the result would land at or before the
    start time - e.g. a 22nd 8 PM start with an 8 AM end time is
    understood as ending on the 23rd, automatically, without the mentor
    ever having to pick a separate end date by hand. Returns (start_dt,
    end_dt)."""
    start_dt = datetime.combine(exam_date, start_t)
    end_dt = datetime.combine(exam_date, end_t)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def _format_exam_window(start_dt, end_dt):
    """Human-friendly 'Aug 22, 2026 · 8:00 PM  →  Aug 23, 2026 · 8:00 AM'
    style labels for a start/end datetime pair. Always shows the full date
    on both sides (not just when they differ) so an overnight exam window
    is never ambiguous at a glance."""
    fmt = "%b %d, %Y · %I:%M %p"

    def _clean(dt):
        # strip a leading zero from the 12-hour hour only (keep the date's
        # own leading zeros, e.g. "Aug 05" should stay "Aug 05")
        s = dt.strftime(fmt)
        hour_part, _, rest = s.partition(":")
        date_part, _, hour_only = hour_part.rpartition(" ")
        return f"{date_part} {hour_only.lstrip('0') or '0'}:{rest}"

    return _clean(start_dt), _clean(end_dt)


def _format_duration(total_minutes):
    total_minutes = int(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def _format_hms(total_seconds):
    """Format a countdown as HH:MM:SS for student-facing exam timers."""
    total_seconds = max(0, int(total_seconds or 0))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _render_exam_window_summary(start_dt, end_dt, duration_min):
    """Small card shown on the exam-creation form (and reused on the
    student Home page's Active Test card) summarizing exactly when the
    exam opens, when it closes (the "exam window" - independently set by
    the mentor's Start/End time, NOT derived from Duration), and how long
    each student is meant to spend on it (Duration - a separate,
    informational field; a mentor can, for example, keep an exam window
    open all night for late submissions while still telling students it's
    a 60-minute test). Includes a clear flag when the window rolls over
    into the next calendar day.

    IMPORTANT: the HTML below is built as ONE joined string with no blank
    or indented "just whitespace" lines inside it (see the identical note
    on render_hero() near the top of this file) - a blank line inside a
    raw HTML block passed to st.markdown(unsafe_allow_html=True) makes
    Streamlit's own markdown parser treat everything after that line as
    literal text instead of HTML, which is exactly what was leaking raw
    "<div>...</div>" tags onto the page here before this fix.
    """
    start_label, end_label = _format_exam_window(start_dt, end_dt)
    crosses_midnight = start_dt.date() != end_dt.date()
    overnight_note = (
        "<div style='font-size:12px;color:var(--mv-accent);margin-top:3px;'>"
        "⚠️ Ends the next day - the date is auto-adjusted for you.</div>"
        if crosses_midnight else ""
    )
    parts = [
        "<div class='app-card' style='display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-top:6px;'>",
        "<div style='flex:1;min-width:240px;'>",
        "<div style='font-size:11px;color:var(--mv-muted);text-transform:uppercase;letter-spacing:.06em;'>Exam Window (online from &rarr; to)</div>",
        f"<div style='font-size:15px;font-weight:700;margin-top:2px;'>🟢 {start_label} &nbsp;→&nbsp; 🔴 {end_label}</div>",
        overnight_note,
        "</div>",
        "<div>",
        "<div style='font-size:11px;color:var(--mv-muted);text-transform:uppercase;letter-spacing:.06em;'>Duration Given to Students</div>",
        f"<div style='font-size:15px;font-weight:700;margin-top:2px;'>⏱️ {_format_duration(duration_min)}</div>",
        "</div>",
        "</div>",
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)



def _answer_rule_defaults(key):
    total = int(key.get("total_questions", 0) or 0)
    rules = key.get("answer_rules") or {}
    if not rules:
        ans = str(key.get("answer_string", ""))
        rules = {str(q): {"type": "normal", "accepted": ([ans[q-1]] if q <= len(ans) and ans[q-1] in "ABCD" else [])} for q in range(1, total+1)}
    else:
        rules = {str(q): dict(rules.get(str(q), {})) for q in range(1, total+1)}
        ans = str(key.get("answer_string", ""))
        for q in range(1, total+1):
            r = rules[str(q)]
            if not r:
                a = ans[q-1:q].upper()
                rules[str(q)] = {"type":"normal", "accepted":[a] if a in "ABCD" else []}
    return rules


def _render_answer_key_readonly(key):
    total = int(key.get("total_questions", 0) or 0)
    rules = _answer_rule_defaults(key)
    notes = key.get("question_notes") or {}
    st.markdown(f"#### 📝 Answer Key · {total} Questions")
    cols = st.columns(2 if total > 40 else 1, gap="small")
    half = (total + 1) // 2 if total > 40 else total
    ranges = [(1, half), (half + 1, total)] if total > 40 else [(1, total)]
    for col, (start, end) in zip(cols, ranges):
        with col:
            for q in range(start, end + 1):
                r = rules.get(str(q), {}) or {}
                typ = str(r.get("type", "normal")).lower()
                accepted = ", ".join(r.get("accepted", []))
                if typ == "bonus":
                    label = "⭐ Bonus / Invalid · +1 mark"
                elif typ == "multiple":
                    label = f"Multiple Correct · {accepted or '—'}"
                else:
                    label = f"Normal · {accepted or '—'}"
                note = str(notes.get(str(q), "") or "")
                st.markdown(f"**Q{q}** &nbsp; `{label}`", unsafe_allow_html=True)
                if note:
                    st.caption(f"📝 {note}")


def _render_answer_key_editor(key):
    """Edit an exam-wide answer key with direct OMR-style bubble clicks."""
    total = int(key.get("total_questions", 0) or 0)
    rules = _answer_rule_defaults(key)
    notes = {str(k): str(v) for k, v in (key.get("question_notes") or {}).items()}
    key_id = str(key.get("key_id"))

    st.markdown("### ✏️ Edit Answer Key")
    st.caption(f"{key.get('exam_name') or key_id} · {total} questions")
    st.warning("⚠️ Saving changes will recalculate every submitted student's result for this exam.")

    show_pdf = st.session_state.get("edit_exam_show_pdf", False)
    if st.button("📄 " + ("Hide Question Paper" if show_pdf else "View Question Paper"), key="edit_exam_pdf_btn", use_container_width=True):
        st.session_state["edit_exam_show_pdf"] = not show_pdf
        st.rerun()
    if st.session_state.get("edit_exam_show_pdf"):
        pdf_id = str(key.get("question_pdf_file_id", "") or "")
        if pdf_id:
            try:
                pdf_bytes = sh.get_question_pdf_bytes(pdf_id)
                if pdf_bytes:
                    _render_question_pdf(pdf_bytes, None, key_id)
                else:
                    st.info("Question PDF is unavailable.")
            except Exception:
                st.info("Question PDF is unavailable.")
        else:
            st.info("No question PDF is attached to this exam.")

    st.markdown("#### 📝 Answer Key")
    st.caption("Tap the correct OMR bubbles directly. 1 bubble = Normal · 2+ bubbles = Multiple Correct · ⭐ = Bonus")

    page_size = 25 if total > 40 else total
    pages = list(range(1, total + 1, page_size))
    page = st.session_state.get("edit_exam_key_page", 0)
    page = min(max(page, 0), max(len(pages) - 1, 0))
    start = pages[page]
    end = min(start + page_size - 1, total)
    st.progress(end / total if total else 0)
    st.caption(f"Questions {start}–{end} of {total}")

    for q in range(start, end + 1):
        k = str(q)
        r = rules.get(k, {}) or {}
        existing_type = str(r.get("type", "normal")).lower()
        existing = [x for x in (r.get("accepted") or []) if x in "ABCD"]
        state_key = f"edit_key_bubbles_{key_id}_{q}"
        bonus_key = f"edit_key_bonus_{key_id}_{q}"

        # Initialise once from the saved answer-key rule.
        if state_key not in st.session_state:
            st.session_state[state_key] = [] if existing_type == "bonus" else list(existing)
        if bonus_key not in st.session_state:
            st.session_state[bonus_key] = existing_type == "bonus"

        with st.container(key=f"mentor_edit_key_q_{q}"):
            top = st.columns([0.8, 4.2, 1.3], gap="small")
            with top[0]:
                st.markdown(f"**Q{q:02d}**")
            with top[1]:
                selected = list(st.session_state.get(state_key, []))
                bubble_cols = st.columns(4, gap="small")
                for col, letter in zip(bubble_cols, "ABCD"):
                    with col:
                        active = letter in selected and not st.session_state.get(bonus_key, False)
                        label = f"🔘 **{letter}**" if active else f"⭕ **{letter}**"
                        if st.button(label, key=f"edit_key_bubble_{key_id}_{q}_{letter}", use_container_width=True):
                            current = list(st.session_state.get(state_key, []))
                            if st.session_state.get(bonus_key, False):
                                st.session_state[bonus_key] = False
                                current = [letter]
                            elif letter in current:
                                current.remove(letter)
                            else:
                                current.append(letter)
                            st.session_state[state_key] = current
                            st.rerun()
            with top[2]:
                bonus_label = "⭐ Bonus ✓" if st.session_state.get(bonus_key, False) else "⭐ Bonus"
                if st.button(bonus_label, key=f"edit_key_bonus_btn_{key_id}_{q}", use_container_width=True):
                    st.session_state[bonus_key] = not st.session_state.get(bonus_key, False)
                    if st.session_state[bonus_key]:
                        st.session_state[state_key] = []
                    st.rerun()

            selected_now = list(st.session_state.get(state_key, []))
            if st.session_state.get(bonus_key, False):
                st.caption("⭐ Bonus / Invalid — everyone receives the question mark.")
            elif len(selected_now) > 1:
                st.caption(f"Multiple Correct · Accepted: {', '.join(selected_now)}")
            elif len(selected_now) == 1:
                st.caption(f"Normal · Correct answer: {selected_now[0]}")
            else:
                st.caption("⚪ No answer selected")

            st.text_input(
                "Note / explanation for students",
                value=notes.get(k, ""),
                key=f"edit_key_note_{key_id}_{q}",
                placeholder="Optional explanation…",
                label_visibility="collapsed",
            )
            st.divider()

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("← Previous", key="edit_exam_prev", use_container_width=True, disabled=page == 0):
            st.session_state["edit_exam_key_page"] = page - 1
            st.rerun()
    with nav2:
        if st.button("Next →", key="edit_exam_next", use_container_width=True, disabled=page >= len(pages)-1):
            st.session_state["edit_exam_key_page"] = page + 1
            st.rerun()

    if st.button("💾 Save Changes & Recalculate All Results", type="primary", use_container_width=True, key="edit_exam_save_all"):
        final_rules = {}
        final_notes = {}
        for q in range(1, total + 1):
            k = str(q)
            selected = [x for x in st.session_state.get(f"edit_key_bubbles_{key_id}_{q}", []) if x in "ABCD"]
            is_bonus = bool(st.session_state.get(f"edit_key_bonus_{key_id}_{q}", False))
            if is_bonus:
                rule_type = "bonus"
                accepted = []
            elif len(selected) > 1:
                rule_type = "multiple"
                accepted = selected
            elif len(selected) == 1:
                rule_type = "normal"
                accepted = selected
            else:
                st.error(f"Q{q}: please select an answer or mark it as Bonus.")
                return
            final_rules[k] = {"type": rule_type, "accepted": accepted}
            note = st.session_state.get(f"edit_key_note_{key_id}_{q}", "").strip()
            if note:
                final_notes[k] = note
        try:
            with st.spinner("Saving answer-key changes and recalculating all results…"):
                sh.update_answer_key_rules(key_id, final_rules, final_notes)
                changed = sh.recalculate_results_for_exam(key_id, final_rules)
                clear_all_caches()
            st.success(f"✅ Answer key updated. {changed} student result(s) recalculated.")
            st.session_state["edit_exam_key_page"] = 0
            st.session_state["edit_exam_show_pdf"] = False
            st.rerun()
        except Exception as e:
            st.error(f"Could not update this exam: {e}")

def render_answer_key_tab():
    """Mentor exam hub: clean create flow + one-click answer-key editing."""
    st.markdown("""
    <style>
    .mentor-exam-hero {
        border:1px solid var(--mv-border); border-radius:20px;
        padding:18px 20px; margin:2px 0 14px;
        background:linear-gradient(135deg, rgba(38,171,140,.12), rgba(249,77,16,.035));
        box-shadow:0 8px 24px rgba(0,0,0,.035);
    }
    .mentor-exam-hero .eyebrow {font-size:11px; text-transform:uppercase; letter-spacing:.12em;
        color:var(--mv-primary); font-weight:800; margin-bottom:5px;}
    .mentor-exam-hero h2 {margin:0; font-size:28px; line-height:1.15;}
    .mentor-exam-hero p {margin:7px 0 0; color:var(--mv-muted); font-size:13px;}
    .mentor-create-card {
        border:1px solid var(--mv-border); border-radius:15px; padding:13px 15px;
        background:var(--mv-card-bg); margin-bottom:8px;
        box-shadow:0 5px 18px rgba(0,0,0,.025);
    }
    .mentor-create-card .title {font-weight:800; font-size:17px; margin-bottom:3px;}
    .mentor-create-card .sub {font-size:12px; color:var(--mv-muted);}
    .mentor-exam-row {
        padding:10px 4px 9px 2px; min-width:0;
    }
    .mentor-exam-row .name {font-weight:800; font-size:15px; color:var(--mv-ink);
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .mentor-exam-row .meta {font-size:11px; color:var(--mv-muted); margin-top:3px;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .mentor-exam-divider {border-bottom:1px solid var(--mv-border); margin:0 0 2px;}
    .mentor-exam-search-note {font-size:11px; color:var(--mv-muted); margin:2px 0 8px;}
    @media(max-width:640px){
        div[data-testid="stHorizontalBlock"] .mentor-exam-row {padding-right:0;}
    }
    .mentor-edit-head {
        border:1px solid var(--mv-border); border-radius:16px; padding:16px 18px;
        background:var(--mv-card-bg); margin-bottom:15px;
    }
    .mentor-edit-head .name {font-size:22px; font-weight:800; line-height:1.2;}
    .mentor-edit-head .meta {font-size:12px; color:var(--mv-muted); margin-top:5px;}
    .mentor-exam-card + div .stButton > button { border-radius:10px !important; min-height:38px !important; font-weight:700 !important; }
    @media(max-width:640px){
        .mentor-exam-hero{padding:15px 14px}.mentor-exam-hero h2{font-size:21px}
        .mentor-exam-hero p{font-size:11px}
        .mentor-create-card{padding:12px 13px}.mentor-edit-head{padding:13px}
        .mentor-edit-head .name{font-size:18px}
        .mentor-exam-row .name{font-size:14px}
        .mentor-exam-row .meta{font-size:10px}
    }
    </style>
    """, unsafe_allow_html=True)

    # If an exam is currently being edited, keep the page focused on that exam.
    edit_key_id = st.session_state.get("mentor_edit_key_id")
    if edit_key_id:
        key = sh.get_answer_key_by_id(edit_key_id)
        if not key:
            st.session_state.pop("mentor_edit_key_id", None)
            st.rerun()

        back_col, title_col = st.columns([0.85, 3.2], gap="small")
        with back_col:
            if st.button("← Exams", use_container_width=True, key="mentor_back_exam_list"):
                st.session_state.pop("mentor_edit_key_id", None)
                st.session_state["edit_exam_key_page"] = 0
                st.session_state["edit_exam_show_pdf"] = False
                st.rerun()
        with title_col:
            st.markdown(
                f"<div class='mentor-edit-head'><div class='name'>✏️ {key.get('exam_name') or edit_key_id}</div>"
                f"<div class='meta'>{key.get('date','')} &nbsp;·&nbsp; {key.get('total_questions',0)} MCQs &nbsp;·&nbsp; {key.get('duration_minutes',0)} min</div></div>",
                unsafe_allow_html=True,
            )
        _render_answer_key_editor(key)
        return

    # Main exam hub — no create/edit toggle.
    st.markdown(
        "<div class='mentor-exam-hero'><div class='eyebrow'>Mentor Workspace</div>"
        "<h2>🗓️ Exams</h2>"
        "<p>Create a new exam or open an existing exam and edit its answer key.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='mentor-create-card'><div class='title'>＋ Create a New Exam</div>"
        "<div class='sub'>Set MCQ count, exam window, duration, question paper and answer key.</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("＋ Create Exam", type="primary", use_container_width=True, key="mentor_create_exam_open"):
        st.session_state["mentor_show_create_exam"] = True
        st.rerun()

    if st.session_state.get("mentor_show_create_exam"):
        st.divider()
        if st.button("← Back to Exams", key="mentor_create_back"):
            st.session_state["mentor_show_create_exam"] = False
            st.rerun()
        _render_create_exam_form()
        return

    st.markdown("### Existing Exams")
    keys_df = cached_answer_keys()
    if keys_df.empty:
        st.info("No exams have been created yet.")
        return

    # Scalable exam browser: searchable typeahead + 10 exams/page.
    # Streamlit's selectbox is natively searchable: type only part of an exam
    # name (e.g. "Phy") and matching exam-name suggestions appear instantly.
    search_col, sort_col = st.columns([4.2, 1.35], gap="small")

    exam_names = (
        keys_df["exam_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    exam_names = sorted({name for name in exam_names if name}, key=str.casefold)
    search_options = ["All Exams"] + exam_names

    with search_col:
        selected_exam = st.selectbox(
            "Search exams",
            search_options,
            key="mentor_exam_search_pick",
            label_visibility="collapsed",
            help="Type part of an exam name to get matching suggestions.",
        )
    with sort_col:
        sort_newest = st.selectbox(
            "Sort", ["Newest", "Oldest"],
            key="mentor_exam_sort",
            label_visibility="collapsed",
        )

    # Reset pagination whenever the filter/sort changes.
    filter_signature = (selected_exam, sort_newest)
    if st.session_state.get("mentor_exam_filter_signature") != filter_signature:
        st.session_state["mentor_exam_list_page"] = 0
        st.session_state["mentor_exam_filter_signature"] = filter_signature

    work_df = keys_df.copy()
    if selected_exam != "All Exams":
        work_df = work_df[
            work_df["exam_name"].fillna("").astype(str).str.strip() == selected_exam
        ]

    # Newest first by default; preserve existing row order as fallback.
    if "date" in work_df.columns:
        work_df["_sort_date"] = pd.to_datetime(work_df["date"], errors="coerce")
        work_df = work_df.sort_values("_sort_date", ascending=(sort_newest == "Oldest"), na_position="last")
    else:
        work_df = work_df.iloc[::-1] if sort_newest == "Newest" else work_df
    work_df = work_df.reset_index(drop=True)

    page_size = 10
    total_pages = max(1, (len(work_df) + page_size - 1) // page_size)
    current_page = int(st.session_state.get("mentor_exam_list_page", 0))
    current_page = min(max(current_page, 0), total_pages - 1)
    st.session_state["mentor_exam_list_page"] = current_page

    if not len(work_df):
        st.info("No exams match your search.")
        return

    start = current_page * page_size
    page_df = work_df.iloc[start:start + page_size]
    st.caption(f"Showing {start + 1}–{start + len(page_df)} of {len(work_df)} exam(s)")

    for idx, (_, row) in enumerate(page_df.iterrows()):
        key_id = str(row.get("key_id", ""))
        name = str(row.get("exam_name") or key_id)
        exam_date = str(row.get("date", ""))
        total = int(row.get("total_questions") or 0)
        duration = int(row.get("duration_minutes") or 0)
        with st.container(key=f"mentor_exam_card_{idx}"):
            # Compact single-row exam list: details on the left, one Edit action on the right.
            c1, c2 = st.columns([5.2, 1.15], gap="small", vertical_alignment="center")
            with c1:
                st.markdown(
                    f"<div class='mentor-exam-row'><div class='name'>{name}</div>"
                    f"<div class='meta'>{exam_date} &nbsp;·&nbsp; {total} MCQs &nbsp;·&nbsp; {duration} min</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✏️ Edit", type="primary", use_container_width=True, key=f"mentor_edit_exam_{key_id}_{idx}"):
                    st.session_state["mentor_edit_key_id"] = key_id
                    st.session_state["edit_exam_key_page"] = 0
                    st.session_state["edit_exam_show_pdf"] = False
                    st.rerun()
            st.markdown("<div class='mentor-exam-divider'></div>", unsafe_allow_html=True)

    if total_pages > 1:
        p1, p2, p3, p4, p5 = st.columns([1.0, 1.0, 2.2, 1.0, 1.0], gap="small")
        with p1:
            if st.button("‹", disabled=current_page == 0, use_container_width=True, key="mentor_exam_prev_page"):
                st.session_state["mentor_exam_list_page"] = current_page - 1
                st.rerun()
        with p2:
            if st.button("1", disabled=total_pages == 1, use_container_width=True, key="mentor_exam_first_page"):
                st.session_state["mentor_exam_list_page"] = 0
                st.rerun()
        with p3:
            st.markdown(f"<div style='text-align:center;padding:8px 0;font-size:12px;color:var(--mv-muted);'>Page <b>{current_page + 1}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
        with p4:
            if st.button(str(total_pages), disabled=total_pages == 1, use_container_width=True, key="mentor_exam_last_page"):
                st.session_state["mentor_exam_list_page"] = total_pages - 1
                st.rerun()
        with p5:
            if st.button("›", disabled=current_page >= total_pages - 1, use_container_width=True, key="mentor_exam_next_page"):
                st.session_state["mentor_exam_list_page"] = current_page + 1
                st.rerun()


def _render_create_exam_form():
    st.subheader("🗓️ Create Exam & Set Answer Key")

    st.markdown("#### ① How many MCQs? (Exam Style)")
    exam_style = st.radio(
        "Exam Style", [
            "📄 40 Questions (Q1-40)",
            "📄 50 Questions (Q1-50)",
            "📄 100 Questions (Q1-100)",
        ],
        horizontal=True, label_visibility="collapsed", key="mentor_exam_style_choice",
    )
    if "100" in exam_style:
        total_q = 100
    elif "50" in exam_style:
        total_q = 50
    else:
        total_q = 40

    if total_q == 100:
        instruction = (
            "Questions 1-50 are shown first (in two columns of 25). Scroll down and "
            "click <b>Next: 51-100</b> to enter the second half. Your Q1-50 answers "
            "stay saved while you fill in 51-100."
        )
    else:
        instruction = (
            "This exam uses the same physical <b>50 / 40 OMR</b> sheet. "
            f"Enter the answer key for Q1-{total_q}; no separate OMR design is needed for 40 and 50 questions."
        )
    st.markdown(
        f"<div class='mv-mobile-hide-instruction'>ℹ️ {instruction}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("mentor_answer_total_q") != total_q:
        st.session_state["mentor_answers"] = {}
        st.session_state["mentor_answer_widget_version"] = (
            st.session_state.get("mentor_answer_widget_version", 0) + 1
        )
        st.session_state["mentor_answer_total_q"] = total_q
        st.session_state["mentor_answer_page"] = 1

    st.divider()

    st.markdown("#### ② Exam Details")
    exam_name = st.text_input("Exam name", placeholder="e.g. Physics Model Test - 3")
    d = st.date_input("Exam date", value=date.today())

    st.markdown(
        "<div class='mv-window-heading'><b>🕒 Exam Access Window</b>"
        "<span>When students can open and submit this exam</span></div>",
        unsafe_allow_html=True,
    )
    start_col, end_col = st.columns(2, gap="medium")
    with start_col:
        start_t = _time_input_12h("Start time", "mentor_start_t", default_hour_24=9, default_minute=0)
    with end_col:
        end_t = _time_input_12h("End time", "mentor_end_t", default_hour_24=9, default_minute=30)
    st.markdown(
        "<div class='mv-mobile-hide-instruction'>ℹ️ This is the window the exam stays "
        "<b>online / open for submission</b> (e.g. 8:00 PM to 8:00 AM). If the end "
        "time is earlier than the start time, it's automatically understood as the "
        "<b>next day</b> - you don't need to pick a separate end date.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Exam duration (how much time each student gets)**")
    duration_min = _duration_picker("mentor_duration", default_minutes=30)
    st.markdown(
        "<div class='mv-mobile-hide-instruction'>ℹ️ This is separate from the window "
        "above - it's shown to students as how long the test itself is meant to take, "
        "even if the submission window stays open longer (e.g. for late starters).</div>",
        unsafe_allow_html=True,
    )

    # Exam WINDOW (start -> end) is entirely independent of Duration - the
    # end date is auto-computed from the mentor's own End Time pick (with
    # next-day rollover when needed), never derived from Duration. This is
    # what lets a mentor keep submissions open overnight (or for days) while
    # still telling students the test itself is only e.g. 30 minutes long.
    start_dt_preview, end_dt_preview = _compute_exam_end_datetime(d, start_t, end_t)

    _render_exam_window_summary(start_dt_preview, end_dt_preview, duration_min)

    st.divider()

    st.markdown("#### 📄 Question Paper")
    question_pdf = st.file_uploader(
        "Upload the question PDF from your device",
        type=["pdf"],
        key="mentor_question_pdf",
        help="Students will open this PDF directly inside the app when they start the exam.",
    )
    if question_pdf is not None:
        st.caption(f"Selected: **{question_pdf.name}** · {question_pdf.size / 1024:.0f} KB")

    st.divider()

    st.markdown("#### ➖ Negative Marking (Optional)")
    negative_marking = st.checkbox(
        "Enable negative marking for this exam (marks deducted for wrong answers; skipped questions are not penalized)",
        key="mentor_neg_marking",
    )
    negative_value = 0.0
    if negative_marking:
        negative_value = st.number_input(
            "Marks deducted per wrong answer (e.g. 0.25 is common for admission exams)",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.2f",
            key="mentor_neg_value",
        )
        st.caption(f"Example: out of {total_q}, 4 wrong answers would deduct {4 * negative_value:.2f} marks.")

    st.divider()

    answered = _count_answered(total_q)
    st.markdown(f"#### ③ ✏️ Fill the Answer Key ({answered}/{total_q} answered)")
    st.progress(answered / total_q if total_q else 0)

    tool_col1, tool_col2 = st.columns(2)
    with tool_col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["mentor_answers"] = {}
            st.session_state["mentor_answer_widget_version"] = (
                st.session_state.get("mentor_answer_widget_version", 0) + 1
            )
            st.rerun()
    with tool_col2:
        with st.popover("⌨️ Fill Quickly with Text", use_container_width=True):
            # Wrapped in a real st.form: without this, clicking "Apply Text"
            # right after typing - without first pressing Enter or clicking
            # somewhere else to blur the field - could submit the OLD,
            # not-yet-committed value of the text_input (a plain text_input
            # only guarantees its value is saved to session_state on blur/
            # Enter, and a bare button click can race that, especially
            # inside a popover). That's what was causing "You must enter
            # exactly 100 characters" even right after typing 100 of them.
            # A form batches every widget's live on-screen value together
            # at the exact moment its own submit button is clicked, which
            # removes that race entirely - Apply Text now always reads
            # exactly what's typed, every time.
            with st.form(key="quick_fill_form", clear_on_submit=True):
                text_val = st.text_input(
                    f"{total_q} characters (A/B/C/D), no spaces", key="quick_text_ans"
                )
                apply_clicked = st.form_submit_button("Apply Text", use_container_width=True)
            if apply_clicked:
                cleaned = text_val.strip().upper().replace(" ", "")
                if len(cleaned) != total_q or any(c not in "ABCD" for c in cleaned):
                    st.error(
                        f"You must enter exactly {total_q} A/B/C/D characters "
                        f"(got {len(cleaned)})."
                    )
                else:
                    # Keep ONE source of truth for the answer key.
                    # Incrementing the widget version forces the next render
                    # to create fresh radio widgets, so Streamlit cannot
                    # reuse a previously-mounted radio value over the newly
                    # applied Quick Fill answers.
                    new_answers = {i + 1: c for i, c in enumerate(cleaned)}
                    st.session_state["mentor_answers"] = new_answers
                    st.session_state["mentor_answer_widget_version"] = (
                        st.session_state.get("mentor_answer_widget_version", 0) + 1
                    )
                    st.session_state["mentor_answer_page"] = 1
                    st.rerun()

    _inject_bubble_grid_css()

    if total_q == 100:
        page = st.session_state.get("mentor_answer_page", 1)
        if page == 1:
            st.caption("Showing questions **1-50**")
            with st.container(key="answer_bubble_grid"):
                col1, col2 = st.columns(2)
                with col1:
                    _render_bubble_block(1, 25)
                with col2:
                    _render_bubble_block(26, 50)
            st.button("Next: 51-100 →", use_container_width=True,
                      on_click=_go_answer_page, args=(2,))
        else:
            st.caption("Showing questions **51-100**")
            with st.container(key="answer_bubble_grid"):
                col1, col2 = st.columns(2)
                with col1:
                    _render_bubble_block(51, 75)
                with col2:
                    _render_bubble_block(76, 100)
            st.button("← Back: 1-50", use_container_width=True,
                      on_click=_go_answer_page, args=(1,))
    else:
        # 40 and 50 exams share the same physical 50 / 40 OMR sheet,
        # but the mentor answer-key input must still show the correct
        # number of questions.  In particular, a 50-MCQ key must expose
        # Q1-Q50 (25 + 25), not stop at Q40.
        if total_q == 50:
            st.caption("Showing questions **1-50** (two columns of 25)")
            with st.container(key="answer_bubble_grid"):
                col1, col2 = st.columns(2)
                with col1:
                    _render_bubble_block(1, 25)
                with col2:
                    _render_bubble_block(26, 50)
        else:
            st.caption("Showing questions **1-40** (two columns of 20)")
            with st.container(key="answer_bubble_grid"):
                col1, col2 = st.columns(2)
                with col1:
                    _render_bubble_block(1, 20)
                with col2:
                    _render_bubble_block(21, 40)

    st.divider()

    if st.button("✅ Save Answer Key", type="primary", use_container_width=True):
        answered = _count_answered(total_q)
        if not exam_name.strip():
            st.error("Please enter an exam name.")
        elif question_pdf is None:
            st.error("Please upload the question PDF before saving this exam.")
        elif answered != total_q:
            st.error(f"You must answer all {total_q} questions (currently {answered} answered).")
        else:
            answer_string = _build_answer_string(total_q)
            # Recomputed here (not just reused from the live preview above)
            # so the saved value always matches exactly what's currently
            # selected in the form at the moment Save is clicked.
            start_dt_final, end_dt_final = _compute_exam_end_datetime(d, start_t, end_t)
            start_str = start_dt_final.strftime("%Y-%m-%d %H:%M")
            end_str = end_dt_final.strftime("%Y-%m-%d %H:%M")
            with st.spinner("Saving..."):
                try:
                    pdf_file_id, pdf_file_name = sh.upload_question_pdf(
                        question_pdf.getvalue(), question_pdf.name
                    )
                    key_id = sh.add_answer_key(
                        exam_name.strip(), d.strftime("%Y-%m-%d"), start_str, end_str,
                        total_q, answer_string,
                        negative_marking=negative_marking, negative_marks_value=negative_value,
                        duration_minutes=duration_min,
                        question_pdf_file_id=pdf_file_id,
                        question_pdf_name=pdf_file_name,
                    )
                except Exception as e:
                    st.error(f"Could not save the question PDF/exam: {e}")
                    return
                st.session_state["mentor_answers"] = {}
                for q in range(1, total_q + 1):
                    st.session_state.pop(f"ans_q_{q}", None)
                clear_all_caches()
            st.success(f"✅ Answer key for '{exam_name}' saved! Key ID: {key_id}")


# =========================================================================
# Mentor: Dashboard / Analytics
# =========================================================================

def page_mentor_dashboard():
    st.markdown(f"### 👋 Welcome, {sh.get_mentor_name()}")
    st.markdown("#### 📊 Dashboard")
    with st.spinner("Loading analytics..."):
        stats = sh.get_mentor_analytics()
    with st.container(key="card_mentor_dashboard_stats"):
        st.markdown(
            f"""
            <div class='metric-row'>
                <div class='metric-box'><div class='label'>Total Students</div><div class='value'>{stats['total_students']}</div></div>
                <div class='metric-box'><div class='label'>Active Students</div><div class='value'>{stats['active_students']}</div></div>
                <div class='metric-box'><div class='label'>Total Submissions</div><div class='value'>{stats['total_submissions']}</div></div>
                <div class='metric-box'><div class='label'>Submissions Today</div><div class='value'>{stats['submissions_today']}</div></div>
                <div class='metric-box'><div class='label'>Average Score</div><div class='value'>{stats['average_score_pct']}%</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if stats["active_exam"]:
            st.success(f"🟢 Active exam right now: **{stats['active_exam']}**")
        else:
            st.info("No exam is active right now.")


# =========================================================================
# Mentor: Leaderboard (reuses the same shared renderer as the student view)
# =========================================================================

def page_mentor_leaderboard():
    st.subheader("🏆 Leaderboard")
    render_leaderboard(sid=None, key_suffix="mentor")


# =========================================================================
# Mentor: Students (view / disable / per-student Test Analysis + drilldown)
# =========================================================================

def page_mentor_students():
    st.subheader("👥 Student Management")
    with st.spinner("Loading students..."):
        df = cached_students()
        results_df = cached_results()

    if df.empty:
        st.info("No students have signed up yet.")
        return

    search = st.text_input("🔍 Search by name or phone", key="mentor_student_search")
    if search:
        q = search.strip().lower()
        if q:
            name_match = df["name"].astype(str).str.lower().str.contains(q, na=False)
            phone_match = df["phone"].astype(str).str.contains(q, na=False)
            df = df[name_match | phone_match]

    if df.empty:
        st.info("No students match your search.")
        return

    for _, row in df.iterrows():
        sid = row["student_id"]
        disabled = sh._to_bool(row.get("disabled", False))
        student_results = results_df[results_df["student_id"].astype(str) == str(sid)] if not results_df.empty else results_df
        tests_taken = len(student_results)
        if tests_taken:
            avg_score = round((
                pd.to_numeric(student_results["marks"], errors="coerce") /
                pd.to_numeric(student_results["total"], errors="coerce").replace(0, np.nan)
            ).mean() * 100, 1)
            avg_score = 0.0 if pd.isna(avg_score) else avg_score
        else:
            avg_score = "-"

        with st.container(key=f"card_student_{sid}"):
            c1, c2, c3 = st.columns([3.2, 1.4, 1.3])
            with c1:
                status = "🔴 Disabled" if disabled else "🟢 Active"
                avatar_html = render_avatar(sid, row["name"], size=30, font_size=12)
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:8px;'>"
                    f"{avatar_html}<span><b>{row['name']}</b> &nbsp; {status}</span></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"ID: {sid} · 📱 {sh.format_bd_phone(row['phone'])} · Tests: {tests_taken} · Avg: {avg_score}%")
            with c2:
                if st.button("📊 View Analysis", key=f"analysis_{sid}", use_container_width=True):
                    st.session_state["mentor_analysis_sid"] = sid
                    st.session_state["mentor_analysis_name"] = row["name"]
                    st.session_state["mentor_analysis_page"] = 1
                    st.session_state.pop("mentor_analysis_view_key_id", None)
                    st.session_state["mentor_page"] = "m_students"
                    go_to("mentor_student_analysis")
            with c3:
                toggle_label = "Enable" if disabled else "Disable"
                if st.button(toggle_label, key=f"toggle_{sid}", use_container_width=True):
                    with st.spinner("Updating..."):
                        sh.set_student_disabled(sid, not disabled)
                        clear_all_caches()
                    st.rerun()

    st.caption("ℹ️ Student analysis is opened on a separate page, so even students with 100+ tests won't make this list unnecessarily long.")


# =========================================================================
# Mentor: Results (edit/override + export)
# =========================================================================

def page_mentor_results():
    st.subheader("🧾 Results")
    st.caption("Select an exam, then open a student's analysis. Student answers are view-only here.")
    keys_df = cached_answer_keys()
    if keys_df.empty:
        st.info("No exams created yet.")
        return
    keys_df = keys_df.iloc[::-1].reset_index(drop=True)
    options = {f"{r.get('exam_name') or r['key_id']} | {r['date']}": r['key_id'] for _, r in keys_df.iterrows()}
    choice = st.selectbox("Select exam", list(options.keys()), key="mentor_results_exam_select")
    key_id = options[choice]
    results = cached_results()
    exam_results = results[results["key_id"] == key_id].copy() if not results.empty else results
    if exam_results.empty:
        st.info("No submissions for this exam yet.")
        return
    exam_results = exam_results.sort_values("marks", ascending=False)
    exp1, exp2 = st.columns(2, gap="small")
    with exp1:
        st.download_button("⬇️ Export CSV", data=sh.df_to_csv_bytes(exam_results), file_name=f"{key_id}_results.csv", mime="text/csv", use_container_width=True)
    with exp2:
        st.download_button("⬇️ Export Excel", data=sh.df_to_excel_bytes(exam_results), file_name=f"{key_id}_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    st.markdown("#### Submissions")
    st.markdown("<div style='display:grid;grid-template-columns:24px minmax(0,1fr) 58px 58px 58px;gap:6px;padding:4px 9px;color:var(--mv-muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;'><span></span><span>Name</span><span>Marks</span><span>Correct</span><span>Wrong</span></div>", unsafe_allow_html=True)
    for _, row in exam_results.iterrows():
        review_needed = bool(str(row.get("omr_double_touch_json", "") or "").strip()) and str(row.get("review_status", "") or "") == ""
        with st.container(key=f"mentor_result_row_{row['student_id']}_{key_id}"):
            c1,c2,c3,c4,c5 = st.columns([0.35,3.2,0.9,0.9,0.9], gap="small")
            with c1:
                st.markdown(f"<div class='mv-review-dot'>{'⚠️' if review_needed else ''}</div>", unsafe_allow_html=True)
            with c2:
                if st.button(str(row["student"]), key=f"mentor_result_student_{row['student_id']}_{key_id}", use_container_width=True):
                    st.session_state["mentor_analysis_sid"] = row["student_id"]
                    st.session_state["mentor_analysis_name"] = row["student"]
                    st.session_state["mentor_analysis_view_key_id"] = key_id
                    go_to("mentor_student_analysis")
            with c3: st.markdown(f"**{row['marks']}**")
            with c4: st.markdown(str(int(row['correct'])))
            with c5: st.markdown(str(int(row['wrong_count'])))


# =========================================================================
# Mentor: OMR Sheet Setup - exactly two physical sheet geometries:
# 50 / 40 OMR and 100 OMR.  A 40-question exam uses Q1-Q40 on the same
# physical 50-question sheet; Q41-Q50 are silently ignored. This is mainly a
# REFERENCE setup step; each student still calibrates their own photo.
# setup step; the grid actually used to read each student's photo is
# always built from that student's own click-calibration (see
# page_tests_results), which is far more tolerant of camera angle/skew.
# =========================================================================

CALIB_LAYOUT_OPTIONS = [
    (50, "📄 50 / 40 OMR"),
    (100, "📄 100 OMR"),
]


def _calibration_status_summary(all_calibration):
    all_calibration = all_calibration or {}
    parts = []
    for total_q, label in CALIB_LAYOUT_OPTIONS:
        done = str(total_q) in all_calibration
        icon = "✅" if done else "⚪"
        parts.append(f"{icon} {label}: {'Set up' if done else 'Not set up yet'}")
    return parts


def page_mentor_calibration():
    st.subheader("🎯 OMR Sheet Setup (only needed once per layout)")
    st.caption("This records where each answer bubble sits on your blank OMR sheet, for each "
               "exam layout. Students will still calibrate their own photo before every "
               "submission (that's what's actually used to read their answers) - this page "
               "is mainly a reference/setup checklist for you.")

    all_calibration = sh.load_calibration() or {}
    for line in _calibration_status_summary(all_calibration):
        st.write(line)
    st.divider()

    layout_labels = [label for _, label in CALIB_LAYOUT_OPTIONS]
    layout_choice = st.radio(
        "Which layout are you setting up?", layout_labels,
        horizontal=True, key="calib_layout_choice",
    )
    total_q = next(tq for tq, label in CALIB_LAYOUT_OPTIONS if label == layout_choice)
    layout_key = str(total_q)

    if st.session_state.get("calib_active_layout") != total_q:
        st.session_state["calib_active_layout"] = total_q
        st.session_state["calib_points"] = []

    existing_layout_calibration = all_calibration.get(layout_key)
    force_key = f"force_recalibrate_{total_q}"

    if existing_layout_calibration and not st.session_state.get(force_key):
        st.success(f"✅ {layout_choice} sheet setup is already saved - no need to redo it.")
        with st.expander("View the currently saved setup"):
            st.json(existing_layout_calibration)
        st.caption("You don't need to visit this page again for this layout unless the sheet design changes.")
        if st.button("🔄 Redo This Layout's Setup", key=f"redo_{total_q}"):
            st.session_state[force_key] = True
            st.session_state["calib_points"] = []
            st.rerun()
        return

    if existing_layout_calibration:
        st.info("You're redoing this layout's setup - the old one will be replaced when you save.")
        if st.button("❌ Go Back to the Previous Setup", key=f"cancel_redo_{total_q}"):
            st.session_state[force_key] = False
            st.rerun()

    points_info = omr_scanner.calibration_points_info(total_q)

    st.markdown(
        f"Upload a **straight, clear photo of a blank {layout_choice.split(' ', 1)[1]} OMR sheet**, "
        f"then click these {len(points_info)} points on the image below in this order "
        "(a top and bottom point for every question block keeps the reading accurate even "
        "if the sheet isn't perfectly flat in the photo):"
    )
    for i, info in enumerate(points_info, start=1):
        st.markdown(f"{i}. **{info['full']}**")

    uploaded = st.file_uploader(
        "Upload blank OMR sheet", type=["png", "jpg", "jpeg"], key=f"calib_upload_{total_q}"
    )
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    image = ImageOps.exif_transpose(image)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # detect_and_warp() is intentionally NOT used here anymore - it sometimes
    # locked onto the wrong rectangle (e.g. just one printed block) and
    # cropped the image down to a tiny section. The student flow never used
    # it either and works reliably, so we just resize the original photo
    # for display instead. This calibration is reference-only.
    warped_display_bgr = omr_scanner.resize_max_dim(
        img_bgr, max_dim=omr_scanner.STUDENT_DISPLAY_MAX_DIM
    )
    warped_rgb = cv2.cvtColor(warped_display_bgr, cv2.COLOR_BGR2RGB)
    warped_pil = Image.fromarray(warped_rgb)

    if "calib_points" not in st.session_state:
        st.session_state["calib_points"] = []

    current_step = len(st.session_state["calib_points"])

    if current_step < len(points_info):
        st.info(f"Now click: **{points_info[current_step]['full']}**")
        coords = streamlit_image_coordinates(warped_pil, key=f"calib_img_{total_q}")
        if coords is not None:
            pt = (coords["x"], coords["y"])
            if not st.session_state["calib_points"] or st.session_state["calib_points"][-1] != pt:
                st.session_state["calib_points"].append(pt)
                st.rerun()
    else:
        st.success(f"All {len(points_info)} points have been clicked!")
        pts = st.session_state["calib_points"]
        for info, pt in zip(points_info, pts):
            st.write(f"- {info['short']}: {pt}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start Over", key=f"calib_restart_{total_q}"):
                st.session_state["calib_points"] = []
                st.rerun()
        with col2:
            if st.button("💾 Save Setup", type="primary", key=f"calib_save_{total_q}"):
                layout_calibration = {
                    info["key"]: pt for info, pt in zip(points_info, pts)
                }
                layout_calibration["total_questions"] = total_q
                updated_calibration = dict(all_calibration)
                updated_calibration[layout_key] = layout_calibration
                with st.spinner("Saving..."):
                    sh.save_calibration(updated_calibration)
                    clear_all_caches()
                st.success(f"{layout_choice} sheet setup saved!")
                st.session_state["calib_points"] = []
                st.session_state[force_key] = False


# =========================================================================
# Mentor: Profile (same card layout/system as the student Profile page -
# header with avatar + role badges, a "Profile Information" card with an
# Update Profile toggle, an Account Status card, a Log Out card, and
# Change Password tucked in an expander underneath. A mentor only really
# has a display name + password (no phone/birth date/gender - those are
# student-only concepts), so the editable surface here is intentionally
# smaller, but the visual system is identical.
# =========================================================================

def page_mentor_profile():
    name = sh.get_mentor_name()

    # ---- Stats used in the strip beside the header - the mentor-side
    # equivalent of the student's Tests Completed / Average Score /
    # Leaderboard Rank / Days Active strip. Reuses the same cached
    # analytics + answer-key list already used elsewhere, so this adds
    # no extra Google Sheets calls. ----
    stats = sh.get_mentor_analytics()
    keys_df = cached_answer_keys()
    exams_created = 0 if keys_df.empty else len(keys_df)

    header_col, stats_col = st.columns([1.3, 2.4], gap="medium")

    # ---- Header: avatar + name + role/verified badges (same structure
    # as the student profile header). Mentor's avatar deliberately uses
    # the app's accent color (not the random per-student palette) so it
    # reads as a distinct "admin" identity at a glance. ----
    with header_col:
        mentor_avatar_html = (
            f"<span style='display:inline-flex; align-items:center; justify-content:center; width:64px; height:64px;"
            f"min-width:64px; border-radius:50%; background:var(--mv-accent); color:#fff; font-family:var(--sans);"
            f"font-weight:700; font-size:24px; letter-spacing:.02em;'>{_avatar_initials(name)}</span>"
        )
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:16px; margin-bottom:14px; flex-wrap:wrap;'>
                {_profile_hero_avatar_html(mentor_avatar_html, "#F94D10")}
                <div>
                    <div style='font-family:var(--serif); font-weight:600; font-size:23px; color:var(--mv-ink); line-height:1.2;'>{name}</div>
                    <div style='display:flex; align-items:center; gap:8px; margin-top:4px;'>
                        <span style='font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--mv-muted); font-weight:700;'>Mentor</span>
                        <span style='font-size:11px; padding:2px 10px; border-radius:999px; background:var(--mv-accent-soft); color:var(--mv-accent); font-weight:700;'>✓ Verified</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Stats strip: Total Students / Average Score / Total
    # Submissions / Exams Created, each with a "View X →" shortcut into
    # the page that actually shows that data - same shared renderer and
    # visual system as the student Profile page. Each link uses the
    # "mentor:<mentor_page_key>" convention (see render_profile_stats_strip)
    # so it lands on the right tab inside the mentor panel. ----
    with stats_col:
        render_profile_stats_strip([
            {"icon": "👥", "icon_bg": "var(--mv-primary-soft)", "number": stats["total_students"],
             "label": "Total Students", "link_text": "View Students →", "go_to_page": "mentor:m_students"},
            {"icon": "📈", "icon_bg": "var(--mv-blue-soft)", "number": f"{stats['average_score_pct']}%",
             "label": "Average Score", "link_text": "View Results →", "go_to_page": "mentor:m_results"},
            {"icon": "🏆", "icon_bg": "var(--mv-accent-soft)", "number": stats["total_submissions"],
             "label": "Total Submissions", "link_text": "View Leaderboard →", "go_to_page": "mentor:m_leaderboard"},
            {"icon": "📝", "icon_bg": "var(--mv-purple-soft)", "number": exams_created,
             "label": "Exams Created", "caption": "Keep creating! 🚀" if exams_created else "Create your first! 🚀"},
        ])

    left_col, right_col = st.columns([1.7, 1], gap="medium")

    # ---- Left: Profile Information (view mode / edit mode), same toggle
    # pattern as the student page - just Display Name is editable here. ----
    with left_col:
        with st.container(key="card_profile_info"):
            hcol1, hcol2 = st.columns([2.4, 1.3])
            with hcol1:
                st.markdown("##### 👤 Profile Information")
            with hcol2:
                edit_open = st.session_state.get("mentor_profile_edit_open", False)
                if st.button("✖ Cancel" if edit_open else "✏️ Update Profile",
                             key="mentor_profile_toggle_edit_btn", use_container_width=True):
                    st.session_state["mentor_profile_edit_open"] = not edit_open
                    st.rerun()

            if not st.session_state.get("mentor_profile_edit_open"):
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.markdown(_profile_info_row("🧑‍🏫", "Display Name", name, "var(--mv-accent-soft)"), unsafe_allow_html=True)
                with fcol2:
                    st.markdown(_profile_info_row("🎓", "Role", "MENTOR", "var(--mv-purple-soft)"), unsafe_allow_html=True)
            else:
                new_name = st.text_input("Display name", value=name, key="mentor_profile_name_input")
                if st.button("💾 Save Changes", type="primary", use_container_width=True, key="mentor_profile_save_btn"):
                    cleaned = new_name.strip()
                    if not cleaned:
                        st.error("Name cannot be empty.")
                    else:
                        with st.spinner("Updating your profile..."):
                            sh.set_mentor_name(cleaned)
                        st.session_state["mentor_profile_edit_open"] = False
                        st.success("Profile updated!")
                        st.rerun()

        # ---- Change Password lives right under Profile Information (in
        # the same left column), same as the student page - fills the
        # gap left by the shorter left column instead of sitting below
        # both columns as its own full-width section. ----
        with st.container(key="card_profile_changepw"):
            pw_open = st.session_state.get("mentor_profile_changepw_open", False)
            if st.button(f"🔑  Change Password {'▾' if pw_open else '▸'}",
                         key="mentor_profile_changepw_toggle_btn", use_container_width=True):
                st.session_state["mentor_profile_changepw_open"] = not pw_open
                st.rerun()
            st.caption("Update your password regularly to keep your account secure.")

            if pw_open:
                # Plain widgets (no st.form) so the strength bar updates live while typing.
                current_pw = st.text_input("Current password", type="password", key="mentor_prof_cur_pw")
                new_pw1 = st.text_input("New password", type="password", key="mentor_prof_new_pw1")
                if new_pw1:
                    score, label, _tips = sh.password_strength(new_pw1)
                    colors = ["#ef4444", "#ef4444", "#f59e0b", "#10b981", "#059669"]
                    st.markdown(
                        f"<div class='strength-bar'><div class='strength-fill' "
                        f"style='width:{(score+1)*20}%; background:{colors[score]};'></div></div>"
                        f"<small>Password strength: <b>{label}</b></small>",
                        unsafe_allow_html=True,
                    )
                new_pw2 = st.text_input("Confirm new password", type="password", key="mentor_prof_new_pw2")
                change_submitted = st.button("Update Password", type="primary", key="mentor_prof_pw_update_btn")

                if change_submitted:
                    if current_pw != sh.get_mentor_password():
                        st.error("Current password is incorrect.")
                    elif not new_pw1:
                        st.error("New password cannot be empty.")
                    elif new_pw1 != new_pw2:
                        st.error("New passwords don't match.")
                    else:
                        _, _, tips = sh.password_strength(new_pw1)
                        if tips:
                            st.error("New password is too weak: " + ", ".join(tips))
                        else:
                            with st.spinner("Updating..."):
                                sh.set_mentor_password(new_pw1)
                            st.session_state["mentor_authed"] = False
                            st.success("Password changed! Please log in again with the new password.")
                            st.rerun()

    # ---- Right: Account Status + Log Out, same cards as the student page ----
    with right_col:
        with st.container(key="card_profile_status"):
            st.markdown("##### 🛡️ Account Status")
            st.markdown(
                _profile_status_pill_html("Account Status", "Active", True),
                unsafe_allow_html=True,
            )

        with st.container(key="card_profile_logout"):
            st.markdown(
                "<div style='display:flex; align-items:center; gap:10px;'>"
                "<div class='mv-logout-icon'>🚪</div>"
                "<span style='font-weight:700; font-size:15px; color:var(--mv-ink);'>Log Out</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.caption("Sign out from the mentor panel securely.")
            if st.button("Log Out", use_container_width=True, key="mentor_profile_logout_btn"):
                st.session_state["mentor_authed"] = False
                go_to("home")


# =========================================================================
# Mentor Panel
# =========================================================================

def is_mentor():
    if st.session_state.get("mentor_authed"):
        return True
    render_hero("Mentor Portal", heading_html="Mentor Login", compact=True, pulse=False,
                show_badge=False, show_byline=False)
    with st.form(key="mentor_login_form", clear_on_submit=False):
        pw = st.text_input("Mentor password", type="password", key="mentor_pw")
        submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
    if submitted:
        if pw == sh.get_mentor_password():
            st.session_state["mentor_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect mentor password.")

    st.write("")
    if st.button("← Back to Student Login", use_container_width=True, key="mentor_back_to_student"):
        go_to("home")
    return False


MENTOR_NAV = [
    ("m_dashboard", "Dashboard"),
    ("m_answerkey", "Create Exam"),
    ("m_calibration", "OMR Sheet Setup"),
    ("m_students", "Students"),
    ("m_results", "Results"),
    ("m_leaderboard", "Leaderboard"),
    ("m_profile", "Profile"),
]

# Same native st.button(icon=":material/...:") approach as the student
# nav's nav_icon() - see that function's comment for why (real, first-
# party icon rendering instead of a fragile custom-SVG overlay).
MENTOR_NAV_MATERIAL_ICONS = {
    "m_dashboard": "dashboard",
    "m_answerkey": "edit_document",
    "m_calibration": "tune",
    "m_students": "group",
    "m_results": "fact_check",
    "m_leaderboard": "emoji_events",
    "m_profile": "person",
}


def mentor_nav_icon(page_key):
    name = MENTOR_NAV_MATERIAL_ICONS.get(page_key)
    return f":material/{name}:" if name else None


def render_mentor_top_nav(current_page):
    """Mentor equivalent of render_top_nav() - same logo-left / flat-nav /
    avatar-right desktop layout and same logo+hamburger mobile bar as the
    student panel, so the whole app shares one consistent navigation
    system instead of the mentor side looking like a different app.
    Profile is reached via the avatar (desktop) / drawer (mobile), same
    as the student side, and is excluded from the desktop pill row."""
    desktop_nav_items = [item for item in MENTOR_NAV if item[0] != "m_profile"]
    with st.container(key="top_nav"):
        logo_col, nav_col = st.columns([1.6, 8.4])
        with logo_col:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; height:100%; padding-top:2px;'>"
                f"<div style='width:26px; height:26px; flex-shrink:0;'>{LOGO_SVG}</div>"
                f"<span style='font-family:var(--serif); font-weight:600; font-size:16px; "
                f"color:var(--mv-ink); white-space:nowrap;'>Med Venture</span></div>",
                unsafe_allow_html=True,
            )
        with nav_col:
            cols = st.columns([1] * len(desktop_nav_items) + [0.85])
            for col, (page_key, label) in zip(cols[:-1], desktop_nav_items):
                with col:
                    is_active = current_page == page_key
                    if st.button(
                        label, key=f"mnav_{page_key}", use_container_width=True,
                        type="primary" if is_active else "secondary",
                        icon=mentor_nav_icon(page_key),
                    ):
                        st.session_state["mentor_page"] = page_key
                        st.session_state.pop("mentor_analysis_sid", None)
                        st.session_state.pop("mentor_analysis_view_key_id", None)
                        go_to("mentor")
            with cols[-1]:
                mentor_name = sh.get_mentor_name()
                initials = _avatar_initials(mentor_name)
                # Mentor avatar uses the app's fixed accent color (not the
                # random per-student palette) so it visibly reads as the
                # admin/mentor identity rather than "just another student".
                st.markdown(
                    f"<style>.st-key-top_nav .st-key-top_nav_avatar_btn button {{ "
                    f"background:var(--mv-accent) !important; border-color:var(--mv-accent) !important; color:#fff !important; "
                    f"border-radius:50% !important; width:34px !important; height:34px !important; "
                    f"min-height:34px !important; padding:0 !important; font-size:13px !important; "
                    f"font-weight:700 !important; }}</style>",
                    unsafe_allow_html=True,
                )
                # Avatar + a small decorative chevron beside it, same
                # visual pattern as the student nav's avatar.
                avatar_sub, chevron_sub = st.columns([2, 1])
                with avatar_sub:
                    with st.container(key="top_nav_avatar_btn"):
                        if st.button(initials, key="top_nav_mentor_avatar_click", help="Profile"):
                            st.session_state["mentor_page"] = "m_profile"
                            st.session_state.pop("mentor_analysis_sid", None)
                            st.session_state.pop("mentor_analysis_view_key_id", None)
                            go_to("mentor")
                with chevron_sub:
                    st.markdown("<div class='mv-nav-avatar-chevron'>⌄</div>", unsafe_allow_html=True)

    # Mobile: same simplified logo + hamburger/close toggle bar as the
    # student panel - no separate quick-access icon button, since Profile
    # is just one tap away in the drawer below like every other nav item.
    with st.container(key="mobile_top_bar"):
        is_open = st.session_state.get("mentor_mobile_menu_open", False)
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:6px; height:40px;'>"
            f"<div style='width:24px; height:24px; flex-shrink:0;'>{LOGO_SVG}</div>"
            f"<span style='font-family:var(--serif); font-weight:600; font-size:15px; "
            f"color:var(--mv-ink); white-space:nowrap;'>Med Venture</span></div>",
            unsafe_allow_html=True,
        )
        with st.container(key="mobile_top_bar_right"):
            if st.button("✕" if is_open else "☰", key="mentor_mobile_menu_toggle", help="Open menu" if not is_open else "Close menu"):
                st.session_state["mentor_mobile_menu_open"] = not is_open
                st.rerun()

    if st.session_state.get("mentor_mobile_menu_open", False):
        st.markdown("<div class='mv-drawer-backdrop'></div>", unsafe_allow_html=True)
        with st.container(key="mobile_menu_mentor"):
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:18px;'>"
                f"<div style='width:24px; height:24px;'>{LOGO_SVG}</div>"
                f"<span style='font-family:var(--serif); font-weight:600; font-size:15px; color:var(--mv-ink);'>Med Venture</span></div>",
                unsafe_allow_html=True,
            )
            for page_key, label in MENTOR_NAV:
                if st.button(label, key=f"mmnav_{page_key}", use_container_width=True, icon=mentor_nav_icon(page_key)):
                    st.session_state["mentor_page"] = page_key
                    st.session_state.pop("mentor_analysis_sid", None)
                    st.session_state.pop("mentor_analysis_view_key_id", None)
                    go_to("mentor")

    st.write("")


def page_mentor():
    if not is_mentor():
        return

    current = st.session_state.get("mentor_page", "m_dashboard")
    is_student_analysis = st.session_state.get("page") == "mentor_student_analysis"
    active_nav = "m_students" if is_student_analysis else current

    render_mentor_top_nav(active_nav)

    if is_student_analysis:
        page_mentor_student_analysis()
    elif current == "m_dashboard":
        page_mentor_dashboard()
    elif current == "m_answerkey":
        render_answer_key_tab()
    elif current == "m_calibration":
        page_mentor_calibration()
    elif current == "m_students":
        page_mentor_students()
    elif current == "m_results":
        page_mentor_results()
    elif current == "m_leaderboard":
        page_mentor_leaderboard()
    elif current == "m_profile":
        page_mentor_profile()

    # The Profile page already has its own "Log Out" card, so we don't
    # repeat a second logout control down here when Profile is the page
    # being viewed - only show this convenience logout on the other
    # mentor pages, same pattern as the student side.
    if current != "m_profile" or is_student_analysis:
        st.divider()
        if st.button("🚪 Log Out of Mentor Panel", key="mentor_bottom_logout"):
            st.session_state["mentor_authed"] = False
            go_to("home")


# =========================================================================
# Main
# =========================================================================

def main():
    inject_global_css()

    # The ENTIRE visible app body - the app-password gate, the Google
    # Sheets "Connecting..." boot sequence, the mentor panel, the login/
    # signup screens, and every logged-in student page - is rendered
    # inside ONE placeholder that gets explicitly, synchronously reset
    # via .container() on every single run.
    #
    # WHY: an earlier version of this fix only wrapped the POST-LOGIN
    # page section, leaving the app-password gate and the boot/
    # "Connecting..." screen as separate, unwrapped top-level calls. That
    # gap is exactly what let a stale boot-loading screen from a previous
    # run stay visible, overlapping with the password form on the very
    # next run - most visibly right after the app woke up from Streamlit
    # Community Cloud's sleep-after-inactivity mode, since waking it
    # resets session_state and re-runs both the password gate and the
    # boot sequence in quick succession (see the "Connecting..." +
    # password form overlap screenshot this fixes). Wrapping everything,
    # including these two earlier screens, closes that gap the same way
    # it was already closed for page-to-page navigation.
    main_area = st.empty()
    with main_area.container():
        if not check_app_password():
            return

        # Only run the sheet-initialization/spinner once per browser
        # session - not on every single click/rerun. Re-running
        # init_sheets() on every interaction was one of the causes of the
        # extra delay/flicker on mobile (a "Connecting..." spinner
        # flashing on every tap).
        if not st.session_state.get("_sheets_ready"):
            render_boot_loading_screen("Connecting...")
            sh.init_sheets()
            st.session_state["_sheets_ready"] = True
            # A full rerun (rather than manually clearing a nested
            # placeholder) is what guarantees the boot screen is
            # completely gone before the real page renders - main_area is
            # a brand-new st.empty() each run, so the next run's content
            # fully replaces this one with no lingering overlap window.
            st.rerun()

        restore_page_from_url()

        role = st.session_state.get("role")
        is_student_logged_in = role == "student" and student_session_is_valid()

        if not is_student_logged_in and st.session_state.get("role") == "student":
            # session was invalidated (password changed / account disabled elsewhere)
            for k in ("student_id", "student_name", "session_version", "role"):
                st.session_state.pop(k, None)
            st.warning("Your session has expired (password may have changed elsewhere). Please log in again.")

        page = st.session_state.get("page", "home")

        if page in ("mentor", "mentor_student_analysis"):
            page_mentor()
        elif not is_student_logged_in:
            # Login page: the mentor entry point lives inline below the
            # login card (see page_student_auth), so no separate top
            # button here.
            page_student_auth()
        else:
            # Logged-in student pages: no separate "Mentor" button
            # anywhere in the top bar (desktop or mobile) any more -
            # Mentor Login now lives on the Profile page instead, so the
            # nav stays a clean, consistent set of items everywhere.
            render_top_nav(page)

            if page == "home":
                page_home()
            elif page in ("tests", "test_detail"):
                page_tests_results()
            elif page == "analysis":
                page_student_analysis()
            elif page == "leaderboard":
                page_leaderboard()
            elif page == "profile":
                page_profile()
            else:
                page_home()


if __name__ == "__main__":
    main()
