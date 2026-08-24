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
from datetime import datetime, date, time as dtime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
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
            --mv-bg: #10201C;
            --mv-surface: #18302A;
            --mv-ink: #EAF2EF;
            --mv-primary: #4FB3A2;
            --mv-primary-hover: #6FC9BA;
            --mv-primary-soft: rgba(79,179,162,0.20);
            --mv-accent: #E68B75;
            --mv-accent-soft: rgba(230,139,117,0.18);
            --mv-muted: #9BAAA2;
            --mv-border: rgba(230,240,235,0.16);
            --mv-card-bg: rgba(230,240,235,0.06);
            --mv-dot: rgba(230,240,235,0.12);
            --mv-input-bg: #0E1F1A;
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
        .st-key-top_nav { margin-bottom: 10px; }
        .st-key-top_nav div[data-testid="stHorizontalBlock"] { gap: 8px; }
        .st-key-top_nav button {
            width: 100%;
            min-height: 40px;
            border-radius: 999px !important;
            border: 1px solid rgba(128,128,128,0.25) !important;
            padding: 7px 8px !important;
            font-size: 13px !important;
            white-space: nowrap !important;
        }
        .st-key-top_nav button[kind="primary"] { border: none !important; }

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
           display:none is in effect. ---- */
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
           throughout this file. */
        .mobile-menu-card, [class*="st-key-mobile_menu_"] {
            margin: 6px 0 10px;
            padding: 8px;
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 14px;
            background: rgba(127,127,127,0.045);
        }
        .mobile-menu-card button, [class*="st-key-mobile_menu_"] button {
            border-radius: 10px !important;
            min-height: 40px !important;
            text-align: left !important;
            margin-bottom: 4px !important;
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

        /* ---- Analysis / Test History cards on mobile: keeps the same
           "Exam name | Marks | Correct | Wrong | View" ROW layout used on
           desktop, instead of Streamlit's default behaviour of stacking
           st.columns() vertically below ~640px (which is what was making
           each metric render as one huge full-width number per line -
           a single card taking up the whole screen). This mirrors the
           same flex-row-nowrap technique already used for
           .st-key-test_history_table above, just tuned for this card's
           narrower columns: everything stays on one compact horizontal
           strip, with the exam-name column shrinking (ellipsis) before
           anything else does, and metric numbers/labels/the View button
           scaled down so the whole row still fits a phone screen without
           wrapping or overflowing. ---- */
        [class*="st-key-acard_"] div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            gap: 6px !important;
        }
        [class*="st-key-acard_"] div[data-testid="column"] {
            min-width: 0 !important;
            width: auto !important;
        }
        [class*="st-key-acard_"] div[data-testid="stMetric"] {
            text-align: center !important;
        }
        @media (max-width: 640px) {
            [class*="st-key-acard_"] div[data-testid="stHorizontalBlock"] { gap: 3px !important; }
            /* Exam-name column: allowed to shrink and ellipsis rather than
               push the metrics off-screen - this is the one column with
               genuinely variable-length content, so it's the one that
               should give way first. */
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(1) {
                flex: 1 1 74px !important;
                min-width: 0 !important;
            }
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(1) .analysis-title {
                font-size: 12.5px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(1) .analysis-subtle {
                font-size: 10px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            /* Marks / Correct / Wrong metric columns: fixed narrow width,
               small label + small value, so three of them plus the exam
               name and the View button all fit on one line. */
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(2),
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(3),
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(4) {
                flex: 0 0 44px !important;
                min-width: 0 !important;
                overflow: hidden !important;
            }
            [class*="st-key-acard_"] [data-testid="stMetricValue"] {
                font-size: 13px !important;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            [class*="st-key-acard_"] [data-testid="stMetricLabel"] p {
                font-size: 9px !important;
                white-space: nowrap;
            }
            /* View button column: fixed compact width, small pill button
               instead of the full-size default. */
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(5) {
                flex: 0 0 46px !important;
            }
            [class*="st-key-acard_"] div[data-testid="column"]:nth-child(5) .stButton > button {
                padding: 4px 2px !important;
                font-size: 11px !important;
                min-height: 30px !important;
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

        .strength-bar { height:6px; border-radius:4px; background:rgba(128,128,128,0.2); overflow:hidden; margin-top:4px; }
        .strength-fill { height:100%; border-radius:4px; }
        .time-row-label { font-weight:600; padding-top:6px; font-size:14px; }

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
    ("home", "🏠 Home"),
    ("tests", "📝 Tests & Results"),
    ("analysis", "📊 Analysis"),
    ("leaderboard", "🏆 Leaderboard"),
    ("profile", "👤 Profile"),
]


def render_top_nav(current_page):
    # Desktop: all navigation options stay visible on laptop/desktop.
    with st.container(key="top_nav"):
        cols = st.columns(len(STUDENT_NAV))
        for col, (page_key, label) in zip(cols, STUDENT_NAV):
            with col:
                is_active = current_page == page_key or (page_key == "tests" and current_page == "test_detail")
                if st.button(
                    label, key=f"nav_{page_key}", use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    go_to(page_key)

    # Mobile: a real toggle so the closed state is ☰ and the open state is ✕.
    # st.popover was removed because its trigger cannot reliably change to a
    # cross after opening. Rebuilt WITHOUT st.columns([1,3,1]) - a narrow
    # spacer column plus two icon columns was overflowing the actual phone
    # viewport width (Streamlit's own column engine reserves more minimum
    # width per column than fits three-across on a real small screen),
    # which pushed the profile icon off-screen and required horizontal
    # scrolling to reach it. The hamburger button now just sits as the
    # first, normal-flow element (top-left), and the profile button is
    # wrapped in its own small container that's pinned to the top-right
    # corner with CSS position:absolute - no column width math involved,
    # so nothing can overflow the screen.
    with st.container(key="mobile_top_bar"):
        is_open = st.session_state.get("student_mobile_menu_open", False)
        if st.button("✕" if is_open else "☰", key="student_mobile_menu_toggle", help="Open menu" if not is_open else "Close menu"):
            st.session_state["student_mobile_menu_open"] = not is_open
            st.rerun()
        # Only show the profile shortcut when the menu is CLOSED - once
        # open, the menu list below already has "Profile" in it, so
        # keeping this icon too was a redundant, confusing duplicate.
        if not is_open:
            with st.container(key="mobile_top_bar_right"):
                if st.button("👤", key="mobile_profile_btn", help="Profile"):
                    go_to("profile")

    if st.session_state.get("student_mobile_menu_open", False):
        with st.container(key="mobile_menu_student"):
            for page_key, label in STUDENT_NAV:
                if st.button(label, key=f"mnav_{page_key}", use_container_width=True):
                    go_to(page_key)

    st.write("")


# =========================================================================
# Student: Home
# =========================================================================

def page_home():
    sid = st.session_state["student_id"]
    name = st.session_state["student_name"]
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
            remaining = active["end_dt"] - sh.now_bd()
            mins_left = max(0, int(remaining.total_seconds() // 60))
            st.markdown(f"#### 🟢 Active Test: {active['exam_name'] or active['key_id']}")
            c1, c2 = st.columns(2)
            c1.metric("Questions", active["total_questions"])
            c2.metric("Time Left", f"{mins_left} min")
            if already:
                st.info("You already submitted this test. Check it in Tests & Results.")
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


def render_result_detail(result_row, key_row):
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
    if bool(result_row.get("edited_by_mentor")):
        st.caption("ℹ️ This result was reviewed/edited by your mentor.")

    import json as _json
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

    st.markdown("#### Wrong & Skipped Answers")
    render_omr_review(rows)


def _reset_submission_state():
    """Clears every piece of session_state used by the per-submission photo
    calibration flow below - called once a submission is saved (or when a
    new photo is uploaded) so leftover state from a previous photo never
    leaks into the next one."""
    for k in ("submit_file_sig", "submit_prepared_image", "submit_validation", "submit_calib_points"):
        st.session_state.pop(k, None)


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
                st.rerun()
            render_result_detail(match.iloc[0], key_match.iloc[0])
        return

    st.markdown("### 📝 Submit OMR / Test History")

    active = cached_active_answer_key()

    with st.container(key="card_submit_omr"):
        st.markdown("#### 📤 Submit Your OMR Sheet")
        if not active:
            st.info("No test is active right now.")
        elif sh.has_submitted(sid, active["key_id"]):
            st.success("✅ You've already submitted this test. Duplicate submissions aren't allowed.")
        else:
            total_q = active["total_questions"]
            st.caption(f"Active test: **{active['exam_name'] or active['key_id']}** · "
                       f"{total_q} questions")
            uploaded = st.file_uploader(
                "Upload a clear, straight photo of your FULL filled OMR sheet (camera or gallery). "
                "Make sure all 4 corners of the sheet are visible in the frame.",
                type=["png", "jpg", "jpeg"], key="omr_upload",
            )

            if uploaded is None:
                _reset_submission_state()
            else:
                # A stable signature for "is this the same photo as last rerun?" -
                # if the student swaps the photo, every bit of calibration/
                # validation state for the OLD photo must be thrown away.
                file_sig = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.get("submit_file_sig") != file_sig:
                    _reset_submission_state()
                    st.session_state["submit_file_sig"] = file_sig

                # ---- Step 0: prepare the photo once per upload (orient + validate) ----
                if "submit_prepared_image" not in st.session_state:
                    pil_img = Image.open(uploaded).convert("RGB")
                    # Fixes photos that come out sideways/upside-down because of
                    # phone camera EXIF orientation - keeps the sheet upright and
                    # fully visible, which the calibration clicks below depend on.
                    pil_img = ImageOps.exif_transpose(pil_img)
                    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # Quality checks ALWAYS run on the original, full-resolution
                    # photo - resizing happens only after, for display/calibration.
                    ok, errors, warnings_ = omr_scanner.validate_omr_image(orig_bgr)
                    proc_bgr = omr_scanner.resize_max_dim(orig_bgr) if ok else orig_bgr

                    st.session_state["submit_prepared_image"] = proc_bgr
                    st.session_state["submit_validation"] = (ok, errors, warnings_)

                img_bgr = st.session_state["submit_prepared_image"]
                ok, errors, warnings_ = st.session_state["submit_validation"]

                display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                display_pil = Image.fromarray(display_rgb)
                st.image(display_rgb, caption="Your uploaded sheet - full photo", use_container_width=True)

                if not ok:
                    for e in errors:
                        st.error(e)
                else:
                    for w in warnings_:
                        st.warning(w)

                    points_info = omr_scanner.calibration_points_info(total_q)
                    calib_points = st.session_state.get("submit_calib_points", [])
                    total_points = len(points_info)

                    st.markdown("#### 🎯 Calibrate Your Sheet")
                    st.caption(
                        f"Tap the exact CENTER of these {total_points} bubbles on YOUR photo above, "
                        "in order (a top AND bottom point for every question block, so the reading "
                        "stays accurate even if the sheet is a little curved, folded, or tilted in "
                        "the photo)."
                    )

                    if len(calib_points) < total_points:
                        step = points_info[len(calib_points)]
                        st.markdown(
                            f"<span class='calib-step-badge'>Step {len(calib_points) + 1} of {total_points}</span> "
                            f"&nbsp; Now tap: **{step['full']}**",
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
                        chip_html = "".join(
                            f"<span class='calib-point-chip'>{info['short']}: {pt}</span>"
                            for info, pt in zip(points_info, calib_points)
                        )
                        st.markdown(chip_html, unsafe_allow_html=True)

                        # Guards against duplicate submissions caused by a
                        # user double-clicking Submit or having two tabs
                        # open: while a submission for THIS photo is being
                        # processed, the button below is disabled so a
                        # second click can't even fire a second request.
                        # This is on top of (not instead of) the atomic
                        # server-side check in
                        # sh.append_result_if_not_submitted() below - the
                        # button-disable stops the common case (an
                        # impatient extra click) instantly with no network
                        # round trip, while the server-side check is what
                        # actually protects against two different tabs/
                        # devices racing each other.
                        submitting_key = f"submitting_{file_sig}"
                        is_submitting = st.session_state.get(submitting_key, False)

                        cb1, cb2 = st.columns(2)
                        with cb1:
                            if st.button("🔄 Redo Calibration Points", use_container_width=True,
                                         disabled=is_submitting):
                                st.session_state["submit_calib_points"] = []
                                st.rerun()
                        with cb2:
                            submit_clicked = st.button(
                                "📤 Submit & See Score", type="primary", use_container_width=True,
                                disabled=is_submitting,
                            )

                        if submit_clicked and not is_submitting:
                            st.session_state[submitting_key] = True
                            try:
                                with st.spinner("Reading your answers..."):
                                    active_now = sh.get_active_answer_key()
                                    if not active_now:
                                        st.error("The test window has just closed. Your result can't be recorded.")
                                    elif sh.has_submitted(sid, active_now["key_id"]):
                                        st.warning("You've already submitted this test.")
                                    else:
                                        calibration = {
                                            info["key"]: pt
                                            for info, pt in zip(points_info, calib_points)
                                        }
                                        grid = omr_scanner.build_grid(calibration, total_questions=active_now["total_questions"])
                                        radius = omr_scanner.compute_bubble_radius(img_bgr)
                                        student_answers = omr_scanner.read_answers(img_bgr, grid, radius=radius)
                                        key_string = active_now["answer_string"]
                                        key_id = active_now["key_id"]

                                        result = omr_scanner.score_answers(
                                            student_answers, key_string,
                                            negative_marking=active_now.get("negative_marking", False),
                                            negative_value=active_now.get("negative_marks_value", 0.0),
                                        )
                                        # Atomic check-and-write: re-verifies against a FRESH
                                        # (uncached) read of the Results sheet right before
                                        # writing, so a near-simultaneous duplicate request
                                        # (double-click that slipped past the disabled button,
                                        # a second open tab, or another device) is caught here
                                        # even if the has_submitted() check above was stale.
                                        saved = sh.append_result_if_not_submitted(
                                            sid, st.session_state["student_name"], key_id, result
                                        )
                                        clear_all_caches()

                                        if not saved:
                                            st.warning("You've already submitted this test (from another tab or device).")
                                        else:
                                            _reset_submission_state()
                                            st.success("✅ Result saved!")

                                            with st.container(key="card_submit_result"):
                                                r1, r2, r3, r4 = st.columns(4)
                                                r1.metric("Correct ✅", result["correct"])
                                                r2.metric("Wrong ❌", result["wrong_count"])
                                                r3.metric("Skipped ⚪", result["skipped"])
                                                r4.metric("🏆 Marks", result["marks"])

                                            rows = omr_scanner.build_review_rows(student_answers, key_string)
                                            review_rows = [r for r in rows if r["status"] in ("wrong", "skipped")]
                                            st.markdown("#### Review")
                                            render_omr_review(review_rows)
                            except Exception as e:
                                # Anything unexpected (network hiccup, Sheets API error,
                                # a bug in the scoring/reading code, etc.) lands here.
                                # Without this except+finally, an exception thrown
                                # anywhere above would skip straight past every
                                # "st.session_state[submitting_key] = False" line that
                                # used to be sprinkled through the branches, leaving the
                                # Submit button disabled FOREVER for this photo - the
                                # student would be stuck with no way to retry short of
                                # re-uploading a new photo. Catching here guarantees the
                                # button always becomes clickable again, and tells the
                                # student plainly that nothing was saved so they know a
                                # retry is safe (not a silent double-submit risk).
                                st.error(
                                    "Something went wrong while saving your result and it was "
                                    "NOT recorded. Please try submitting again."
                                )
                                st.caption(f"Technical detail: {e}")
                            finally:
                                # Runs no matter what happened above (success, a
                                # handled error/warning, or the exception path) - this
                                # is the single place that re-enables the Submit
                                # button, so there's exactly one thing to check when
                                # confirming the button can never get stuck.
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

    st.markdown(f"### 👤 {name}")
    st.caption(f"Student ID: **{sid}**")

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
            c2.metric("Marks", marks)
            c3.metric("Correct", correct)
            c4.metric("Wrong", wrong)
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
            st.markdown(f"### 👤 {name}")
            st.caption(f"Student ID: **{sid}**")
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
            st.markdown(f"### 👤 {name}")
            st.caption(f"Student ID: **{sid}**")
            render_result_detail(match.iloc[0], key_match.iloc[0])
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


def render_leaderboard_rows(df, mode, sid=None):
    for _, row in df.head(50).iterrows():
        rank = int(row["rank"])
        is_me = sid is not None and row["student_id"] == sid
        css_class = "lb-row me" if is_me else "lb-row"
        icon = _rank_icon(rank) if rank <= 3 else f"#{rank}"
        badge_class = _rank_class(rank)
        name_html = f"{row['student']}{' (You)' if is_me else ''}"

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

    render_leaderboard_rows(df, mode, sid=sid)

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

def page_profile():
    sid = st.session_state["student_id"]
    student = sh.get_student_by_id(sid)
    st.markdown("### 👤 Profile")
    with st.container(key="card_profile_info"):
        st.write(f"**Name:** {student['name']}")
        st.write(f"**Phone:** {sh.format_bd_phone(student['phone'])}")

    with st.expander("🔑 Change Password"):
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

    # ---- Mentor Login now lives here instead of a separate top-of-page
    # button/bar, per request: keeps the student nav to 4 items everywhere. ----
    with st.container(key="card_profile_mentor"):
        st.markdown("Are you a mentor?")
        if st.button("👨‍🏫 Mentor Login", use_container_width=True, key="profile_mentor_login_btn"):
            go_to("mentor")

    if st.button("🚪 Log Out", use_container_width=True):
        for k in ("student_id", "student_name", "session_version", "role"):
            st.session_state.pop(k, None)
        go_to("home")


# =========================================================================
# Mentor: Answer Key tab (native bubble-grid input)
# =========================================================================

def _inject_bubble_grid_css():
    st.markdown(
        """
        <style>
        .st-key-answer_bubble_grid div[data-testid="stRadio"] { margin-bottom: -14px; }
        .st-key-answer_bubble_grid div[data-testid="stRadio"] > label { display: none; }
        .st-key-answer_bubble_grid div[role="radiogroup"] { gap: 6px; }
        .st-key-answer_bubble_grid div[role="radiogroup"] label {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 999px;
            padding: 2px 10px 2px 6px;
            margin-right: 0 !important;
        }
        .q-num-badge {
            display: inline-block; min-width: 28px; font-weight: 600;
            color: var(--mv-ink); opacity: 0.75; padding-top: 6px;
        }
        /* Force number + options to stay on the SAME row on mobile too -
           Streamlit stacks st.columns vertically below ~640px by default,
           which is what was pushing the A/B/C/D options under the question
           number. Overriding with flex row + nowrap here keeps them side
           by side on every screen size, matching the desktop look. */
        .st-key-answer_bubble_grid div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            gap: 4px !important;
        }
        .st-key-answer_bubble_grid div[data-testid="column"] {
            width: auto !important;
            min-width: 0 !important;
        }
        .st-key-answer_bubble_grid div[data-testid="column"]:first-child {
            flex: 0 0 26px !important;
        }
        .st-key-answer_bubble_grid div[data-testid="column"]:last-child {
            flex: 1 1 auto !important;
            min-width: 0 !important;
        }
        @media (max-width: 640px) {
            .st-key-answer_bubble_grid div[role="radiogroup"] { gap: 4px !important; }
            .st-key-answer_bubble_grid div[role="radiogroup"] label {
                padding: 2px 6px 2px 4px !important;
                font-size: 12px !important;
            }
            .q-num-badge { min-width: 20px; font-size: 13px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _answers_store():
    """Single dict in session_state holding every question's chosen answer,
    independent from any individual widget's mount/unmount lifecycle. Using
    ONE dict (instead of one session_state key per question tied 1:1 to a
    widget) is what protects against Streamlit losing answers when a page
    of the bubble-grid (e.g. Q1-50) is unmounted while the mentor is on the
    other page (Q51-100) - the data lives here regardless of which
    questions are currently rendered on screen."""
    if "mentor_answers" not in st.session_state:
        st.session_state["mentor_answers"] = {}
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
    for q in range(q_start, q_end + 1):
        num_col, radio_col = st.columns([0.55, 3], gap="small")
        widget_key = f"ans_q_{q}"
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
    """Compact single-line time picker: a short label followed by
    Hour / Minute / AM-PM selects all on the same row (no extra caption
    row above each select, so Start + End together take just 2 lines)."""
    default_period = "PM" if default_hour_24 >= 12 else "AM"
    default_hour_12 = default_hour_24 % 12
    if default_hour_12 == 0:
        default_hour_12 = 12
    lbl_col, h_col, m_col, p_col = st.columns([1.1, 1, 1, 1])
    with lbl_col:
        st.markdown(f"<div class='time-row-label'>{label}</div>", unsafe_allow_html=True)
    with h_col:
        hour = st.selectbox("Hour", list(range(1, 13)), index=default_hour_12 - 1,
                             key=f"{key_prefix}_hour", label_visibility="collapsed")
    with m_col:
        minute = st.selectbox("Min", [f"{m:02d}" for m in range(60)], index=default_minute,
                               key=f"{key_prefix}_min", label_visibility="collapsed")
    with p_col:
        period = st.selectbox("AM/PM", ["AM", "PM"], index=0 if default_period == "AM" else 1,
                               key=f"{key_prefix}_period", label_visibility="collapsed")
    hour_24 = hour % 12
    if period == "PM":
        hour_24 += 12
    return dtime(hour_24, int(minute))


def render_answer_key_tab():
    st.subheader("🗓️ Create Exam & Set Answer Key")

    st.markdown("#### ① How many MCQs? (Exam Style)")
    exam_style = st.radio(
        "Exam Style", ["📄 100 Questions (Q1-100)", "📄 40 Questions (Q1-40)"],
        horizontal=True, label_visibility="collapsed", key="mentor_exam_style_choice",
    )
    total_q = 100 if "100" in exam_style else 40

    if total_q == 100:
        st.info("ℹ️ Questions 1-50 are shown first (in two columns of 25). "
                 "Scroll down and click **Next: 51-100** to enter the second half. "
                 "Your Q1-50 answers stay saved while you fill in 51-100.")
    else:
        st.info("ℹ️ Questions are shown in two columns: 1-20 and 21-40, on the same page.")

    if st.session_state.get("mentor_answer_total_q") != total_q:
        st.session_state["mentor_answers"] = {}
        for q in range(1, 101):
            st.session_state.pop(f"ans_q_{q}", None)
        st.session_state["mentor_answer_total_q"] = total_q
        st.session_state["mentor_answer_page"] = 1

    st.divider()

    st.markdown("#### ② Exam Details")
    exam_name = st.text_input("Exam name", placeholder="e.g. Physics Model Test - 3")
    d = st.date_input("Exam date", value=date.today())
    start_t = _time_input_12h("Start time", "mentor_start_t", default_hour_24=9, default_minute=0)
    end_t = _time_input_12h("End time", "mentor_end_t", default_hour_24=9, default_minute=30)

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
            for q in range(1, total_q + 1):
                st.session_state.pop(f"ans_q_{q}", None)
            st.rerun()
    with tool_col2:
        with st.popover("⌨️ Fill Quickly with Text", use_container_width=True):
            text_val = st.text_input(f"{total_q} characters (A/B/C/D), no spaces", key="quick_text_ans")
            if st.button("Apply Text"):
                cleaned = text_val.strip().upper().replace(" ", "")
                if len(cleaned) != total_q or any(c not in "ABCD" for c in cleaned):
                    st.error(f"You must enter exactly {total_q} A/B/C/D characters.")
                else:
                    store = _answers_store()
                    for i, c in enumerate(cleaned):
                        store[i + 1] = c
                        st.session_state.pop(f"ans_q_{i + 1}", None)
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
        elif answered != total_q:
            st.error(f"You must answer all {total_q} questions (currently {answered} answered).")
        else:
            answer_string = _build_answer_string(total_q)
            start_str = f"{d.strftime('%Y-%m-%d')} {start_t.strftime('%H:%M')}"
            end_str = f"{d.strftime('%Y-%m-%d')} {end_t.strftime('%H:%M')}"
            with st.spinner("Saving..."):
                key_id = sh.add_answer_key(
                    exam_name.strip(), d.strftime("%Y-%m-%d"), start_str, end_str,
                    total_q, answer_string,
                    negative_marking=negative_marking, negative_marks_value=negative_value,
                )
                st.session_state["mentor_answers"] = {}
                for q in range(1, total_q + 1):
                    st.session_state.pop(f"ans_q_{q}", None)
                clear_all_caches()
            st.success(f"✅ Answer key for '{exam_name}' saved! Key ID: {key_id}")


# =========================================================================
# Mentor: Dashboard / Analytics
# =========================================================================

def page_mentor_dashboard():
    st.subheader("📊 Mentor Dashboard")
    with st.spinner("Loading analytics..."):
        stats = sh.get_mentor_analytics()
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
                st.markdown(f"**{row['name']}** &nbsp; {status}")
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
    st.subheader("🧾 Results & Result Override")
    keys_df = cached_answer_keys()
    if keys_df.empty:
        st.info("No exams created yet.")
        return

    keys_df = keys_df.iloc[::-1].reset_index(drop=True)
    options = {f"{r.get('exam_name') or r['key_id']} | {r['date']}": r["key_id"] for _, r in keys_df.iterrows()}
    choice = st.selectbox("Select exam", list(options.keys()))
    key_id = options[choice]

    results = cached_results()
    exam_results = results[results["key_id"] == key_id] if not results.empty else results

    if exam_results.empty:
        st.info("No submissions for this exam yet.")
        return

    exam_results = exam_results.sort_values("marks", ascending=False)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            "⬇️ Export CSV", data=sh.df_to_csv_bytes(exam_results),
            file_name=f"{key_id}_results.csv", mime="text/csv", use_container_width=True,
        )
    with exp_col2:
        st.download_button(
            "⬇️ Export Excel", data=sh.df_to_excel_bytes(exam_results),
            file_name=f"{key_id}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("#### Submissions")
    for _, row in exam_results.iterrows():
        with st.expander(f"{row['student']} — Marks: {row['marks']}"
                          f"{' (edited)' if bool(row.get('edited_by_mentor')) else ''}"):
            with st.container(key=f"card_mentor_result_{row['student_id']}_{key_id}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Correct", int(row["correct"]))
                c2.metric("Wrong", int(row["wrong_count"]))
                c3.metric("Skipped", int(row["skipped"]))

            with st.form(key=f"edit_form_{row['student_id']}"):
                new_correct = st.number_input("Correct", min_value=0, max_value=int(row["total"]),
                                               value=int(row["correct"]))
                new_wrong = st.number_input("Wrong", min_value=0, max_value=int(row["total"]),
                                             value=int(row["wrong_count"]))
                new_marks = st.number_input("Marks (override)", value=float(row["marks"]), step=0.25)
                submitted = st.form_submit_button("💾 Save Override", type="primary")
                if submitted:
                    with st.spinner("Saving..."):
                        sh.update_result(
                            row["student_id"], key_id,
                            new_marks=new_marks, new_correct=new_correct, new_wrong_count=new_wrong,
                        )
                        clear_all_caches()
                    st.success("Result updated.")
                    st.rerun()


# =========================================================================
# Mentor: OMR Sheet Setup (calibration) - now supports both the 100Q and
# 40Q layouts, saved independently under the same "calibration" config
# entry (a dict keyed by "100" / "40"). This is now mainly a REFERENCE
# setup step; the grid actually used to read each student's photo is
# always built from that student's own click-calibration (see
# page_tests_results), which is far more tolerant of camera angle/skew.
# =========================================================================

CALIB_LAYOUT_OPTIONS = [
    (100, "📄 100 Questions"),
    (40, "📄 40 Questions"),
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
# Mentor: Settings (mentor password)
# =========================================================================

def page_mentor_settings():
    st.subheader("🔑 Change Mentor Password")
    st.caption("This password is for you (the mentor) only.")
    # Plain widgets (no st.form) so the strength bar updates live while typing.
    current_pw = st.text_input("Current password", type="password", key="cur_pw")
    new_pw1 = st.text_input("New password", type="password", key="new_pw1")
    if new_pw1:
        score, label, _tips = sh.password_strength(new_pw1)
        colors = ["#ef4444", "#ef4444", "#f59e0b", "#10b981", "#059669"]
        st.markdown(
            f"<div class='strength-bar'><div class='strength-fill' "
            f"style='width:{(score+1)*20}%; background:{colors[score]};'></div></div>"
            f"<small>Password strength: <b>{label}</b></small>",
            unsafe_allow_html=True,
        )
    new_pw2 = st.text_input("Re-enter new password", type="password", key="new_pw2")
    submitted = st.button("✅ Update Password", type="primary")

    if submitted:
        if current_pw != sh.get_mentor_password():
            st.error("Current password is incorrect.")
        elif not new_pw1:
            st.error("New password cannot be empty.")
        elif new_pw1 != new_pw2:
            st.error("The two new password entries don't match.")
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

    st.divider()
    if st.button("🚪 Log Out of Mentor Panel", key="mentor_settings_logout", use_container_width=True):
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
    ("m_dashboard", "📊 Dashboard"),
    ("m_answerkey", "📝 Create Exam"),
    ("m_calibration", "🎯 OMR Sheet Setup"),
    ("m_students", "👥 Students"),
    ("m_results", "🧾 Results"),
    ("m_leaderboard", "🏆 Leaderboard"),
    ("m_settings", "⚙️ Settings"),
]


def page_mentor():
    if not is_mentor():
        return

    st.header("👨‍🏫 Mentor Panel")

    current = st.session_state.get("mentor_page", "m_dashboard")
    is_student_analysis = st.session_state.get("page") == "mentor_student_analysis"
    active_nav = "m_students" if is_student_analysis else current

    # Desktop: all mentor options remain visible on laptop/desktop.
    with st.container(key="top_nav"):
        cols = st.columns(len(MENTOR_NAV))
        for col, (page_key, label) in zip(cols, MENTOR_NAV):
            with col:
                if st.button(
                    label, key=f"mnav_{page_key}", use_container_width=True,
                    type="primary" if active_nav == page_key else "secondary",
                ):
                    st.session_state["mentor_page"] = page_key
                    st.session_state.pop("mentor_analysis_sid", None)
                    st.session_state.pop("mentor_analysis_view_key_id", None)
                    go_to("mentor")

    # Mobile: ☰ when closed, ✕ when open. Rebuilt without st.columns([1,3,1])
    # for the same reason as the student nav above - three columns don't
    # reliably fit a real phone viewport, which pushed the Settings icon
    # off-screen. Hamburger is the normal first element (top-left);
    # Settings is pinned top-right via CSS on its own small container.
    with st.container(key="mobile_top_bar"):
        is_open = st.session_state.get("mentor_mobile_menu_open", False)
        if st.button("✕" if is_open else "☰", key="mentor_mobile_menu_toggle", help="Open menu" if not is_open else "Close menu"):
            st.session_state["mentor_mobile_menu_open"] = not is_open
            st.rerun()
        # Only show the Settings shortcut when the menu is CLOSED - once
        # open, "⚙️ Settings" is already in the list below, so keeping
        # both was a redundant, confusing duplicate. Settings itself has
        # both the password-change form and Log Out, matching the PC
        # experience (a Settings nav item, not a bare logout icon).
        if not is_open:
            with st.container(key="mobile_top_bar_right"):
                if st.button("⚙️", key="mobile_mentor_settings_btn", help="Settings"):
                    st.session_state["mentor_page"] = "m_settings"
                    st.session_state.pop("mentor_analysis_sid", None)
                    st.session_state.pop("mentor_analysis_view_key_id", None)
                    go_to("mentor")

    if st.session_state.get("mentor_mobile_menu_open", False):
        with st.container(key="mobile_menu_mentor"):
            for page_key, label in MENTOR_NAV:
                if st.button(label, key=f"mmnav_{page_key}", use_container_width=True):
                    st.session_state["mentor_page"] = page_key
                    st.session_state.pop("mentor_analysis_sid", None)
                    st.session_state.pop("mentor_analysis_view_key_id", None)
                    go_to("mentor")

    st.write("")

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
    elif current == "m_settings":
        page_mentor_settings()

    # The Settings page already has its own "Log Out of Mentor Panel"
    # button right below the password form, so we don't repeat a second
    # one (and a second explanatory caption) down here when Settings is
    # the page being viewed - only show this convenience logout on the
    # other mentor pages.
    if current != "m_settings" or is_student_analysis:
        st.divider()
        if st.button("🚪 Log Out of Mentor Panel", key="mentor_bottom_logout"):
            st.session_state["mentor_authed"] = False
            go_to("home")


# =========================================================================
# Main
# =========================================================================

def main():
    inject_global_css()

    if not check_app_password():
        return

    # Only run the sheet-initialization/spinner once per browser session -
    # not on every single click/rerun. Re-running init_sheets() on every
    # interaction was one of the causes of the extra delay/flicker on
    # mobile (a "Connecting..." spinner flashing on every tap).
    if not st.session_state.get("_sheets_ready"):
        boot_placeholder = st.empty()
        with boot_placeholder.container():
            render_boot_loading_screen("Connecting...")
        sh.init_sheets()
        st.session_state["_sheets_ready"] = True
        boot_placeholder.empty()

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
        return

    if not is_student_logged_in:
        # Login page: the mentor entry point lives inline below the login
        # card (see page_student_auth), so no separate top button here.
        page_student_auth()
        return

    # Logged-in student pages: no separate "Mentor" button anywhere in the
    # top bar (desktop or mobile) any more - Mentor Login now lives on the
    # Profile page instead, so the nav stays a clean, consistent 4 items
    # (Home / Tests & Results / Leaderboard / Profile) everywhere.
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
