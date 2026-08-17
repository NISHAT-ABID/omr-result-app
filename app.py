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
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import omr_scanner
import sheets_helper as sh

st.set_page_config(page_title="OMR Result App", page_icon="📝", layout="centered")

# =========================================================================
# Global styling - one shared stylesheet for the whole app (mobile + desktop)
# =========================================================================

def inject_global_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 3.5rem;
            max-width: 820px;
        }
        * { transition: background-color .15s ease, color .15s ease, opacity .15s ease; }

        /* ---- Top nav (desktop: full row of buttons) ---- */
        .st-key-top_nav div[data-testid="stHorizontalBlock"] { gap: 6px; }
        .st-key-top_nav button {
            width: 100%;
            border-radius: 999px !important;
            border: 1px solid rgba(128,128,128,0.25) !important;
            padding: 8px 4px !important;
            font-size: 13px !important;
        }
        .st-key-top_nav button[kind="primary"] {
            border: none !important;
        }

        /* ---- Mobile top bar: hamburger (left) + profile icon (right) ---- */
        .st-key-mobile_top_bar { display: none; }
        .st-key-mobile_top_bar div[data-testid="stHorizontalBlock"] { gap: 6px; align-items: center; }
        /* Icon-only circle buttons (hamburger trigger + profile icon) */
        .st-key-mobile_top_bar button {
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 16px !important;
            margin: 0 auto !important;
            border: 1px solid rgba(128,128,128,0.25) !important;
        }

        /* ---- Mentor entry point (login page only - the small quiet link
           shown under the login card for logged-out users) ---- */
        .st-key-mentor_entry_login button {
            background: transparent !important;
            color: #b45309 !important;
            border: 1px solid rgba(180,83,9,0.45) !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            font-size: 11px !important;
            padding: 4px 10px !important;
            box-shadow: none !important;
        }
        .st-key-mentor_entry_login button:hover {
            background: rgba(180,83,9,0.08) !important;
        }
        .mentor-entry-caption {
            text-align: center;
            opacity: .65;
            font-size: 12px;
            margin-top: 22px;
            margin-bottom: 6px;
        }

        /* ---- Generic cards ---- */
        .app-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 14px;
            background: rgba(127,127,127,0.04);
        }
        .app-card h4 { margin-top: 0; }
        .metric-row { display: flex; gap: 10px; flex-wrap: wrap; }
        .metric-box {
            flex: 1 1 120px;
            border-radius: 12px;
            padding: 12px 14px;
            background: rgba(127,127,127,0.06);
            border: 1px solid rgba(128,128,128,0.18);
        }
        .metric-box .label { font-size: 12px; opacity: .7; margin-bottom: 2px; }
        .metric-box .value { font-size: 22px; font-weight: 700; }

        .rank-badge {
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-weight: 700; font-size: 13px;
        }
        .rank-gold { background:#fde68a; color:#78350f; }
        .rank-silver { background:#e5e7eb; color:#374151; }
        .rank-bronze { background:#fbcfe8; color:#831843; }
        .rank-you { background:#bfdbfe; color:#1e3a8a; }

        .lb-row {
            display:flex; align-items:center; gap:10px; padding:10px 12px;
            border-radius:10px; margin-bottom:6px; border:1px solid rgba(128,128,128,0.12);
            flex-wrap: wrap;
        }
        .lb-row.me { background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.4); }

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

        .strength-bar { height:6px; border-radius:4px; background:rgba(128,128,128,0.2); overflow:hidden; margin-top:4px;}
        .strength-fill { height:100%; border-radius:4px; }

        /* ---- Compact time row (exam create) ---- */
        .time-row-label { font-weight:600; padding-top:6px; font-size:14px; }

        /* ---- Fixed +880 phone prefix box ---- */
        .bd-phone-prefix {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 8px;
            padding: 9px 10px;
            text-align: center;
            font-weight: 600;
            opacity: .85;
            background: rgba(127,127,127,0.06);
        }

        /* ---- Phone field: force the +880 box and digit input to stay on
           ONE row, even on narrow/mobile screens (Streamlit's own columns
           stack vertically below ~640px by default - this overrides that
           just for the phone-number row). Matched via a substring on the
           container's key class so it works for every key_prefix
           (login/su/fp) without a separate rule for each. ---- */
        div[class*="_phone_row"] div[data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            gap: 8px !important;
        }
        div[class*="_phone_row"] div[data-testid="column"]:first-child {
            flex: 0 0 78px !important;
            width: 78px !important;
            min-width: 78px !important;
        }
        div[class*="_phone_row"] div[data-testid="column"]:last-child {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            width: auto !important;
        }

        @media (max-width: 640px) {
            /* Hide the wide desktop nav row and show the compact mobile bar instead */
            .st-key-top_nav { display: none !important; }
            .st-key-mobile_top_bar { display: block !important; }
            .metric-box { flex: 1 1 45%; }
            .lb-row { font-size: 13px; }
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


def clear_all_caches():
    cached_answer_keys.clear()
    cached_results.clear()
    cached_students.clear()
    sh.clear_data_caches()


# =========================================================================
# Routing helpers
# =========================================================================

def go_to(page, **params):
    st.session_state["page"] = page
    for k, v in params.items():
        st.session_state[k] = v
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

    st.markdown(
        "<h1 style='text-align:center;'>📝 OMR Result App</h1>"
        "<p style='text-align:center; color:gray;'>Enter the access password to continue</p>",
        unsafe_allow_html=True,
    )
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


def phone_field(key_prefix, placeholder="1712345678"):
    """A phone number field with a fixed, non-editable '+880' prefix - the
    student only ever types the 10 digits that follow. This removes the
    'leading 0 disappears' class of bug entirely (there's no 0 for the
    user to type or for anything to drop), and keeps every number stored
    in one consistent format. The '+880' box and the digit input are kept
    on ONE row (see the '_phone_row' CSS rule in inject_global_css) even
    on mobile, where Streamlit's columns would otherwise stack vertically.
    Returns whatever raw digits the user has typed so far (validate with
    sh.validate_bd_phone_digits before use)."""
    st.markdown("**Phone number**")
    with st.container(key=f"{key_prefix}_phone_row"):
        c1, c2 = st.columns([0.9, 3.1], gap="small")
        with c1:
            st.markdown("<div class='bd-phone-prefix'>+880</div>", unsafe_allow_html=True)
        with c2:
            digits = st.text_input(
                "Phone number", key=f"{key_prefix}_digits", label_visibility="collapsed",
                placeholder=placeholder, max_chars=10,
            )
    return digits


def student_session_is_valid():
    """Session security: if the password was changed (or account disabled)
    elsewhere, session_version on the sheet will differ from what we
    stored at login time - force logout."""
    sid = st.session_state.get("student_id")
    if not sid:
        return False
    live_version = sh.get_session_version(sid)
    if live_version is None:
        return False
    return live_version == st.session_state.get("session_version")


def page_student_auth():
    st.markdown("<h2 style='text-align:center;'>🎓 Student Login</h2>", unsafe_allow_html=True)
    tab_login, tab_signup, tab_forgot = st.tabs(["Log In", "Sign Up", "Forgot Password"])

    with tab_login:
        # Wrapped in a real st.form: this both (a) lets the browser detect
        # it as a login form for autofill / "remember password", and
        # (b) makes pressing Enter inside any field submit the form - no
        # extra click needed after autofill/paste. No live password-
        # strength feedback is needed on this tab, so a form (which only
        # reruns on submit) doesn't cost us anything here.
        with st.form(key="login_form", clear_on_submit=False):
            login_digits = phone_field("login")
            pw = st.text_input("Password", type="password", key="login_pw")
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
        if submitted:
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

    with tab_signup:
        # NOT wrapped in st.form on purpose: a form only reruns the script
        # when its submit button is clicked, so a password-strength meter
        # inside a form only ever updates AFTER you hit submit - which is
        # exactly the confusing behaviour we're fixing here. Plain widgets
        # rerun on every keystroke, so the strength bar updates live while
        # typing, before the button is ever pressed.
        name = st.text_input("Full name", key="su_name")
        phone_digits = phone_field("su")
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
                        st.success("Account created! Please log in from the 'Log In' tab.")

    with tab_forgot:
        st.caption("Reset your password using the security question you set at sign up.")
        f_phone_digits = phone_field("fp")
        ok_preview, err_preview, canonical_preview = sh.validate_bd_phone_digits(f_phone_digits)
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
        elif f_phone_digits.strip():
            if not ok_preview:
                st.caption(err_preview)
            else:
                st.warning("No account found with this phone number.")

    # ---- Small, quiet mentor entry point right below the login card ----
    st.markdown("<p class='mentor-entry-caption'>Are you a mentor?</p>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        with st.container(key="mentor_entry_login"):
            if st.button("Mentor Login", use_container_width=True, key="mentor_entry_login_btn"):
                go_to("mentor")


# =========================================================================
# Top navigation (persistent on every student page)
# =========================================================================

STUDENT_NAV = [("home", "🏠 Home"), ("tests", "📝 Tests & Results"),
               ("leaderboard", "🏆 Leaderboard"), ("profile", "👤 Profile")]


def render_top_nav(current_page):
    # ---- Desktop: full row of nav buttons (hidden on narrow screens via CSS) ----
    with st.container(key="top_nav"):
        cols = st.columns(len(STUDENT_NAV))
        for col, (page_key, label) in zip(cols, STUDENT_NAV):
            with col:
                is_active = current_page == page_key or (page_key == "tests" and current_page == "test_detail")
                if st.button(label, key=f"nav_{page_key}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    go_to(page_key)

    # ---- Mobile: compact bar - hamburger (☰, opens a popover with the
    # same nav links) on the left, a direct profile icon on the right.
    # Both render as small circle icon buttons (see CSS). Mentor Login is
    # NOT in this bar - it lives on the Profile page instead. Hidden on
    # desktop via CSS; shown only under the mobile breakpoint. ----
    with st.container(key="mobile_top_bar"):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            with st.popover("☰"):
                for page_key, label in STUDENT_NAV:
                    if st.button(label, key=f"mnav_pop_{page_key}", use_container_width=True):
                        go_to(page_key)
        with c3:
            if st.button("👤", key="mobile_profile_btn"):
                go_to("profile")

    st.write("")


# =========================================================================
# Student: Home
# =========================================================================

def page_home():
    sid = st.session_state["student_id"]
    name = st.session_state["student_name"]
    st.markdown(f"### 👋 Welcome, {name}")

    active = sh.get_active_answer_key()
    with st.container():
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
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
            upcoming = sh.get_upcoming_answer_key()
            if upcoming:
                st.info(f"No test is active right now. Next up: **{upcoming['exam_name'] or upcoming['key_id']}** "
                        f"at **{upcoming['start_dt'].strftime('%Y-%m-%d %H:%M')}**.")
            else:
                st.info("No test is active or upcoming right now.")
        st.markdown("</div>", unsafe_allow_html=True)

    results = cached_results()
    my_results = results[results["student_id"] == sid] if not results.empty else results

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Last Result")
    if my_results.empty:
        st.caption("You haven't submitted any test yet.")
    else:
        last = my_results.sort_values("timestamp", ascending=False).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Marks", last["marks"])
        c2.metric("Correct", int(last["correct"]))
        c3.metric("Wrong", int(last["wrong_count"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### 📈 Overall Progress")
    if my_results.empty:
        st.caption("Your progress will show up here after your first test.")
    else:
        tests_completed = len(my_results)
        avg_pct = round((my_results["marks"] / my_results["total"]).mean() * 100, 1)
        rank, out_of = sh.get_rank_for_student(sid)

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
            trend_html = f"<p>{arrow} {abs(diff)}% {'better' if diff >= 0 else 'lower'} than last month</p>"

        rank_html = f"<span class='rank-badge rank-you'>🏆 Current Rank: {rank} / {out_of}</span>" if rank else ""

        st.markdown(
            f"""
            <div class='metric-row'>
                <div class='metric-box'><div class='label'>Tests Completed</div><div class='value'>{tests_completed}</div></div>
                <div class='metric-box'><div class='label'>Average Score</div><div class='value'>{avg_pct}%</div></div>
            </div>
            <p style='margin-top:10px;'>{rank_html}</p>
            {trend_html}
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='app-card'>💡 <i>{motivation_for(sid)}</i></div>",
        unsafe_allow_html=True,
    )


# =========================================================================
# Student: Tests & Results (list + detail)
# =========================================================================

def render_omr_review(rows):
    """rows = omr_scanner.build_review_rows() output, already filtered to
    wrong + skipped only."""
    if not rows:
        st.success("🎉 No wrong or skipped answers!")
        return
    html = ["<div>"]
    for row in rows:
        q, given, correct_ans, status = row["q"], row["given"], row["correct"], row["status"]
        tag = "<span class='omr-tag wrong-tag'>Wrong</span>" if status == "wrong" else "<span class='omr-tag skip-tag'>Skipped</span>"
        bubbles = ""
        for opt in ["A", "B", "C", "D"]:
            cls = "omr-bubble"
            if opt == correct_ans:
                cls += " correct"
            elif status == "wrong" and opt == given:
                cls += " wrong"
            bubbles += f"<span class='{cls}'>{opt}</span>"
        html.append(
            f"<div class='omr-row'><span class='omr-qnum'>Q{q}</span>{tag}<span>{bubbles}</span></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


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

    active = sh.get_active_answer_key()
    calibration = sh.load_calibration()

    with st.container():
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("#### 📤 Submit Your OMR Sheet")
        if not calibration:
            st.error("The mentor hasn't calibrated the OMR sheet yet. Please check back later.")
        elif not active:
            st.info("No test is active right now.")
        elif sh.has_submitted(sid, active["key_id"]):
            st.success("✅ You've already submitted this test. Duplicate submissions aren't allowed.")
        else:
            st.caption(f"Active test: **{active['exam_name'] or active['key_id']}** · "
                       f"{active['total_questions']} questions")
            uploaded = st.file_uploader(
                "Upload a photo of your filled OMR sheet (camera or gallery)",
                type=["png", "jpg", "jpeg"], key="omr_upload",
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded photo", use_container_width=True)

                if st.button("📤 Submit & See Score", type="primary", use_container_width=True):
                    with st.spinner("Validating image..."):
                        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        ok, errors, warnings_ = omr_scanner.validate_omr_image(img_bgr)

                    if not ok:
                        for e in errors:
                            st.error(e)
                    else:
                        for w in warnings_:
                            st.warning(w)
                        with st.spinner("Reading your answers..."):
                            active_now = sh.get_active_answer_key()
                            if not active_now:
                                st.error("The test window has just closed. Your result can't be recorded.")
                            elif sh.has_submitted(sid, active_now["key_id"]):
                                st.warning("You've already submitted this test.")
                            else:
                                warped, warp_ok = omr_scanner.detect_and_warp(img_bgr)
                                if not warp_ok:
                                    st.warning("Couldn't clearly detect the sheet's corners - "
                                               "still scoring, but retake a straighter photo if the result looks wrong.")
                                grid = omr_scanner.build_grid(calibration)
                                student_answers = omr_scanner.read_answers(warped, grid)
                                key_string = active_now["answer_string"]
                                key_id = active_now["key_id"]

                                result = omr_scanner.score_answers(
                                    student_answers, key_string,
                                    negative_marking=active_now.get("negative_marking", False),
                                    negative_value=active_now.get("negative_marks_value", 0.0),
                                )
                                sh.append_result(sid, st.session_state["student_name"], key_id, result)
                                clear_all_caches()
                                st.success("✅ Result saved!")

                                r1, r2, r3 = st.columns(3)
                                r1.metric("Correct ✅", result["correct"])
                                r2.metric("Wrong ❌", result["wrong_count"])
                                r3.metric("Skipped ⚪", result["skipped"])
                                st.metric("🏆 Marks", result["marks"])

                                rows = omr_scanner.build_review_rows(student_answers, key_string)
                                review_rows = [r for r in rows if r["status"] in ("wrong", "skipped")]
                                st.markdown("#### Review")
                                render_omr_review(review_rows)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 📋 Test History")
    results = cached_results()
    keys_df = cached_answer_keys()
    my_results = results[results["student_id"] == sid] if not results.empty else results

    if my_results.empty:
        st.caption("No tests submitted yet.")
        return

    my_results = my_results.sort_values("timestamp", ascending=False)
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
        my_rank, _ = sh.get_rank_for_student(sid, key_id)
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
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.write(f"**Name:** {student['name']}")
    st.write(f"**Phone:** {sh.format_bd_phone(student['phone'])}")
    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("Are you a mentor?")
    if st.button("👨‍🏫 Mentor Login", use_container_width=True, key="profile_mentor_login_btn"):
        go_to("mentor")
    st.markdown("</div>", unsafe_allow_html=True)

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
            color: var(--text-color, inherit); opacity: 0.75; padding-top: 6px;
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
    # ---- Drilldown: mentor clicked "View" on one of a student's exams ----
    view = st.session_state.get("mentor_view_result")
    if view:
        view_sid, view_key_id = view
        results_df = cached_results()
        keys_df = cached_answer_keys()
        match = results_df[(results_df["student_id"] == view_sid) & (results_df["key_id"] == view_key_id)]
        key_match = keys_df[keys_df["key_id"] == view_key_id]
        if st.button("← Back to Students"):
            st.session_state["mentor_view_result"] = None
            st.rerun()
        if match.empty or key_match.empty:
            st.warning("Result not found.")
        else:
            render_result_detail(match.iloc[0], key_match.iloc[0])
        return

    st.subheader("👥 Student Management")
    with st.spinner("Loading students..."):
        df = cached_students()
        results_df = cached_results()
        keys_df = cached_answer_keys()

    if df.empty:
        st.info("No students have signed up yet.")
        return

    search = st.text_input("🔍 Search by name or phone")
    if search:
        mask = df["name"].astype(str).str.contains(search, case=False) | \
               df["phone"].astype(str).str.contains(search, case=False)
        df = df[mask]

    for _, row in df.iterrows():
        sid = row["student_id"]
        disabled = sh._to_bool(row.get("disabled", False))
        student_results = results_df[results_df["student_id"] == sid] if not results_df.empty else results_df
        tests_taken = len(student_results)
        avg_score = round((student_results["marks"] / student_results["total"]).mean() * 100, 1) if tests_taken else "-"

        with st.container():
            st.markdown("<div class='app-card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1.4])
            with c1:
                status = "🔴 Disabled" if disabled else "🟢 Active"
                st.markdown(f"**{row['name']}** &nbsp; {status}")
                st.caption(f"📱 {sh.format_bd_phone(row['phone'])} · Tests: {tests_taken} · Avg: {avg_score}%")
            with c2:
                toggle_label = "Enable" if disabled else "Disable"
                if st.button(toggle_label, key=f"toggle_{sid}", use_container_width=True):
                    with st.spinner("Updating..."):
                        sh.set_student_disabled(sid, not disabled)
                        clear_all_caches()
                    st.rerun()

            with st.expander(f"📊 Test Analysis — {row['name']}"):
                if student_results.empty:
                    st.caption("This student hasn't submitted any test yet.")
                else:
                    sr = student_results.sort_values("timestamp", ascending=False)
                    header_cols = st.columns([2.2, 1.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
                    for c, label in zip(header_cols, ["Exam", "Date", "Total", "Correct", "Wrong", "Skipped", "Marks", ""]):
                        c.markdown(f"**{label}**")
                    for _, r in sr.iterrows():
                        key_match = keys_df[keys_df["key_id"] == r["key_id"]] if not keys_df.empty else pd.DataFrame()
                        exam_name = (
                            key_match.iloc[0]["exam_name"]
                            if not key_match.empty and key_match.iloc[0]["exam_name"]
                            else r["key_id"]
                        )
                        cols = st.columns([2.2, 1.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
                        cols[0].write(exam_name)
                        cols[1].write(str(r["timestamp"]).split(" ")[0])
                        cols[2].write(int(r["total"]))
                        cols[3].write(int(r["correct"]))
                        cols[4].write(int(r["wrong_count"]))
                        cols[5].write(int(r["skipped"]))
                        cols[6].write(r["marks"])
                        # Mentor drilldown: same full report (score + wrong/skipped
                        # bubble review) the student sees on their own history page.
                        if cols[7].button("View", key=f"mview_{sid}_{r['key_id']}"):
                            st.session_state["mentor_view_result"] = (sid, r["key_id"])
                            st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption("ℹ️ Students reset their own forgotten passwords from the login page's "
               "'Forgot Password' tab (using their security question) - no mentor action needed.")


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
# Mentor: OMR Sheet Setup (calibration)
# =========================================================================

def page_mentor_calibration():
    st.subheader("🎯 OMR Sheet Setup (only needed once)")
    st.caption("This tells the app exactly where each answer bubble sits on your OMR sheet, "
               "so it can automatically read every student's scanned sheet correctly.")

    existing_calibration = sh.load_calibration()
    if existing_calibration and not st.session_state.get("force_recalibrate"):
        st.success("✅ Sheet setup is already saved - no need to redo it.")
        with st.expander("View the currently active setup"):
            st.json(existing_calibration)
        st.caption("Students can submit OMR sheets normally. You don't need to visit this page again unless the sheet design changes.")
        if st.button("🔄 Redo Sheet Setup"):
            st.session_state["force_recalibrate"] = True
            st.session_state["calib_points"] = []
            st.rerun()
        return

    if existing_calibration:
        st.info("You're redoing the sheet setup - the old one will be replaced when you save.")
        if st.button("❌ Go Back to the Previous Setup"):
            st.session_state["force_recalibrate"] = False
            st.rerun()

    st.markdown(
        """
        Upload a **straight, clear photo of a blank OMR sheet**, then click 4 points
        on the image below in this order:
        1. Question **1** - center of bubble **A**
        2. Question **1** - center of bubble **D**
        3. Question **25** - center of bubble **A**
        4. Question **26** - center of bubble **A**
        """
    )
    uploaded = st.file_uploader("Upload blank OMR sheet", type=["png", "jpg", "jpeg"], key="calib_upload")
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    with st.spinner("Analyzing sheet..."):
        warped, ok = omr_scanner.detect_and_warp(img_bgr)
    if not ok:
        st.warning("Couldn't automatically detect the sheet's 4 corners. You can still click below to set it up, but retaking the photo straighter/flatter will help.")

    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    warped_pil = Image.fromarray(warped_rgb)

    if "calib_points" not in st.session_state:
        st.session_state["calib_points"] = []

    labels = ["Q1-A", "Q1-D", "Q25-A", "Q26-A"]
    current_step = len(st.session_state["calib_points"])

    if current_step < 4:
        st.info(f"Now click: **{labels[current_step]}**")
        coords = streamlit_image_coordinates(warped_pil, key="calib_img")
        if coords is not None:
            pt = (coords["x"], coords["y"])
            if not st.session_state["calib_points"] or st.session_state["calib_points"][-1] != pt:
                st.session_state["calib_points"].append(pt)
                st.rerun()
    else:
        st.success("All 4 points have been clicked!")
        pts = st.session_state["calib_points"]
        for lbl, pt in zip(labels, pts):
            st.write(f"- {lbl}: {pt}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start Over"):
                st.session_state["calib_points"] = []
                st.rerun()
        with col2:
            if st.button("💾 Save Setup", type="primary"):
                calibration = {"q1_a": pts[0], "q1_d": pts[1], "q25_a": pts[2], "q26_a": pts[3]}
                with st.spinner("Saving..."):
                    sh.save_calibration(calibration)
                st.success("Sheet setup saved! Students can now upload OMR sheets.")
                st.session_state["calib_points"] = []
                st.session_state["force_recalibrate"] = False


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


# =========================================================================
# Mentor Panel
# =========================================================================

def is_mentor():
    if st.session_state.get("mentor_authed"):
        return True
    st.markdown("### 👨‍🏫 Mentor Login")
    with st.form(key="mentor_login_form", clear_on_submit=False):
        pw = st.text_input("Mentor password", type="password", key="mentor_pw")
        submitted = st.form_submit_button("Mentor Login", type="primary")
    if submitted:
        if pw == sh.get_mentor_password():
            st.session_state["mentor_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect mentor password.")
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
    st.header("👨‍🏫 Mentor Panel")
    if not is_mentor():
        return

    current = st.session_state.get("mentor_page", "m_dashboard")

    # ---- Desktop: full nav row ----
    with st.container(key="top_nav"):
        cols = st.columns(len(MENTOR_NAV))
        for col, (page_key, label) in zip(cols, MENTOR_NAV):
            with col:
                if st.button(label, key=f"mnav_{page_key}", use_container_width=True,
                             type="primary" if current == page_key else "secondary"):
                    st.session_state["mentor_page"] = page_key
                    st.rerun()

    # ---- Mobile: hamburger with the same links + logout icon spot ----
    with st.container(key="mobile_top_bar"):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            with st.popover("☰"):
                for page_key, label in MENTOR_NAV:
                    if st.button(label, key=f"mmnav_pop_{page_key}", use_container_width=True):
                        st.session_state["mentor_page"] = page_key
                        st.rerun()
        with c3:
            if st.button("🚪", key="mobile_mentor_logout_btn",
                         help="Log out of Mentor Panel"):
                st.session_state["mentor_authed"] = False
                go_to("home")

    st.write("")

    if current == "m_dashboard":
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

    st.divider()
    if st.button("🚪 Log Out of Mentor Panel"):
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
        with st.spinner("Connecting..."):
            sh.init_sheets()
        st.session_state["_sheets_ready"] = True

    restore_page_from_url()

    role = st.session_state.get("role")
    is_student_logged_in = role == "student" and student_session_is_valid()

    if not is_student_logged_in and st.session_state.get("role") == "student":
        # session was invalidated (password changed / account disabled elsewhere)
        for k in ("student_id", "student_name", "session_version", "role"):
            st.session_state.pop(k, None)
        st.warning("Your session has expired (password may have changed elsewhere). Please log in again.")

    page = st.session_state.get("page", "home")

    if page == "mentor":
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
    elif page == "leaderboard":
        page_leaderboard()
    elif page == "profile":
        page_profile()
    else:
        page_home()


if __name__ == "__main__":
    main()
