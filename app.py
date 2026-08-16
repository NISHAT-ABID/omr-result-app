"""
app.py
------
OMR Result App - main Streamlit application.

Pages (all reachable from the top navbar):
- Home            -> submit an OMR sheet for the currently active exam
- My Tests        -> table of the logged-in student's past attempts
- Leaderboard     -> daily + overall rankings
- Profile         -> name, roll, best/lowest score, change password, logout

Login is per-student (Roll number + Password), stored in the "Users" sheet.
"Remember this device" is implemented with a browser cookie (extra-streamlit-
components) + a row in the "Sessions" sheet, so returning students don't have
to log in every time.

The Mentor Panel is still reached the same way as before (a small link, not
part of the main navbar), guarded by the shared mentor password/invite code.

Run with: streamlit run app.py
"""

import hashlib
import json
import uuid
from datetime import date
from datetime import time as dtime
from datetime import timedelta

import cv2
import extra_streamlit_components as stx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import omr_scanner
import sheets_helper as sh

st.set_page_config(page_title="OMR Result App", page_icon="📝", layout="centered")

REMEMBER_DAYS = 30
COOKIE_NAME = "omr_session_token"


# =====================================================================
# Small helpers
# =====================================================================

def hash_password(password, roll):
    """Simple salted hash (salt = the roll number itself)."""
    salt = str(roll).strip().lower()
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def make_token():
    return uuid.uuid4().hex


@st.cache_resource(show_spinner=False)
def get_cookie_manager():
    return stx.CookieManager(key="omr_cookie_manager")


def inject_global_css():
    st.markdown(
        """
        <style>
        .st-key-topnav {
            background: linear-gradient(90deg, #14162b 0%, #1b1035 100%);
            border-radius: 16px;
            padding: 10px 16px;
            margin-bottom: 18px;
            border: 1px solid rgba(140,120,255,0.25);
        }
        .st-key-topnav button {
            border-radius: 999px !important;
        }
        .navbar-brand-title { font-weight: 700; font-size: 17px; margin: 0; }
        .navbar-brand-sub { font-size: 11px; opacity: 0.6; margin: 0; }
        .score-row {
            border: 1px solid rgba(140,120,255,0.25);
            border-radius: 12px;
            padding: 10px 14px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_countdown(end_dt, label="Time Remaining"):
    """Pure-JS live ticking countdown - no server round-trips."""
    # end_dt is a naive datetime representing Bangladesh (UTC+6) wall-clock time.
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+06:00"
    html = f"""
    <div style="font-family: 'Source Sans Pro', sans-serif; text-align:center;
                padding: 10px 0 4px 0;">
      <div id="cd-time" style="font-size:30px; font-weight:800; letter-spacing:1px;
                                color:#c9c3ff;">--:--</div>
      <div style="font-size:12px; opacity:.65; margin-top:2px;">{label}</div>
    </div>
    <script>
      const end = new Date("{end_iso}").getTime();
      function tick() {{
        const now = new Date().getTime();
        let diff = end - now;
        const box = document.getElementById('cd-time');
        if (diff <= 0) {{
          box.innerText = "00:00:00";
          box.style.color = "#ff6b6b";
          clearInterval(timer);
          return;
        }}
        const h = Math.floor(diff / 3600000);
        const m = Math.floor((diff % 3600000) / 60000);
        const s = Math.floor((diff % 60000) / 1000);
        const pad = (n) => String(n).padStart(2, '0');
        box.innerText = (h > 0 ? pad(h) + ":" : "") + pad(m) + ":" + pad(s);
      }}
      tick();
      const timer = setInterval(tick, 1000);
    </script>
    """
    components.html(html, height=80)


# =====================================================================
# Auth: login / register / remember-device / logout
# =====================================================================

def try_auto_login(cookie_manager):
    if "user" in st.session_state:
        return
    token = cookie_manager.get(cookie=COOKIE_NAME)
    if not token:
        return
    session = sh.get_session(token)
    if not session:
        return
    user = sh.get_user(session.get("user_id"))
    if user:
        st.session_state["user"] = user
        st.session_state["session_token"] = token


def do_login(roll, password, remember, cookie_manager):
    user = sh.get_user(roll)
    if not user:
        st.error("No account found for that Roll number. Please register first.")
        return False
    if hash_password(password, roll) != str(user.get("password_hash", "")):
        st.error("Incorrect password.")
        return False

    st.session_state["user"] = user
    if remember:
        token = make_token()
        expires_str = (sh.now_bd() + timedelta(days=REMEMBER_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
        sh.create_session(token, user["user_id"], expires_str)
        cookie_manager.set(
            COOKIE_NAME, token,
            expires_at=sh.now_bd() + timedelta(days=REMEMBER_DAYS),
            key="set_omr_cookie",
        )
        st.session_state["session_token"] = token
    return True


def do_register(roll, name, password, confirm):
    roll = roll.strip()
    name = name.strip()
    if not roll or not name:
        st.error("Please fill in both Roll number and Name.")
        return False
    if sh.user_exists(roll):
        st.error("This Roll number is already registered. Please log in instead.")
        return False
    if len(password) < 4:
        st.error("Password must be at least 4 characters.")
        return False
    if password != confirm:
        st.error("Passwords do not match.")
        return False

    sh.create_user(roll, name, hash_password(password, roll), "student")
    st.session_state["user"] = sh.get_user(roll)
    return True


def do_logout(cookie_manager):
    token = st.session_state.get("session_token")
    if token:
        sh.delete_session(token)
    try:
        cookie_manager.delete(COOKIE_NAME, key="delete_omr_cookie")
    except KeyError:
        pass
    for k in ["user", "session_token", "page", "analysis_row", "mentor_authed"]:
        st.session_state.pop(k, None)
    st.rerun()


def render_auth_page(cookie_manager):
    st.markdown(
        "<h1 style='text-align:center;'>📝 OMR Result App</h1>"
        "<p style='text-align:center; color:gray;'>Smart. Fast. Accurate.</p>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            roll = st.text_input("Roll Number", key="login_roll")
            pw = st.text_input("Password", type="password", key="login_pw")
            remember = st.checkbox("Remember this device", value=True, key="login_remember")
            if st.button("Login", use_container_width=True, type="primary"):
                if not roll.strip() or not pw:
                    st.error("Please enter both Roll number and Password.")
                elif do_login(roll.strip(), pw, remember, cookie_manager):
                    st.success("Logged in!")
                    st.rerun()

        with tab_register:
            r_name = st.text_input("Full Name", key="reg_name")
            r_roll = st.text_input("Roll Number", key="reg_roll")
            r_pw1 = st.text_input("Password", type="password", key="reg_pw1")
            r_pw2 = st.text_input("Confirm Password", type="password", key="reg_pw2")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if do_register(r_roll, r_name, r_pw1, r_pw2):
                    token = make_token()
                    expires_str = (sh.now_bd() + timedelta(days=REMEMBER_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
                    sh.create_session(token, r_roll.strip(), expires_str)
                    cookie_manager.set(
                        COOKIE_NAME, token,
                        expires_at=sh.now_bd() + timedelta(days=REMEMBER_DAYS),
                        key="set_omr_cookie_reg",
                    )
                    st.session_state["session_token"] = token
                    st.success("Account created! You're logged in.")
                    st.rerun()

        st.caption("Are you a mentor? [Go to Mentor Login](?page=mentor)")


# =====================================================================
# Top navbar
# =====================================================================

NAV_ITEMS = [
    ("home", "🏠 Home"),
    ("mytests", "📋 My Tests"),
    ("leaderboard", "🏆 Leaderboard"),
    ("profile", "👤 Profile"),
]


def render_navbar():
    current = st.session_state.get("page", "home")
    with st.container(key="topnav"):
        cols = st.columns([2.4, 1, 1, 1, 1])
        with cols[0]:
            st.markdown(
                "<p class='navbar-brand-title'>📝 OMR Result App</p>"
                "<p class='navbar-brand-sub'>Smart. Fast. Accurate.</p>",
                unsafe_allow_html=True,
            )
        for i, (page_id, label) in enumerate(NAV_ITEMS):
            with cols[i + 1]:
                clicked = st.button(
                    label,
                    key=f"nav_{page_id}",
                    type="primary" if current == page_id else "secondary",
                    use_container_width=True,
                )
                if clicked:
                    st.session_state["page"] = page_id
                    st.session_state.pop("analysis_row", None)
                    st.rerun()


# =====================================================================
# Home page (submit OMR)
# =====================================================================

def page_home():
    user = st.session_state["user"]
    name = user["name"]

    calibration_100 = sh.load_calibration(100)
    calibration_40 = sh.load_calibration(40)

    active = sh.get_active_answer_key()

    if active:
        with st.container(border=True):
            st.markdown(f"### 🟢 Active Exam: {active['exam_name'] or active['key_id']}")
            c1, c2 = st.columns(2)
            c1.metric("Total Questions/Marks", active["total_questions"])
            if active.get("negative_marking"):
                c2.metric("Negative Marking", f"-{active['negative_marks_value']} / wrong")
            else:
                c2.metric("Negative Marking", "Off")
            render_countdown(active["end_dt"], label="Time Remaining")
    else:
        upcoming = sh.get_upcoming_answer_key()
        if upcoming:
            st.warning(
                f"No exam is active right now. Next exam: **{upcoming['exam_name'] or upcoming['key_id']}** "
                f"starts at **{upcoming['start_dt'].strftime('%Y-%m-%d %H:%M')}**."
            )
        else:
            st.info("No exam is active or upcoming right now.")

    calibration = None
    if active:
        calibration = calibration_100 if active["total_questions"] == 100 else calibration_40
        if calibration is None:
            calibration = sh.load_calibration(active["total_questions"])

    if active and not calibration:
        st.error("The mentor hasn't calibrated this sheet layout yet. Please ask the mentor to complete calibration first.")
        return

    st.markdown("#### Upload Your OMR Sheet")
    uploaded = st.file_uploader("Upload a photo of your filled OMR sheet", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded photo", use_container_width=True)

    disabled = not active or not uploaded
    if st.button("📤 Submit & See Score", type="primary", disabled=disabled, use_container_width=True):
        with st.spinner("Checking your answers..."):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            warped, ok = omr_scanner.detect_and_warp(img_bgr)
            if not ok:
                st.warning("Couldn't clearly detect the sheet's corners - still trying anyway. Retake the photo straighter if the result looks wrong.")

            active_now = sh.get_active_answer_key()
            if not active_now:
                st.error("No exam is active right now (outside the exam time window). The result cannot be recorded.")
                return

            calib_now = sh.load_calibration(active_now["total_questions"])
            grid = omr_scanner.build_grid(calib_now)
            student_answers = omr_scanner.read_answers(warped, grid)

            key_string = active_now["answer_string"]
            key_id = active_now["key_id"]
            end_dt = active_now["end_dt"]

            result = omr_scanner.score_answers(
                student_answers, key_string,
                negative_marking=active_now.get("negative_marking", False),
                negative_value=active_now.get("negative_marks_value", 0.0),
            )
            sh.append_result(name, key_id, result)
            st.success("✅ Result saved!")

            st.markdown("### 📊 Result Summary")
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("Total Questions", result["total"])
            r1c2.metric("Answered", result["answered"])
            r1c3.metric("Skipped", result["skipped"])

            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.metric("Correct ✅", result["correct"])
            r2c2.metric("Wrong ❌", result["wrong_count"])
            r2c3.metric("Accuracy", f"{result['accuracy']}%")

            st.metric("🏆 Marks", result["marks"])

            if result["negative_marking"]:
                st.caption(
                    f"Negative marking was ON: -{result['negative_value']} per wrong answer "
                    f"(skipped questions were not penalized)."
                )

            if result["wrong"]:
                window_closed = end_dt is not None and sh.now_bd() > end_dt
                with st.container(border=True):
                    st.markdown("#### ❌ Wrong Answers")
                    if window_closed:
                        rows = [
                            {
                                "Question": f"Q{q}",
                                "Your Answer": result["wrong_details"][q]["given"],
                                "Correct Answer": result["wrong_details"][q]["correct"],
                            }
                            for q in result["wrong"]
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.write("Question numbers you got wrong:", ", ".join(str(w) for w in result["wrong"]))
                        st.caption("Correct answers will be shown once the exam time window closes.")
            elif result["answered"] > 0:
                st.success("🎉 No wrong answers!")


# =====================================================================
# My Tests page
# =====================================================================

def page_my_tests():
    st.markdown("### 📋 My Tests")
    user = st.session_state["user"]
    name = user["name"]

    df = sh.get_results_for_student(name)
    if df.empty:
        st.info("You haven't submitted any test yet. Go to Home to submit your first OMR sheet.")
        return

    keys_df = sh.get_all_answer_keys()
    key_name_map = {}
    if not keys_df.empty:
        for _, r in keys_df.iterrows():
            key_name_map[r["key_id"]] = r.get("exam_name") or r["key_id"]

    for _, row in df.iterrows():
        key_id = row.get("key_id")
        exam_name = key_name_map.get(key_id, key_id)
        lb = sh.get_leaderboard_by_key(key_id)
        highest = lb.iloc[0]["marks"] if not lb.empty else row.get("marks")

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.2])
            with c1:
                st.markdown(f"**{exam_name}**")
                st.caption(row.get("timestamp", ""))
            c2.metric("Your Score", row.get("marks", 0))
            c3.metric("Highest Score", highest)
            with c4:
                if st.button("View Analysis", key=f"analysis_{row.get('timestamp')}_{key_id}", use_container_width=True):
                    st.session_state["analysis_row"] = row.to_dict()
                    st.session_state["page"] = "analysis"
                    st.rerun()


def page_analysis():
    row = st.session_state.get("analysis_row")
    if not row:
        st.session_state["page"] = "mytests"
        st.rerun()
        return

    if st.button("← Back to My Tests"):
        st.session_state["page"] = "mytests"
        st.session_state.pop("analysis_row", None)
        st.rerun()

    st.markdown("### 🔍 Result Analysis")
    st.caption(row.get("timestamp", ""))

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Total", row.get("total", 0))
    r1c2.metric("Correct ✅", row.get("correct", 0))
    r1c3.metric("Wrong ❌", row.get("wrong_count", 0))

    r2c1, r2c2 = st.columns(2)
    r2c1.metric("Marks", row.get("marks", 0))
    r2c2.metric("Accuracy", f"{row.get('accuracy', 0)}%")

    try:
        wrong_details = json.loads(row.get("wrong_details") or "{}")
    except Exception:
        wrong_details = {}

    if wrong_details:
        rows = [
            {"Question": f"Q{q}", "Your Answer": d.get("given", ""), "Correct Answer": d.get("correct", "")}
            for q, d in wrong_details.items()
        ]
        st.markdown("#### ❌ Wrong Answers")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("🎉 No wrong answers on this attempt!")


# =====================================================================
# Leaderboard page
# =====================================================================

def page_leaderboard():
    st.markdown("### 🏆 Leaderboard")
    tab1, tab2 = st.tabs(["📅 Daily Exam Leaderboard", "📊 Overall Analysis"])

    with tab1:
        keys_df = sh.get_all_answer_keys()
        if keys_df.empty:
            st.info("No exam/answer key has been set yet.")
        else:
            keys_df = keys_df.iloc[::-1].reset_index(drop=True)
            options = {}
            for _, row in keys_df.iterrows():
                label = f"{row.get('exam_name') or row['key_id']} | {row['date']} | {row['start_time'][-5:]}-{row['end_time'][-5:]}"
                options[label] = row["key_id"]
            choice = st.selectbox("Choose which exam's results to view", list(options.keys()))
            if st.button("🔄 Refresh", key="refresh_daily"):
                st.rerun()
            selected_key = options[choice]
            df = sh.get_leaderboard_by_key(selected_key)
            if df.empty:
                st.info("No results have been submitted for this exam yet.")
            else:
                show_df = df[["rank", "student", "marks", "correct", "wrong_count",
                               "skipped", "accuracy", "total", "timestamp"]].copy()
                show_df.columns = ["Rank", "Student", "Marks", "Correct", "Wrong",
                                    "Skipped", "Accuracy %", "Total", "Timestamp"]
                st.dataframe(show_df, use_container_width=True, hide_index=True)

    with tab2:
        st.caption("Ranking by average percentage across all exams combined.")
        if st.button("🔄 Refresh", key="refresh_overall"):
            st.rerun()
        df = sh.get_overall_leaderboard()
        if df.empty:
            st.info("No results have been submitted yet.")
        else:
            show_df = df[["rank", "student", "avg_percent", "exams_taken", "total_marks", "total_possible"]].copy()
            show_df.columns = ["Rank", "Student", "Average %", "Exams Taken", "Total Marks", "Total Possible"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)


# =====================================================================
# Profile page
# =====================================================================

def page_profile(cookie_manager):
    st.markdown("### 👤 Profile")
    user = st.session_state["user"]
    name = user["name"]

    df = sh.get_results_for_student(name)
    best = int(df["marks"].astype(float).max()) if not df.empty else "-"
    lowest = int(df["marks"].astype(float).min()) if not df.empty else "-"
    exams_taken = len(df)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Name**  \n{user['name']}")
        c2.markdown(f"**Roll**  \n{user['user_id']}")

        c3, c4, c5 = st.columns(3)
        c3.metric("Best Score", best)
        c4.metric("Lowest Score", lowest)
        c5.metric("Exams Taken", exams_taken)

    st.markdown("#### 🔑 Change Password")
    cur_pw = st.text_input("Current Password", type="password", key="profile_cur_pw")
    new_pw1 = st.text_input("New Password", type="password", key="profile_new_pw1")
    new_pw2 = st.text_input("Confirm New Password", type="password", key="profile_new_pw2")
    if st.button("Update Password", type="primary"):
        if hash_password(cur_pw, user["user_id"]) != str(user.get("password_hash", "")):
            st.error("Current password is incorrect.")
        elif len(new_pw1) < 4:
            st.error("New password must be at least 4 characters.")
        elif new_pw1 != new_pw2:
            st.error("The two new password entries don't match.")
        else:
            new_hash = hash_password(new_pw1, user["user_id"])
            sh.update_user_password(user["user_id"], new_hash)
            user["password_hash"] = new_hash
            st.session_state["user"] = user
            st.success("Password updated!")

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        do_logout(cookie_manager)


# =====================================================================
# Mentor panel (unchanged features, English text, reached via a link)
# =====================================================================

def is_mentor():
    if st.session_state.get("mentor_authed"):
        return True
    pw = st.text_input("Mentor password", type="password", key="mentor_pw")
    if st.button("Mentor Login"):
        if pw == sh.get_mentor_password():
            st.session_state["mentor_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect mentor password.")
    return False


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


def _answer_key(q):
    return f"ans_q_{q}"


def _count_answered(total_q):
    return sum(1 for q in range(1, total_q + 1) if st.session_state.get(_answer_key(q)) is not None)


def _build_answer_string(total_q):
    return "".join(st.session_state.get(_answer_key(q)) or "?" for q in range(1, total_q + 1))


def _render_bubble_block(q_start, q_end):
    for q in range(q_start, q_end + 1):
        num_col, radio_col = st.columns([0.55, 3], gap="small")
        with num_col:
            st.markdown(f"<div class='q-num-badge'>{q}</div>", unsafe_allow_html=True)
        with radio_col:
            st.radio(
                f"Q{q}", options=["A", "B", "C", "D"], index=None, horizontal=True,
                key=_answer_key(q), label_visibility="collapsed",
            )


def _time_input_12h(key_prefix, default_hour_24=9, default_minute=0):
    default_period = "PM" if default_hour_24 >= 12 else "AM"
    default_hour_12 = default_hour_24 % 12
    if default_hour_12 == 0:
        default_hour_12 = 12

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.caption("Hour")
        hour = st.selectbox("Hour", list(range(1, 13)), index=default_hour_12 - 1,
                             key=f"{key_prefix}_hour", label_visibility="collapsed")
    with c2:
        st.caption("Minute")
        minute = st.selectbox("Minute", [f"{m:02d}" for m in range(60)], index=default_minute,
                               key=f"{key_prefix}_min", label_visibility="collapsed")
    with c3:
        st.caption("AM/PM")
        period = st.selectbox("AM/PM", ["AM", "PM"], index=0 if default_period == "AM" else 1,
                               key=f"{key_prefix}_period", label_visibility="collapsed")

    hour_24 = hour % 12
    if period == "PM":
        hour_24 += 12
    return dtime(hour_24, int(minute))


def render_answer_key_tab():
    st.subheader("🗓️ Set Today's Answer Key & Exam Time")

    st.markdown("#### Step 1: How Many MCQs? (Exam Style)")
    exam_style = st.radio(
        "Exam Style", ["📄 100 Questions (Q1-100)", "📄 40 Questions (Q1-40)"],
        horizontal=True, label_visibility="collapsed", key="mentor_exam_style_choice",
    )
    total_q = 100 if "100" in exam_style else 40

    if st.session_state.get("mentor_answer_total_q") != total_q:
        for q in range(1, 101):
            st.session_state.pop(_answer_key(q), None)
        st.session_state["mentor_answer_total_q"] = total_q

    st.divider()
    st.markdown("#### Step 2: Exam Details")
    exam_name = st.text_input("Exam name", placeholder="e.g. Physics Model Test - 3")
    d = st.date_input("Exam date", value=date.today())
    st.markdown("**Start time**")
    start_t = _time_input_12h("mentor_start_t", default_hour_24=9, default_minute=0)
    st.markdown("**End time**")
    end_t = _time_input_12h("mentor_end_t", default_hour_24=9, default_minute=30)

    st.divider()
    st.markdown("#### ➖ Negative Marking (Optional)")
    negative_marking = st.checkbox(
        "Enable negative marking for this exam (marks are deducted for wrong answers; skipped questions are not penalized)",
        key="mentor_neg_marking",
    )
    negative_value = 0.0
    if negative_marking:
        negative_value = st.number_input(
            "How many marks should be deducted per wrong answer? (e.g. 0.25 is common for admission exams)",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.2f",
            key="mentor_neg_value",
        )
        st.caption(f"Example: 4 wrong answers out of {total_q} would deduct {4 * negative_value:.2f} marks.")

    st.divider()
    answered = _count_answered(total_q)
    st.markdown(f"#### Step 3: ✏️ Fill the Answer Key ({answered}/{total_q} answered)")
    st.progress(answered / total_q if total_q else 0)

    tool_col1, tool_col2 = st.columns(2)
    with tool_col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            for q in range(1, total_q + 1):
                st.session_state.pop(_answer_key(q), None)
            st.rerun()
    with tool_col2:
        with st.popover("⌨️ Fill Quickly with Text", use_container_width=True):
            text_val = st.text_input(f"{total_q} characters (A/B/C/D), no spaces", key="quick_text_ans")
            if st.button("Apply Text"):
                cleaned = text_val.strip().upper().replace(" ", "")
                if len(cleaned) != total_q or any(c not in "ABCD" for c in cleaned):
                    st.error(f"You must enter exactly {total_q} A/B/C/D characters.")
                else:
                    for i, c in enumerate(cleaned):
                        st.session_state[_answer_key(i + 1)] = c
                    st.rerun()

    _inject_bubble_grid_css()
    with st.container(key="answer_bubble_grid"):
        blocks = [(1, 25), (26, 50), (51, 75), (76, 100)] if total_q == 100 else [(1, 20), (21, 40)]
        grid_cols = st.columns(len(blocks))
        for col, (b_start, b_end) in zip(grid_cols, blocks):
            with col:
                _render_bubble_block(b_start, b_end)

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
            key_id = sh.add_answer_key(
                exam_name.strip(), d.strftime("%Y-%m-%d"), start_str, end_str,
                total_q, answer_string,
                negative_marking=negative_marking, negative_marks_value=negative_value,
            )
            for q in range(1, total_q + 1):
                st.session_state.pop(_answer_key(q), None)
            st.success(f"✅ Answer key for '{exam_name}' saved! Key ID: {key_id}")


def page_mentor():
    top_c1, top_c2 = st.columns([1, 5])
    with top_c1:
        if st.button("← Back"):
            st.session_state["page"] = "home"
            st.session_state.pop("mentor_authed", None)
            st.rerun()

    st.header("👨‍🏫 Mentor Panel")
    if not is_mentor():
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Answer Key", "🎯 Calibration", "📋 All Answer Keys", "🔑 Password"]
    )

    with tab1:
        render_answer_key_tab()

    with tab2:
        st.markdown("#### Sheet Layout")
        layout_choice = st.radio("Which sheet layout are you calibrating?",
                                  ["100 Questions", "40 Questions"], horizontal=True, key="calib_layout_choice")
        layout_q = 100 if layout_choice == "100 Questions" else 40

        existing_calibration = sh.load_calibration(layout_q)
        if existing_calibration and not st.session_state.get("force_recalibrate"):
            st.success(f"✅ Calibration for the {layout_q}-question layout is already saved.")
            with st.expander("View the currently active calibration"):
                st.json(existing_calibration)
            st.caption("Students can submit OMR sheets normally. You don't need to visit this page again unless the sheet design changes.")
            if st.button("🔄 Recalibrate"):
                st.session_state["force_recalibrate"] = True
                st.session_state["calib_points"] = []
                st.rerun()
        else:
            if existing_calibration:
                st.info("You're creating a new calibration - the old one will be replaced when you save.")
                if st.button("❌ Go Back to the Previous Calibration"):
                    st.session_state["force_recalibrate"] = False
                    st.rerun()

            st.subheader("🎯 OMR Sheet Calibration (only needed once per layout)")
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
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                warped, ok = omr_scanner.detect_and_warp(img_bgr)
                if not ok:
                    st.warning("Couldn't automatically detect the sheet's 4 corners. You can still click below to calibrate, but retaking the photo straighter/flatter will help.")

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
                        if st.button("💾 Save Calibration", type="primary"):
                            calibration = {
                                "q1_a": pts[0], "q1_d": pts[1],
                                "q25_a": pts[2], "q26_a": pts[3],
                            }
                            sh.save_calibration(calibration, layout_q)
                            st.success("Calibration saved! Students can now upload OMR sheets for this layout.")
                            st.session_state["calib_points"] = []
                            st.session_state["force_recalibrate"] = False

    with tab3:
        st.subheader("📋 All Answer Keys")
        df = sh.get_all_answer_keys()
        if df.empty:
            st.info("No answer key has been set yet.")
        else:
            show_cols = ["key_id", "exam_name", "date", "start_time", "end_time",
                         "total_questions", "negative_marking", "negative_marks_value"]
            show_cols = [c for c in show_cols if c in df.columns]
            display_df = df[show_cols].iloc[::-1].reset_index(drop=True)
            if "negative_marking" in display_df.columns:
                display_df["negative_marking"] = display_df["negative_marking"].apply(
                    lambda v: "Yes" if str(v).strip() in ("1", "True", "TRUE") else "No"
                )
            rename_map = {"negative_marking": "Negative Marking", "negative_marks_value": "Per Wrong"}
            display_df = display_df.rename(columns=rename_map)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("🔑 Change Mentor Password")
        st.caption("This password/invite code is for you (the mentor) only.")
        current_pw = st.text_input("Current password", type="password", key="cur_pw")
        new_pw1 = st.text_input("New password", type="password", key="new_pw1")
        new_pw2 = st.text_input("Re-enter new password", type="password", key="new_pw2")
        if st.button("✅ Update Password", type="primary"):
            if current_pw != sh.get_mentor_password():
                st.error("Current password is incorrect.")
            elif not new_pw1:
                st.error("New password cannot be empty.")
            elif new_pw1 != new_pw2:
                st.error("The two new password entries don't match.")
            else:
                sh.set_mentor_password(new_pw1)
                st.session_state["mentor_authed"] = False
                st.success("Password changed! Please log in again with the new password.")
                st.rerun()


# =====================================================================
# Main
# =====================================================================

def main():
    inject_global_css()
    cookie_manager = get_cookie_manager()
    sh.init_sheets()

    query_page = st.query_params.get("page")
    if query_page == "mentor" and "page" not in st.session_state:
        st.session_state["page"] = "mentor"

    try_auto_login(cookie_manager)

    if st.session_state.get("page") == "mentor":
        page_mentor()
        return

    if "user" not in st.session_state:
        render_auth_page(cookie_manager)
        return

    render_navbar()
    page = st.session_state.get("page", "home")

    if page == "home":
        page_home()
    elif page == "mytests":
        page_my_tests()
    elif page == "analysis":
        page_analysis()
    elif page == "leaderboard":
        page_leaderboard()
    elif page == "profile":
        page_profile(cookie_manager)
    else:
        page_home()


if __name__ == "__main__":
    main()
