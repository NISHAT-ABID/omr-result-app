"""
app.py
------
OMR Result App - main Streamlit application.

Pages:
- Login (shared password to keep the app private)
- Mentor: Set answer key (visual click) + exam time (12hr AM/PM),
  optional negative marking, calibration, password change,
  student list + password reset
- Student: Sign up / log in with name + password (no email/OTP), then
  upload an OMR sheet to see your result
- Leaderboard: Daily + Overall analysis - open to everyone

Run with: streamlit run app.py
"""

from datetime import datetime, date, time as dtime, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import omr_scanner
import sheets_helper as sh

st.set_page_config(page_title="OMR Result App", page_icon="📝", layout="centered")


# ---------------- Auth ----------------

def check_password():
    if st.session_state.get("authed"):
        return True

    st.markdown(
        "<h1 style='text-align:center;'>📝 OMR Result App</h1>"
        "<p style='text-align:center; color:gray;'>Enter the password to log in</p>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        pw = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
        if st.button("Login", use_container_width=True, type="primary"):
            if pw == st.secrets.get("APP_PASSWORD", ""):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


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


# ---------------- Student Auth (name + password, no email/OTP) ----------------
#
# Sign up = name + password -> account created instantly, unique ID handed
# back (STU0001, STU0002, ...). Returning students log in with ID + password.
# If a student forgets their password, only the mentor can reset it, from
# the Mentor Panel -> Students tab.

def student_login_gate():
    if st.session_state.get("student_authed"):
        return True

    st.markdown("### 🎓 Student লগইন")

    tab_login, tab_signup = st.tabs(["🔑 লগইন করুন", "🆕 নতুন অ্যাকাউন্ট"])

    with tab_login:
        sid = st.text_input("Student ID (যেমন STU0001)", key="login_sid")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", key="student_login_btn", type="primary"):
            if not sid.strip() or not pw:
                st.error("Student ID এবং Password দুটোই দিন।")
            else:
                student = sh.authenticate_student(sid, pw)
                if student:
                    st.session_state["student_authed"] = True
                    st.session_state["student_id"] = student["student_id"]
                    st.session_state["student_name"] = student["name"]
                    st.rerun()
                else:
                    st.error(
                        "ভুল Student ID অথবা Password। পাসওয়ার্ড ভুলে গেলে "
                        "মেন্টরকে বলুন - উনি Mentor Panel থেকে রিসেট করে দিতে পারবেন।"
                    )

    with tab_signup:
        if st.session_state.get("pending_new_student_id"):
            new_id = st.session_state["pending_new_student_id"]
            st.success(f"🎉 আপনার Account তৈরি হয়ে গেছে! আপনার Student ID: **{new_id}**")
            st.info("এই ID টা লিখে/মনে রাখুন - পরের বার লগইন করতে এই ID আর পাসওয়ার্ড লাগবে।")
            if st.button("➡️ Continue", key="student_signup_continue", type="primary"):
                st.session_state["student_authed"] = True
                st.session_state["student_id"] = st.session_state.pop("pending_new_student_id")
                st.session_state["student_name"] = st.session_state.pop("pending_new_student_name")
                st.rerun()
        else:
            name = st.text_input("আপনার নাম", key="signup_name", placeholder="e.g. Rahim Ahmed")
            pw1 = st.text_input("একটি Password দিন", type="password", key="signup_pw1")
            pw2 = st.text_input("Password আবার লিখুন", type="password", key="signup_pw2")
            if st.button("✅ Account তৈরি করুন", key="student_signup_btn", type="primary"):
                if not name.strip():
                    st.error("নাম দিন।")
                elif not pw1:
                    st.error("Password দিন।")
                elif pw1 != pw2:
                    st.error("দুটো Password মিলছে না।")
                else:
                    new_id = sh.create_student(name.strip(), pw1)
                    st.session_state["pending_new_student_id"] = new_id
                    st.session_state["pending_new_student_name"] = name.strip()
                    st.rerun()

    return False


# ---------------- Mentor: Answer Key tab (native bubble-grid input) ----------------
#
# This does NOT depend on any sheet image or calibration - it's a plain,
# self-contained UI. Step 1 is always "how many MCQs" (40 or 100), then a
# clean bubble grid (A/B/C/D radio per question) to fill the key.
#
# Layout:
#   - 40 questions -> 20 + 20 in 2 columns, shown all at once.
#   - 100 questions -> 25 + 25 in 2 columns per "page"; first 50 questions
#     shown first, then a Next button reveals the last 50 (25 + 25).

def _inject_bubble_grid_css():
    st.markdown(
        """
        <style>
        .st-key-answer_bubble_grid div[data-testid="stRadio"] {
            margin-bottom: -14px;
        }
        .st-key-answer_bubble_grid div[data-testid="stRadio"] > label {
            display: none;
        }
        .st-key-answer_bubble_grid div[role="radiogroup"] {
            gap: 6px;
        }
        .st-key-answer_bubble_grid div[role="radiogroup"] label {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 999px;
            padding: 2px 10px 2px 6px;
            margin-right: 0 !important;
        }
        .q-num-badge {
            display: inline-block;
            min-width: 28px;
            font-weight: 600;
            color: var(--text-color, inherit);
            opacity: 0.75;
            padding-top: 6px;
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
                f"Q{q}",
                options=["A", "B", "C", "D"],
                index=None,
                horizontal=True,
                key=_answer_key(q),
                label_visibility="collapsed",
            )


def _render_two_col_blocks(q_start, q_mid_end, q_mid_start, q_end):
    """Renders questions q_start..q_mid_end in the left column and
    q_mid_start..q_end in the right column."""
    col1, col2 = st.columns(2)
    with col1:
        _render_bubble_block(q_start, q_mid_end)
    with col2:
        _render_bubble_block(q_mid_start, q_end)


def _time_input_12h(key_prefix, default_hour_24=9, default_minute=0):
    """
    A 12-hour Hour / Minute / AM-PM picker (Streamlit's built-in time_input
    doesn't give control over 12h vs 24h display). Returns a datetime.time.
    """
    default_period = "PM" if default_hour_24 >= 12 else "AM"
    default_hour_12 = default_hour_24 % 12
    if default_hour_12 == 0:
        default_hour_12 = 12

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.caption("ঘন্টা (Hour)")
        hour = st.selectbox(
            "Hour", list(range(1, 13)), index=default_hour_12 - 1,
            key=f"{key_prefix}_hour", label_visibility="collapsed",
        )
    with c2:
        st.caption("মিনিট (Min)")
        minute = st.selectbox(
            "Minute", [f"{m:02d}" for m in range(60)], index=default_minute,
            key=f"{key_prefix}_min", label_visibility="collapsed",
        )
    with c3:
        st.caption("AM/PM")
        period = st.selectbox(
            "AM/PM", ["AM", "PM"], index=0 if default_period == "AM" else 1,
            key=f"{key_prefix}_period", label_visibility="collapsed",
        )

    hour_24 = hour % 12
    if period == "PM":
        hour_24 += 12
    return dtime(hour_24, int(minute))


def render_answer_key_tab():
    st.subheader("🗓️ Set Today's Answer Key & Exam Time")

    # ---- Step 1: how many MCQs (always asked first) ----
    st.markdown("#### ① কতগুলো MCQ থাকবে? (Exam Style)")
    exam_style = st.radio(
        "Exam Style",
        ["📄 100 Questions (Q1-100)", "📄 40 Questions (Q1-40)"],
        horizontal=True,
        label_visibility="collapsed",
        key="mentor_exam_style_choice",
    )
    total_q = 100 if "100" in exam_style else 40

    # reset the answer grid whenever the question-count changes
    if st.session_state.get("mentor_answer_total_q") != total_q:
        for q in range(1, 101):
            st.session_state.pop(_answer_key(q), None)
        st.session_state["mentor_answer_total_q"] = total_q
        st.session_state["mentor_answer_page"] = 1

    st.divider()

    # ---- Step 2: exam details ----
    st.markdown("#### ② Exam Details")
    exam_name = st.text_input("Exam name", placeholder="e.g. Physics Model Test - 3")
    d = st.date_input("Exam date", value=date.today())

    st.markdown("**Start time**")
    start_t = _time_input_12h("mentor_start_t", default_hour_24=9, default_minute=0)
    st.markdown("**End time**")
    end_t = _time_input_12h("mentor_end_t", default_hour_24=9, default_minute=30)

    st.divider()

    # ---- Negative marking (optional) ----
    st.markdown("#### ➖ Negative Marking (Optional)")
    negative_marking = st.checkbox(
        "এই exam এ negative marking রাখবেন? (ভুল উত্তরে মার্ক কাটা যাবে, blank/skip এ কাটা যাবে না)",
        key="mentor_neg_marking",
    )
    negative_value = 0.0
    if negative_marking:
        negative_value = st.number_input(
            "প্রতিটি ভুল উত্তরে কত মার্ক কাটা হবে? (যেমন medical admission exam এ সাধারণত 0.25)",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.2f",
            key="mentor_neg_value",
        )
        st.caption(f"উদাহরণ: {total_q} এর মধ্যে ৪টা ভুল হলে মার্ক কাটা যাবে {4 * negative_value:.2f}")

    st.divider()

    # ---- Step 3: fill answers (native bubble grid) ----
    answered = _count_answered(total_q)
    st.markdown(f"#### ③ ✏️ Fill the Answer Key ({answered}/{total_q} answered)")
    st.progress(answered / total_q if total_q else 0)

    tool_col1, tool_col2 = st.columns(2)
    with tool_col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            for q in range(1, total_q + 1):
                st.session_state.pop(_answer_key(q), None)
            st.session_state["mentor_answer_page"] = 1
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
        if total_q == 100:
            # Page 1: Q1-50 (25 + 25 in two columns). Page 2: Q51-100.
            page = st.session_state.get("mentor_answer_page", 1)

            if page == 1:
                st.caption("প্রশ্ন ১ - ৫০")
                _render_two_col_blocks(1, 25, 26, 50)
            else:
                st.caption("প্রশ্ন ৫১ - ১০০")
                _render_two_col_blocks(51, 75, 76, 100)

            nav_prev, nav_info, nav_next = st.columns([1, 1, 1])
            with nav_prev:
                if page == 2:
                    if st.button("⬅️ Previous 50", use_container_width=True):
                        st.session_state["mentor_answer_page"] = 1
                        st.rerun()
            with nav_info:
                st.caption(f"Page {page}/2")
            with nav_next:
                if page == 1:
                    if st.button("Next 50 ➡️", use_container_width=True, type="primary"):
                        st.session_state["mentor_answer_page"] = 2
                        st.rerun()
        else:
            # 40 questions - 20 + 20 in two columns, shown all at once.
            _render_two_col_blocks(1, 20, 21, 40)

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
            key_id = sh.add_answer_key(exam_name.strip(), d.strftime("%Y-%m-%d"), start_str, end_str,
                                        total_q, answer_string,
                                        negative_marking=negative_marking,
                                        negative_marks_value=negative_value)
            for q in range(1, total_q + 1):
                st.session_state.pop(_answer_key(q), None)
            st.session_state["mentor_answer_page"] = 1
            st.success(f"✅ Answer key for '{exam_name}' saved! Key ID: {key_id}")


# ---------------- Mentor Panel ----------------

def page_mentor():
    st.header("👨‍🏫 Mentor Panel")
    if not is_mentor():
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📝 Answer Key", "🎯 Calibration", "📋 All Answer Keys", "👥 Students", "🔑 Password"]
    )

    with tab1:
        render_answer_key_tab()

    with tab2:
        existing_calibration = sh.load_calibration()
        if existing_calibration and not st.session_state.get("force_recalibrate"):
            st.success("✅ Calibration is already saved - no need to redo it.")
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

            st.subheader("🎯 OMR Sheet Calibration (only needed once)")
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
                                "q1_a": pts[0],
                                "q1_d": pts[1],
                                "q25_a": pts[2],
                                "q26_a": pts[3],
                            }
                            sh.save_calibration(calibration)
                            st.success("Calibration saved! Students can now upload OMR sheets.")
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
            rename_map = {
                "negative_marking": "Negative Marking",
                "negative_marks_value": "Per Wrong",
            }
            display_df = display_df.rename(columns=rename_map)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("👥 Students")
        st.caption("এখানে সব student একাউন্ট দেখা যাবে, আর কেউ পাসওয়ার্ড ভুলে গেলে এখান থেকেই রিসেট করে দেওয়া যাবে - কোনো email/OTP লাগবে না।")
        students_df = sh.get_all_students()
        if students_df.empty:
            st.info("এখনো কোনো student সাইন আপ করেনি।")
        else:
            show_cols = [c for c in ["student_id", "name", "created_at"] if c in students_df.columns]
            st.dataframe(students_df[show_cols].iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### 🔁 Password Reset")
            options = {f"{row['student_id']} - {row['name']}": row["student_id"] for _, row in students_df.iterrows()}
            chosen_label = st.selectbox("কোন student এর password reset করবেন?", list(options.keys()), key="reset_student_choice")
            new_pw = st.text_input("নতুন Password", type="password", key="reset_student_pw")
            if st.button("🔁 Reset Password", key="reset_student_btn", type="primary"):
                if not new_pw:
                    st.error("নতুন password দিন।")
                else:
                    ok = sh.reset_student_password(options[chosen_label], new_pw)
                    if ok:
                        st.success(f"✅ {options[chosen_label]} এর password reset হয়ে গেছে। এখন এই নতুন password দিয়ে student লগইন করতে পারবে।")
                    else:
                        st.error("কিছু একটা সমস্যা হয়েছে, আবার চেষ্টা করুন।")

    with tab5:
        st.subheader("🔑 Change Mentor Password")
        st.caption("This password is for you (the mentor) only - no coding needed, you can change it right here.")
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


# ---------------- Student Panel ----------------

def page_student():
    st.header("🎓 Student - Submit OMR")

    if not student_login_gate():
        return

    student_name = st.session_state["student_name"]
    student_id = st.session_state["student_id"]

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.caption(f"👤 লগইন করা আছে: **{student_name}** ({student_id})")
    with top_r:
        if st.button("Logout", key="student_logout"):
            for k in ("student_authed", "student_id", "student_name"):
                st.session_state.pop(k, None)
            st.rerun()

    calibration = sh.load_calibration()
    if not calibration:
        st.error("The mentor hasn't calibrated the OMR sheet yet. Please ask the mentor to complete calibration first.")
        return

    active = sh.get_active_answer_key()
    if active:
        remaining = active["end_dt"] - sh.now_bd()
        mins_left = max(0, int(remaining.total_seconds() // 60))
        with st.container(border=True):
            st.markdown(f"### 🟢 Active Exam: {active['exam_name'] or active['key_id']}")
            c1, c2 = st.columns(2)
            c1.metric("Total Questions/Marks", active["total_questions"])
            c2.metric("Time Remaining", f"{mins_left} min")
            if active.get("negative_marking"):
                st.caption(f"⚠️ Negative marking is ON: -{active['negative_marks_value']} per wrong answer (skipped questions are not penalized).")
    else:
        upcoming = sh.get_upcoming_answer_key()
        if upcoming:
            st.warning(
                f"No exam is active right now. Next exam: **{upcoming['exam_name'] or upcoming['key_id']}** "
                f"starts at **{upcoming['start_dt'].strftime('%Y-%m-%d %H:%M')}**."
            )
        else:
            st.info("No exam is active or upcoming right now.")

    uploaded = st.file_uploader("Upload a photo of your filled OMR sheet", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded photo", use_container_width=True)

        disabled = not active
        if st.button("📤 Submit & See Score", type="primary", disabled=disabled):
            with st.spinner("Checking your answers..."):
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                warped, ok = omr_scanner.detect_and_warp(img_bgr)
                if not ok:
                    st.warning("Couldn't clearly detect the sheet's corners - still trying anyway. The result may be inaccurate; retake the photo straighter and try again if it looks wrong.")

                grid = omr_scanner.build_grid(calibration)
                student_answers = omr_scanner.read_answers(warped, grid)

                active_now = sh.get_active_answer_key()
                if not active_now:
                    st.error("No exam is active right now (outside the mentor's time window). The result cannot be recorded.")
                    return

                key_string = active_now["answer_string"]
                key_id = active_now["key_id"]
                end_dt = active_now["end_dt"]

                result = omr_scanner.score_answers(
                    student_answers, key_string,
                    negative_marking=active_now.get("negative_marking", False),
                    negative_value=active_now.get("negative_marks_value", 0.0),
                )

                sh.append_result(f"{student_name} ({student_id})", key_id, result)
                st.success("✅ Result saved!")

                # ---- Result summary ----
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

                # ---- Wrong-answers box ----
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


# ---------------- Leaderboard Panel ----------------

def page_leaderboard():
    st.header("🏆 Leaderboard")

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


# ---------------- Main ----------------

def main():
    if not check_password():
        return

    sh.init_sheets()

    st.sidebar.markdown("## 📝 OMR Result App")
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Menu",
        ["🎓 Student - Submit OMR", "🏆 Leaderboard", "👨‍🏫 Mentor Panel"],
        label_visibility="collapsed",
    )

    if page == "🎓 Student - Submit OMR":
        page_student()
    elif page == "🏆 Leaderboard":
        page_leaderboard()
    elif page == "👨‍🏫 Mentor Panel":
        page_mentor()


if __name__ == "__main__":
    main()
