"""
app.py
------
OMR Result App - main Streamlit application.

Pages:
- Login / Sign Up (single ID + password login; role is auto-detected)
- Student: Live Exam (submit OMR), Result Analysis (own history), Leaderboard
- Mentor: Answer Key, Calibration, All Answer Keys, Settings, Leaderboard

Run with: streamlit run app.py
"""

import json
from datetime import datetime, date, time as dtime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import auth
import omr_scanner
import sheets_helper as sh

st.set_page_config(page_title="OMR Result App", page_icon="📝", layout="centered")


# =====================================================================
# Login / Sign Up
# =====================================================================

def render_login_page():
    st.markdown(
        "<h1 style='text-align:center;'>📝 OMR Result App</h1>"
        "<p style='text-align:center; color:gray;'>Login করে শুরু করো</p>",
        unsafe_allow_html=True,
    )

    tab_login, tab_student, tab_mentor = st.tabs(
        ["🔑 Login", "🎓 Student Sign Up", "👨‍🏫 Mentor Sign Up"]
    )

    with tab_login:
        with st.form("login_form"):
            uid = st.text_input("User ID")
            pw = st.text_input("Password", type="password")
            remember = st.checkbox("এই ডিভাইসে মনে রাখো", value=True)
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
        if submitted:
            user = auth.login(uid, pw)
            if user:
                auth.start_session(user, remember=remember)
                st.rerun()
            else:
                st.error("ID অথবা Password ভুল।")

    with tab_student:
        st.caption("একবার Account বানালেই পরে শুধু Login করলেই চলবে - নাম আর বার বার লিখতে হবে না।")
        with st.form("student_signup_form"):
            s_id = st.text_input("Unique ID / Roll Number")
            s_name = st.text_input("তোমার নাম")
            s_pw = st.text_input("Password", type="password", key="s_pw")
            s_pw2 = st.text_input("Password আবার লিখো", type="password", key="s_pw2")
            s_submit = st.form_submit_button("Account তৈরি করো", use_container_width=True)
        if s_submit:
            if s_pw != s_pw2:
                st.error("দুইটা Password মিলছে না।")
            else:
                ok, msg = auth.create_account(s_id, s_name, s_pw, role="student")
                if ok:
                    st.success("✅ Account তৈরি হয়েছে! এখন 'Login' ট্যাব থেকে Login করো।")
                else:
                    st.error(msg)

    with tab_mentor:
        st.caption("Mentor Account খুলতে একটা Invite Code লাগবে (এটা তোমার admin/আগের mentor দিতে পারবে)।")
        with st.form("mentor_signup_form"):
            m_id = st.text_input("Unique ID")
            m_name = st.text_input("নাম")
            m_pw = st.text_input("Password", type="password", key="m_pw")
            m_pw2 = st.text_input("Password আবার লিখো", type="password", key="m_pw2")
            m_invite = st.text_input("Mentor Invite Code", type="password")
            m_submit = st.form_submit_button("Mentor Account তৈরি করো", use_container_width=True)
        if m_submit:
            if m_pw != m_pw2:
                st.error("দুইটা Password মিলছে না।")
            elif not m_invite or m_invite != sh.get_mentor_password():
                st.error("Invite Code ভুল।")
            else:
                ok, msg = auth.create_account(m_id, m_name, m_pw, role="mentor")
                if ok:
                    st.success("✅ Mentor Account তৈরি হয়েছে! এখন 'Login' ট্যাব থেকে Login করো।")
                else:
                    st.error(msg)


# =====================================================================
# Mentor: Answer Key tab (native bubble-grid input)
# =====================================================================

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
        st.caption("ঘন্টা (Hour)")
        hour = st.selectbox("Hour", list(range(1, 13)), index=default_hour_12 - 1,
                             key=f"{key_prefix}_hour", label_visibility="collapsed")
    with c2:
        st.caption("মিনিট (Min)")
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

    st.markdown("#### ① কতগুলো MCQ থাকবে? (Exam Style)")
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
    st.markdown("#### ② Exam Details")
    exam_name = st.text_input("Exam name", placeholder="e.g. Physics Model Test - 3")
    d = st.date_input("Exam date", value=date.today())
    st.markdown("**Start time**")
    start_t = _time_input_12h("mentor_start_t", default_hour_24=9, default_minute=0)
    st.markdown("**End time**")
    end_t = _time_input_12h("mentor_end_t", default_hour_24=9, default_minute=30)

    st.divider()
    st.markdown("#### ➖ Negative Marking (Optional)")
    negative_marking = st.checkbox(
        "এই exam এ negative marking রাখবেন? (ভুল উত্তরে মার্ক কাটা যাবে, blank/skip এ কাটা যাবে না)",
        key="mentor_neg_marking",
    )
    negative_value = 0.0
    if negative_marking:
        negative_value = st.number_input(
            "প্রতিটি ভুল উত্তরে কত মার্ক কাটা হবে?",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.2f",
            key="mentor_neg_value",
        )
        st.caption(f"উদাহরণ: {total_q} এর মধ্যে ৪টা ভুল হলে মার্ক কাটা যাবে {4 * negative_value:.2f}")

    st.divider()
    answered = _count_answered(total_q)
    st.markdown(f"#### ③ ✏️ Fill the Answer Key ({answered}/{total_q} answered)")
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
            key_id = sh.add_answer_key(exam_name.strip(), d.strftime("%Y-%m-%d"), start_str, end_str,
                                        total_q, answer_string,
                                        negative_marking=negative_marking,
                                        negative_marks_value=negative_value)
            for q in range(1, total_q + 1):
                st.session_state.pop(_answer_key(q), None)
            st.success(f"✅ Answer key for '{exam_name}' saved! Key ID: {key_id}")


# =====================================================================
# Mentor Panel
# =====================================================================

def page_mentor():
    st.header("👨‍🏫 Mentor Panel")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Answer Key", "🎯 Calibration", "📋 All Answer Keys", "⚙️ Settings"]
    )

    with tab1:
        render_answer_key_tab()

    with tab2:
        st.subheader("🎯 OMR Sheet Calibration (only needed once per sheet size)")
        calib_style = st.radio(
            "কোন সাইজের শিট ক্যালিব্রেট করবে?", ["100 Questions", "40 Questions"],
            horizontal=True, key="calib_style_choice",
        )
        total_q_calib = 100 if "100" in calib_style else 40

        existing_calibration = sh.load_calibration(total_q_calib)
        force_key = f"force_recalibrate_{total_q_calib}"

        if existing_calibration and not st.session_state.get(force_key):
            st.success(f"✅ {total_q_calib}-question শিটের Calibration আগে থেকেই সেভ করা আছে।")
            with st.expander("View the currently active calibration"):
                st.json(existing_calibration)
            st.caption("Students can submit OMR sheets normally. You don't need to visit this page again unless the sheet design changes.")
            if st.button("🔄 Recalibrate", key=f"recal_btn_{total_q_calib}"):
                st.session_state[force_key] = True
                st.session_state["calib_points"] = []
                st.rerun()
        else:
            if existing_calibration:
                st.info("You're creating a new calibration - the old one will be replaced when you save.")
                if st.button("❌ Go Back to the Previous Calibration", key=f"back_btn_{total_q_calib}"):
                    st.session_state[force_key] = False
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
            uploaded = st.file_uploader("Upload blank OMR sheet", type=["png", "jpg", "jpeg"],
                                         key=f"calib_upload_{total_q_calib}")
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
                        if st.button("🔄 Start Over", key=f"startover_{total_q_calib}"):
                            st.session_state["calib_points"] = []
                            st.rerun()
                    with col2:
                        if st.button("💾 Save Calibration", type="primary", key=f"savecal_{total_q_calib}"):
                            calibration = {
                                "q1_a": pts[0], "q1_d": pts[1], "q25_a": pts[2], "q26_a": pts[3],
                            }
                            sh.save_calibration(calibration, total_q_calib)
                            st.success("Calibration saved! Students can now upload OMR sheets.")
                            st.session_state["calib_points"] = []
                            st.session_state[force_key] = False

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
            display_df = display_df.rename(columns={
                "negative_marking": "Negative Marking", "negative_marks_value": "Per Wrong",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("⚙️ Settings")

        st.markdown("##### 🔑 আমার Password পরিবর্তন করো")
        with st.form("mentor_change_own_pw"):
            cur_pw = st.text_input("Current password", type="password")
            new_pw1 = st.text_input("New password", type="password")
            new_pw2 = st.text_input("Re-enter new password", type="password")
            submit_pw = st.form_submit_button("✅ Update Password", type="primary")
        if submit_pw:
            me = sh.get_user(st.session_state["user_id"])
            if not me or not auth.verify_password(cur_pw, me.get("password_hash", "")):
                st.error("Current password is incorrect.")
            elif not new_pw1:
                st.error("New password cannot be empty.")
            elif new_pw1 != new_pw2:
                st.error("The two new password entries don't match.")
            else:
                sh.update_user_password(st.session_state["user_id"], auth.hash_password(new_pw1))
                st.success("Password changed!")

        st.divider()
        st.markdown("##### ✉️ Mentor Invite Code")
        st.caption("নতুন Mentor Account খুলতে যে Code লাগবে, সেটা এখান থেকে বদলাতে পারো।")
        with st.form("mentor_invite_code_form"):
            new_invite = st.text_input("নতুন Invite Code", type="password")
            submit_invite = st.form_submit_button("✅ Update Invite Code", type="primary")
        if submit_invite:
            if not new_invite:
                st.error("Invite code খালি রাখা যাবে না।")
            else:
                sh.set_mentor_password(new_invite)
                st.success("Invite code পরিবর্তন হয়েছে!")


# =====================================================================
# Student: Live Exam
# =====================================================================

def page_student_live_exam():
    name = st.session_state["name"]
    st.header(f"🎓 Live Exam")

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

    calibration = None
    if active:
        calibration = sh.load_calibration(active["total_questions"])
        if not calibration:
            st.error(f"মেন্টর এখনো {active['total_questions']}-প্রশ্নের শিট calibrate করেননি। মেন্টরকে জানাও।")

    uploaded = st.file_uploader("Upload a photo of your filled OMR sheet", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded photo", use_container_width=True)

    disabled = not active or not calibration
    if st.button("📤 Submit & See Score", type="primary", disabled=disabled or not uploaded):
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
                        {"Question": f"Q{q}", "Your Answer": result["wrong_details"][q]["given"],
                         "Correct Answer": result["wrong_details"][q]["correct"]}
                        for q in result["wrong"]
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("Question numbers you got wrong:", ", ".join(str(w) for w in result["wrong"]))
                    st.caption("Correct answers will be shown once the exam time window closes.")
        elif result["answered"] > 0:
            st.success("🎉 No wrong answers!")


# =====================================================================
# Student: Result Analysis
# =====================================================================

def page_result_analysis():
    name = st.session_state["name"]
    st.header("📊 আমার Result Analysis")

    mine = sh.get_results_for_student(name)
    if mine.empty:
        st.info("তুমি এখনো কোনো exam জমা দাওনি। Live Exam থেকে OMR শিট জমা দিলে এখানে report দেখতে পাবে।")
        return

    keys_df = sh.get_all_answer_keys()

    total_marks = mine["marks"].sum()
    total_possible = mine["total"].sum()
    avg_pct = round((total_marks / total_possible) * 100, 2) if total_possible else 0.0

    st.markdown("#### 📈 সার্বিক ফলাফল (Overall)")
    c1, c2, c3 = st.columns(3)
    c1.metric("মোট Exam দিয়েছো", mine["key_id"].nunique())
    c2.metric("Average %", f"{avg_pct}%")
    c3.metric("মোট Marks", round(total_marks, 2))

    st.divider()
    st.markdown("#### 🗂️ প্রতিটি Exam এর রিপোর্ট")

    for _, row in mine.iterrows():
        key_row = keys_df[keys_df["key_id"] == row["key_id"]] if not keys_df.empty else pd.DataFrame()
        exam_name = key_row["exam_name"].values[0] if not key_row.empty and key_row["exam_name"].values[0] else row["key_id"]

        with st.expander(f"📝 {exam_name}  —  {row.get('timestamp', '')}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Marks", row["marks"])
            c2.metric("Correct ✅", row["correct"])
            c3.metric("Wrong ❌", row["wrong_count"])
            c4, c5 = st.columns(2)
            c4.metric("Answered", row["answered"])
            c5.metric("Accuracy", f"{row['accuracy']}%")

            wrong_q = [int(q) for q in str(row.get("wrong", "")).split(",") if str(q).strip().isdigit()]

            if not wrong_q:
                if row["answered"] > 0:
                    st.success("🎉 এই exam এ কোনো ভুল হয়নি!")
                continue

            window_closed = True
            end_dt = None
            if not key_row.empty:
                try:
                    end_dt = datetime.strptime(str(key_row["end_time"].values[0]), "%Y-%m-%d %H:%M")
                    window_closed = sh.now_bd() > end_dt
                except Exception:
                    window_closed = True

            if not window_closed:
                st.caption("⏳ এই exam এর সময় এখনো শেষ হয়নি, তাই সঠিক উত্তর এখন দেখানো যাচ্ছে না।")
                st.write("ভুল হওয়া প্রশ্ন:", ", ".join(str(q) for q in wrong_q))
                continue

            st.markdown("**❌ ভুল প্রশ্ন ও সঠিক উত্তর (Solution):**")
            details = {}
            raw_details = row.get("wrong_details", "")
            if raw_details:
                try:
                    details = {int(k): v for k, v in json.loads(raw_details).items()}
                except Exception:
                    details = {}

            if details:
                rows = [
                    {"Question": f"Q{q}", "তোমার উত্তর": details.get(q, {}).get("given", "-"),
                     "সঠিক উত্তর": details.get(q, {}).get("correct", "-")}
                    for q in wrong_q
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            elif not key_row.empty:
                ans_str = str(key_row["answer_string"].values[0])
                rows = [
                    {"Question": f"Q{q}", "সঠিক উত্তর": ans_str[q - 1] if q - 1 < len(ans_str) else "-"}
                    for q in wrong_q
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =====================================================================
# Leaderboard (open to everyone, shows names only - never the login ID)
# =====================================================================

def page_leaderboard():
    st.header("🏆 Leaderboard")
    my_name = st.session_state.get("name")
    tab1, tab2 = st.tabs(["📅 Daily Exam Leaderboard", "📊 Overall Analysis"])

    with tab1:
        keys_df = sh.get_all_answer_keys()
        if keys_df.empty:
            st.info("No exam/answer key has been set yet.")
        else:
            keys_df = keys_df.iloc[::-1].reset_index(drop=True)
            options = {}
            for _, row in keys_df.iterrows():
                label = f"{row.get('exam_name') or row['key_id']} | {row['date']} | {str(row['start_time'])[-5:]}-{str(row['end_time'])[-5:]}"
                options[label] = row["key_id"]
            choice = st.selectbox("Choose which exam's results to view", list(options.keys()))
            if st.button("🔄 Refresh", key="refresh_daily"):
                st.cache_data.clear()
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
                st.dataframe(
                    show_df, use_container_width=True, hide_index=True,
                    column_config={"Student": st.column_config.TextColumn("Student")},
                )

    with tab2:
        st.caption("Ranking by average percentage across all exams combined.")
        if st.button("🔄 Refresh", key="refresh_overall"):
            st.cache_data.clear()
            st.rerun()
        df = sh.get_overall_leaderboard()
        if df.empty:
            st.info("No results have been submitted yet.")
        else:
            show_df = df[["rank", "student", "avg_percent", "exams_taken", "total_marks", "total_possible"]].copy()
            show_df.columns = ["Rank", "Student", "Average %", "Exams Taken", "Total Marks", "Total Possible"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)


# =====================================================================
# Main
# =====================================================================

def main():
    sh.init_sheets()
    auth.try_auto_login()

    if not st.session_state.get("authed"):
        render_login_page()
        return

    role = st.session_state["role"]
    name = st.session_state["name"]

    st.sidebar.markdown("## 📝 OMR Result App")
    st.sidebar.success(f"👋 স্বাগতম, **{name}**")
    st.sidebar.caption("🎓 Student" if role == "student" else "👨‍🏫 Mentor")
    st.sidebar.divider()

    if role == "mentor":
        pages = ["🏆 Leaderboard", "👨‍🏫 Mentor Panel"]
    else:
        pages = ["🟢 Live Exam", "📊 Result Analysis", "🏆 Leaderboard"]

    page = st.sidebar.radio("Menu", pages, label_visibility="collapsed")

    st.sidebar.divider()
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth.logout()
        st.rerun()

    if page == "🟢 Live Exam":
        page_student_live_exam()
    elif page == "📊 Result Analysis":
        page_result_analysis()
    elif page == "🏆 Leaderboard":
        page_leaderboard()
    elif page == "👨‍🏫 Mentor Panel":
        page_mentor()


if __name__ == "__main__":
    main()
