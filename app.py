"""
app.py
------
OMR Result App - main Streamlit application.

Pages:
  - Login (shared password to keep the app private)
  - Mentor: Set answer key (visual click) + exam time, calibration,
            question PDF upload, password change
  - Student: Enter your name and upload an OMR sheet to see your result
  - Leaderboard: Daily + Overall analysis - open to everyone

Run with: streamlit run app.py
"""

from datetime import datetime, date, time as dtime, timedelta

import cv2
import numpy as np
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


# ---------------- Mentor: Answer Key tab (visual click input) ----------------

def render_answer_key_tab():
    st.subheader("🗓️ Set Today's Answer Key & Exam Time")

    exam_name = st.text_input("Exam name", placeholder="e.g. Physics Model Test - 3")

    d = st.date_input("Exam date", value=date.today())
    col1, col2 = st.columns(2)
    with col1:
        start_t = st.time_input("Start time", value=dtime(9, 0))
    with col2:
        end_t = st.time_input("End time", value=dtime(9, 30))

    exam_style = st.radio(
        "Exam Style",
        ["100 Marks (Question 1-100)", "40 Marks (Question 1-40)"],
        horizontal=True,
    )
    total_q = 100 if "100" in exam_style else 40

    pdf_file = st.file_uploader(
        "Question PDF (optional - if provided, students can view it in the app)",
        type=["pdf"],
        key="question_pdf",
    )

    st.divider()

    calibration = sh.load_calibration()
    if not calibration:
        st.warning("⚠️ Calibrate the OMR sheet first under Mentor Panel > Calibration, then you can fill in answers here by clicking bubbles. For now, you can enter answers as text below.")
        _render_text_fallback(exam_name, d, start_t, end_t, total_q, pdf_file)
        return

    grid = omr_scanner.build_grid(calibration)

    # session state setup / reset if exam style changed
    if st.session_state.get("mentor_answer_total_q") != total_q:
        st.session_state["mentor_answer_map"] = {}
        st.session_state["mentor_answer_total_q"] = total_q
        st.session_state["mentor_last_click"] = None

    answer_map = st.session_state.setdefault("mentor_answer_map", {})

    st.markdown(f"### ✏️ Fill the Answer Key - Click the Bubbles ({len(answer_map)}/{total_q} answered)")
    st.progress(len(answer_map) / total_q if total_q else 0)

    img = omr_scanner.render_sheet_image(grid, total_questions=total_q, answers=answer_map)
    coords = streamlit_image_coordinates(img, key="answer_click_img")

    if coords is not None:
        pt = (coords["x"], coords["y"])
        if st.session_state.get("mentor_last_click") != pt:
            st.session_state["mentor_last_click"] = pt
            hit = omr_scanner.find_clicked_bubble(grid, total_q, pt[0], pt[1])
            if hit:
                q, opt = hit
                if answer_map.get(q) == opt:
                    del answer_map[q]
                else:
                    answer_map[q] = opt
                st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear All"):
            st.session_state["mentor_answer_map"] = {}
            st.rerun()
    with col_b:
        with st.popover("⌨️ Fill Quickly with Text"):
            text_val = st.text_input(f"{total_q} characters (A/B/C/D), no spaces", key="quick_text_ans")
            if st.button("Apply Text"):
                cleaned = text_val.strip().upper().replace(" ", "")
                if len(cleaned) != total_q or any(c not in "ABCD" for c in cleaned):
                    st.error(f"You must enter exactly {total_q} A/B/C/D characters.")
                else:
                    st.session_state["mentor_answer_map"] = {i + 1: c for i, c in enumerate(cleaned)}
                    st.rerun()

    st.divider()

    if st.button("✅ Save Answer Key", type="primary", use_container_width=True):
        if not exam_name.strip():
            st.error("Please enter an exam name.")
        elif len(answer_map) != total_q:
            st.error(f"You must answer all {total_q} questions (currently {len(answer_map)} answered).")
        else:
            answer_string = "".join(answer_map[i] for i in range(1, total_q + 1))
            pdf_url = ""
            if pdf_file is not None:
                with st.spinner("Uploading question PDF..."):
                    try:
                        pdf_url = sh.upload_pdf_to_drive(pdf_file.getvalue(), pdf_file.name)
                    except Exception as e:
                        st.warning(f"Could not upload the PDF (answer key was still saved): {e}")

            start_str = f"{d.strftime('%Y-%m-%d')} {start_t.strftime('%H:%M')}"
            end_str = f"{d.strftime('%Y-%m-%d')} {end_t.strftime('%H:%M')}"
            key_id = sh.add_answer_key(exam_name.strip(), d.strftime("%Y-%m-%d"), start_str, end_str,
                                        total_q, answer_string, pdf_url)
            st.session_state["mentor_answer_map"] = {}
            st.session_state["mentor_last_click"] = None
            st.success(f"✅ Answer key for '{exam_name}' saved! Key ID: {key_id}")


def _render_text_fallback(exam_name, d, start_t, end_t, total_q, pdf_file):
    answer_string = st.text_input(f"Enter the {total_q} correct answers (e.g. ABCDABCD...)", max_chars=total_q)
    st.caption(f"Character count: {len(answer_string)}/{total_q}")

    if st.button("✅ Save Answer Key (Text)", type="primary"):
        cleaned = answer_string.strip().upper().replace(" ", "")
        if not exam_name.strip():
            st.error("Please enter an exam name.")
        elif len(cleaned) != total_q or any(c not in "ABCD" for c in cleaned):
            st.error(f"You must enter exactly {total_q} A/B/C/D characters, no spaces or other characters.")
        else:
            pdf_url = ""
            if pdf_file is not None:
                with st.spinner("Uploading question PDF..."):
                    try:
                        pdf_url = sh.upload_pdf_to_drive(pdf_file.getvalue(), pdf_file.name)
                    except Exception as e:
                        st.warning(f"Could not upload the PDF (answer key was still saved): {e}")
            start_str = f"{d.strftime('%Y-%m-%d')} {start_t.strftime('%H:%M')}"
            end_str = f"{d.strftime('%Y-%m-%d')} {end_t.strftime('%H:%M')}"
            key_id = sh.add_answer_key(exam_name.strip(), d.strftime("%Y-%m-%d"), start_str, end_str,
                                        total_q, cleaned, pdf_url)
            st.success(f"✅ Answer key for '{exam_name}' saved! Key ID: {key_id}")


# ---------------- Mentor Panel ----------------

def page_mentor():
    st.header("👨‍🏫 Mentor Panel")
    if not is_mentor():
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Answer Key", "🎯 Calibration", "📋 All Answer Keys", "🔑 Password"]
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
            show_cols = ["key_id", "exam_name", "date", "start_time", "end_time", "total_questions"]
            show_cols = [c for c in show_cols if c in df.columns]
            display_df = df[show_cols].iloc[::-1].reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if "question_pdf_url" in df.columns:
                pdfs = df[df["question_pdf_url"].astype(str).str.strip() != ""]
                if not pdfs.empty:
                    st.caption("Question PDFs:")
                    for _, row in pdfs.iloc[::-1].iterrows():
                        st.markdown(f"- **{row.get('exam_name', row['key_id'])}**: [View PDF]({row['question_pdf_url']})")

    with tab4:
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

    calibration = sh.load_calibration()
    if not calibration:
        st.error("The mentor hasn't calibrated the OMR sheet yet. Please ask the mentor to complete calibration first.")
        return

    name = st.text_input("Enter your name", placeholder="e.g. Rahim Ahmed")

    active = sh.get_active_answer_key()

    if active:
        remaining = active["end_dt"] - sh.now_bd()
        mins_left = max(0, int(remaining.total_seconds() // 60))
        with st.container(border=True):
            st.markdown(f"### 🟢 Active Exam: {active['exam_name'] or active['key_id']}")
            c1, c2 = st.columns(2)
            c1.metric("Total Questions/Marks", active["total_questions"])
            c2.metric("Time Remaining", f"{mins_left} min")
            if active.get("question_pdf_url"):
                st.markdown(f"📄 [View Question PDF]({active['question_pdf_url']})")
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

        disabled = not active or not name.strip()
        if not name.strip():
            st.caption("⚠️ Please enter your name first.")

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

                score, total, wrong = omr_scanner.score_answers(student_answers, key_string)
                sh.append_result(name.strip(), key_id, score, total, wrong)

                st.success(f"Result saved! Score: {score} / {total}")
                st.metric("Your Score", f"{score} / {total}")

                if wrong:
                    window_closed = end_dt is not None and sh.now_bd() > end_dt
                    if window_closed:
                        st.write("Questions you got wrong, with the correct answer:")
                        lines = [f"Q{q}: correct answer {key_string[q - 1]}" for q in wrong]
                        st.write(", ".join(lines))
                    else:
                        st.write("Question numbers you got wrong:", ", ".join(str(w) for w in wrong))
                        st.caption("Correct answers will be shown once the exam time window closes.")


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
                st.dataframe(
                    df[["rank", "student", "score", "total", "timestamp"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab2:
        st.caption("Ranking by average percentage across all exams combined.")
        if st.button("🔄 Refresh", key="refresh_overall"):
            st.rerun()

        df = sh.get_overall_leaderboard()
        if df.empty:
            st.info("No results have been submitted yet.")
        else:
            show_df = df[["rank", "student", "avg_percent", "exams_taken", "total_score", "total_possible"]].copy()
            show_df.columns = ["Rank", "Student", "Average %", "Exams Taken", "Total Score", "Total Possible"]
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
