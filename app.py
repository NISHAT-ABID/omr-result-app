"""
app.py
------
OMR Result App - main Streamlit application.

Page gula:
  - Login (shared password diye private rakha)
  - Mentor: Answer key + exam time set kora, calibration, leaderboard dekha
  - Student: OMR sheet upload kore result dekha
  - Leaderboard: shobar jonno khola, live rank dekhay

Run korte: streamlit run app.py
"""

from datetime import datetime, date, time as dtime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import omr_scanner
import sheets_helper as sh

st.set_page_config(page_title="OMR Result App", page_icon="📝", layout="centered")

STUDENTS = ["Student 1", "Student 2", "Student 3"]  # eikhane tomader 3 jon er naam boshao


# ---------------- Auth ----------------

def check_password():
    """Simple shared password diye app private rakha hocche."""
    if st.session_state.get("authed"):
        return True

    st.title("🔒 OMR Result App")
    pw = st.text_input("Password dao", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Password vul hoyeche")
    return False


def is_mentor():
    if st.session_state.get("mentor_authed"):
        return True
    pw = st.text_input("Mentor password", type="password", key="mentor_pw")
    if st.button("Mentor Login"):
        if pw == st.secrets.get("MENTOR_PASSWORD", ""):
            st.session_state["mentor_authed"] = True
            st.rerun()
        else:
            st.error("Mentor password vul hoyeche")
    return False


# ---------------- Pages ----------------

def page_mentor():
    st.header("👨‍🏫 Mentor Panel")
    if not is_mentor():
        return

    tab1, tab2, tab3 = st.tabs(["Answer Key Set kora", "Calibration", "Sob Answer Key"])

    with tab1:
        st.subheader("Ajker Answer Key o Exam Time")
        d = st.date_input("Exam er tarikh", value=date.today())
        col1, col2 = st.columns(2)
        with col1:
            start_t = st.time_input("Shuru shomoy", value=dtime(9, 0))
        with col2:
            end_t = st.time_input("Shesh shomoy", value=dtime(9, 30))

        answer_string = st.text_input(
            "100 tar shothik uttor likho (jemon: ABCDABCD... - total 100 character, kono space chara)",
            max_chars=100,
        )
        st.caption(f"Character gonona: {len(answer_string)}/100")

        if st.button("✅ Answer Key Save Koro", type="primary"):
            cleaned = answer_string.strip().upper().replace(" ", "")
            if len(cleaned) != 100 or any(c not in "ABCD" for c in cleaned):
                st.error("Thik 100 ta A/B/C/D character dite hobe, kono space/onno character chara.")
            else:
                start_str = f"{d.strftime('%Y-%m-%d')} {start_t.strftime('%H:%M')}"
                end_str = f"{d.strftime('%Y-%m-%d')} {end_t.strftime('%H:%M')}"
                key_id = sh.add_answer_key(d.strftime("%Y-%m-%d"), start_str, end_str, cleaned)
                st.success(f"Answer key save hoyeche! Key ID: {key_id}")

    with tab2:
        st.subheader("🎯 OMR Sheet Calibration (ekbar korle hoy)")
        st.markdown(
            """
Ekta **blank OMR sheet** er sojo/straight chobi upload koro, tarpor niche
chobir upor click kore 4 ta point dekhao (order onujayi):

1. Question **1** - option **A** er bubble er kendro
2. Question **1** - option **D** er bubble er kendro
3. Question **25** - option **A** er bubble er kendro
4. Question **26** - option **A** er bubble er kendro
            """
        )
        uploaded = st.file_uploader("Blank OMR sheet upload koro", type=["png", "jpg", "jpeg"], key="calib_upload")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            warped, ok = omr_scanner.detect_and_warp(img_bgr)
            if not ok:
                st.warning("Sheet er 4 kona automatic khuje pawa jayni. Tarpor o niche click kore calibrate korte paro, kintu chobi ta shoja/flat tুলে abar try korle valo hobe.")

            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            warped_pil = Image.fromarray(warped_rgb)

            if "calib_points" not in st.session_state:
                st.session_state["calib_points"] = []

            labels = ["Q1-A", "Q1-D", "Q25-A", "Q26-A"]
            current_step = len(st.session_state["calib_points"])

            if current_step < 4:
                st.info(f"Ekhon click koro: **{labels[current_step]}**")
                coords = streamlit_image_coordinates(warped_pil, key="calib_img")
                if coords is not None:
                    pt = (coords["x"], coords["y"])
                    if not st.session_state["calib_points"] or st.session_state["calib_points"][-1] != pt:
                        st.session_state["calib_points"].append(pt)
                        st.rerun()
            else:
                st.success("4 ta point e click kora hoyeche!")
                pts = st.session_state["calib_points"]
                for lbl, pt in zip(labels, pts):
                    st.write(f"- {lbl}: {pt}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Abar Shuru Koro"):
                        st.session_state["calib_points"] = []
                        st.rerun()
                with col2:
                    if st.button("💾 Calibration Save Koro", type="primary"):
                        calibration = {
                            "q1_a": pts[0],
                            "q1_d": pts[1],
                            "q25_a": pts[2],
                            "q26_a": pts[3],
                        }
                        sh.save_calibration(calibration)
                        st.success("Calibration save hoye geche! Ekhon student ra OMR upload korte parbe.")
                        st.session_state["calib_points"] = []

    with tab3:
        st.subheader("Sob Answer Key")
        df = sh.get_all_answer_keys()
        if df.empty:
            st.info("Ekhono kono answer key dewa hoyni.")
        else:
            st.dataframe(df, use_container_width=True)


def page_student():
    st.header("🎓 Student - OMR Submit Koro")

    calibration = sh.load_calibration()
    if not calibration:
        st.error("Mentor ekhono OMR sheet calibrate koreni. Age mentor ke calibration korte bolo.")
        return

    name = st.selectbox("Tomar naam select koro", STUDENTS)
    uploaded = st.file_uploader("Vora OMR sheet er chobi upload koro", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Upload kora chobi", use_container_width=True)

        if st.button("📤 Submit & Score Dekho", type="primary"):
            with st.spinner("Check kora hocche..."):
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                warped, ok = omr_scanner.detect_and_warp(img_bgr)
                if not ok:
                    st.warning("Sheet er kona thik moto khuje pawa jayni, tobuo try kora hocche. Result vul hote pare - chobi ta aro shoja tুলে abar try koro jodi result thik na lage.")

                grid = omr_scanner.build_grid(calibration)
                student_answers = omr_scanner.read_answers(warped, grid)

                key_id, key_string = sh.get_active_answer_key()
                if not key_id:
                    st.error("Ekhon kono exam active nei (mentor er set kora time window er baire). Result dewa jabe na.")
                    return

                score, total, wrong = omr_scanner.score_answers(student_answers, key_string)
                sh.append_result(name, key_id, score, total, wrong)

                st.success(f"Result save hoyeche! Score: {score} / {total}")
                st.metric("Tomar Score", f"{score} / {total}")
                if wrong:
                    st.write("Vul howa question number gulo:", ", ".join(str(w) for w in wrong))


def page_leaderboard():
    st.header("🏆 Leaderboard")
    if st.button("🔄 Refresh"):
        st.rerun()

    df = sh.get_leaderboard()
    if df.empty:
        st.info("Ekhono kono result submit hoyni.")
        return

    st.dataframe(
        df[["rank", "student", "score", "total", "timestamp"]],
        use_container_width=True,
        hide_index=True,
    )


# ---------------- Main ----------------

def main():
    if not check_password():
        return

    sh.init_sheets()

    st.sidebar.title("📝 OMR Result App")
    page = st.sidebar.radio("Menu", ["Student - OMR Submit", "Leaderboard", "Mentor Panel"])

    if page == "Student - OMR Submit":
        page_student()
    elif page == "Leaderboard":
        page_leaderboard()
    elif page == "Mentor Panel":
        page_mentor()


if __name__ == "__main__":
    main()
