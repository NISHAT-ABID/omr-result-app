"""
sheets_helper.py
-----------------
All functions for using Google Sheets as the app's database.

Worksheets (tabs) used inside one Google Sheet:

1. AnswerKeys -> key_id, exam_name, date, start_time, end_time,
                 total_questions, answer_string,
                 negative_marking, negative_marks_value

2. Config     -> config_key, config_value (generic key-value store,
                 used for calibration + mentor password)

3. Results    -> timestamp, student_id, student, key_id, total, answered,
                 skipped, correct, wrong_count, wrong, marks, accuracy,
                 negative_marking, negative_value, edited_by_mentor

4. Students   -> student_id, name, phone, password_hash, salt,
                 security_question, security_answer_hash, disabled,
                 session_version, created_at

The Sheet ID is stored as SHEET_ID in .streamlit/secrets.toml.
All worksheets are created automatically the first time the app runs.

IMPORTANT - value_input_option="RAW":
Every write below explicitly uses RAW input mode. Without this, Google
Sheets' default ("USER_ENTERED") auto-parses text as if a human typed it
in the UI - which silently turns a phone number like "01745678901" into
the number 1745678901 (dropping the leading zero), and can also mangle
date/time strings. RAW mode stores exactly the text we send, with no
reinterpretation, which is what this app relies on everywhere (phone
lookups, date/time parsing with strptime, etc.).
"""

import hashlib
import io
import json
import re
import secrets
import string
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

RAW = "RAW"  # value_input_option used for every write - see module docstring

ANSWERKEYS_HEADER = [
    "key_id", "exam_name", "date", "start_time", "end_time",
    "total_questions", "answer_string", "negative_marking", "negative_marks_value",
]

RESULTS_HEADER = [
    "timestamp", "student_id", "student", "key_id", "total", "answered", "skipped",
    "correct", "wrong_count", "wrong", "marks", "accuracy",
    "negative_marking", "negative_value", "edited_by_mentor",
    "wrong_details_json", "skipped_json",
]

CONFIG_HEADER = ["config_key", "config_value"]

STUDENTS_HEADER = [
    "student_id", "name", "phone", "password_hash", "salt",
    "security_question", "security_answer_hash", "disabled",
    "session_version", "created_at",
]

BD_TZ = ZoneInfo("Asia/Dhaka")


def now_bd():
    """
    Streamlit Cloud's server runs on UTC time, but exam times are set in
    Bangladesh time. Always use this function for "what time is it right
    now" so it stays correct regardless of where the server runs.
    """
    return datetime.now(BD_TZ).replace(tzinfo=None)


def _with_retry(func, *args, **kwargs):
    """Google Sheets API sometimes rate-limits (429); wait and retry."""
    delays = [1, 2, 4, 8]
    last_err = None
    for delay in delays:
        try:
            return func(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            last_err = e
            time.sleep(delay)
    return func(*args, **kwargs) if last_err is None else (_ for _ in ()).throw(last_err)


def _to_bool(val):
    return str(val).strip().upper() in ("TRUE", "1", "YES")


def _to_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _normalize_phone(phone):
    """Keeps only digits, preserves leading zeros. Used everywhere a phone
    number is stored or looked up, so signup/login always agree."""
    return re.sub(r"\D", "", str(phone or ""))


@st.cache_resource(show_spinner=False)
def get_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource(show_spinner=False)
def get_spreadsheet():
    client = get_client()
    return _with_retry(client.open_by_key, st.secrets["SHEET_ID"])


def _get_or_create_worksheet(sh, title, header):
    try:
        ws = _with_retry(sh.worksheet, title)
    except gspread.WorksheetNotFound:
        ws = _with_retry(sh.add_worksheet, title=title, rows=2000, cols=len(header) + 2)
        _with_retry(ws.append_row, header, value_input_option=RAW)
        return ws

    values = _with_retry(ws.get_all_values)
    if not values:
        _with_retry(ws.append_row, header, value_input_option=RAW)
    else:
        existing_header = values[0]
        if existing_header != header and len(header) > len(existing_header):
            # header grew (we added new columns in an update) -> extend it,
            # existing rows just get blank values in the new columns
            _with_retry(ws.update, "A1", [header], value_input_option=RAW)
    return ws


@st.cache_resource(show_spinner=False)
def _cached_worksheet(title):
    header_map = {
        "AnswerKeys": ANSWERKEYS_HEADER,
        "Config": CONFIG_HEADER,
        "Results": RESULTS_HEADER,
        "Students": STUDENTS_HEADER,
    }
    sh = get_spreadsheet()
    return _get_or_create_worksheet(sh, title, header_map[title])


def init_sheets():
    _cached_worksheet("AnswerKeys")
    _cached_worksheet("Config")
    _cached_worksheet("Results")
    _cached_worksheet("Students")
    return get_spreadsheet()


def clear_data_caches():
    """Call after any write so the next read gets fresh data."""
    for key in ("_all_answer_keys_cached", "_all_results_cached", "_all_students_cached"):
        st.session_state.pop(key, None)


# ================= Answer Keys =================

def add_answer_key(exam_name, date_str, start_time_str, end_time_str,
                    total_questions, answer_string,
                    negative_marking=False, negative_marks_value=0.0):
    ws = _cached_worksheet("AnswerKeys")
    existing = _with_retry(ws.get_all_records)
    key_id = f"K{len(existing) + 1:04d}"
    _with_retry(
        ws.append_row,
        [key_id, exam_name, date_str, start_time_str, end_time_str,
         total_questions, answer_string, negative_marking, negative_marks_value],
        value_input_option=RAW,
    )
    clear_data_caches()
    return key_id


def get_all_answer_keys():
    ws = _cached_worksheet("AnswerKeys")
    records = _with_retry(ws.get_all_records)
    return pd.DataFrame(records)


def get_answer_key_by_id(key_id):
    df = get_all_answer_keys()
    if df.empty:
        return None
    match = df[df["key_id"] == key_id]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "key_id": row["key_id"],
        "exam_name": row.get("exam_name", ""),
        "date": row.get("date", ""),
        "answer_string": row["answer_string"],
        "total_questions": _to_int(row.get("total_questions"), len(str(row["answer_string"]))),
        "negative_marking": _to_bool(row.get("negative_marking", False)),
        "negative_marks_value": _to_float(row.get("negative_marks_value"), 0.0),
    }


def get_active_answer_key(now=None):
    """
    Finds which answer key is active right now.
    Returns: dict {key_id, exam_name, answer_string, total_questions,
                   start_dt, end_dt, negative_marking, negative_marks_value}
    or None.
    """
    if now is None:
        now = now_bd()
    df = get_all_answer_keys()
    if df.empty:
        return None
    for _, row in df.iterrows():
        try:
            start_dt = datetime.strptime(str(row["start_time"]), "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(str(row["end_time"]), "%Y-%m-%d %H:%M")
        except Exception:
            continue
        if start_dt <= now <= end_dt:
            return {
                "key_id": row["key_id"],
                "exam_name": row.get("exam_name", ""),
                "answer_string": row["answer_string"],
                "total_questions": _to_int(row.get("total_questions"), len(str(row["answer_string"]))),
                "start_dt": start_dt,
                "end_dt": end_dt,
                "negative_marking": _to_bool(row.get("negative_marking", False)),
                "negative_marks_value": _to_float(row.get("negative_marks_value"), 0.0),
            }
    return None


def get_upcoming_answer_key(now=None):
    """Finds the nearest upcoming exam (start time in the future), or None."""
    if now is None:
        now = now_bd()
    df = get_all_answer_keys()
    if df.empty:
        return None
    best_row = None
    best_dt = None
    for _, row in df.iterrows():
        try:
            start_dt = datetime.strptime(str(row["start_time"]), "%Y-%m-%d %H:%M")
        except Exception:
            continue
        if start_dt > now and (best_dt is None or start_dt < best_dt):
            best_dt = start_dt
            best_row = row
    if best_row is None:
        return None
    return {
        "key_id": best_row["key_id"],
        "exam_name": best_row.get("exam_name", ""),
        "start_dt": best_dt,
    }


# ================= Config (generic key-value store) =================

def set_config_value(key, value):
    ws = _cached_worksheet("Config")
    values = _with_retry(ws.get_all_values)
    json_str = json.dumps(value)
    row_idx = None
    for i, row in enumerate(values):
        if row and row[0] == key:
            row_idx = i + 1
            break
    if row_idx:
        _with_retry(ws.update, f"A{row_idx}:B{row_idx}", [[key, json_str]], value_input_option=RAW)
    else:
        _with_retry(ws.append_row, [key, json_str], value_input_option=RAW)


def get_config_value(key, default=None):
    ws = _cached_worksheet("Config")
    values = _with_retry(ws.get_all_values)
    for row in values:
        if row and row[0] == key:
            try:
                return json.loads(row[1])
            except Exception:
                return default
    return default


# ================= Calibration =================

def save_calibration(calibration_dict):
    set_config_value("calibration", calibration_dict)


def load_calibration():
    return get_config_value("calibration", default=None)


# ================= Mentor Password =================

def get_mentor_password():
    saved = get_config_value("mentor_password", default=None)
    if saved:
        return saved
    return st.secrets.get("MENTOR_PASSWORD", "")


def set_mentor_password(new_password):
    set_config_value("mentor_password", new_password)


# ================= Password hashing & strength =================

def _hash(value, salt):
    return hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(8)
    return _hash(password, salt), salt


def verify_password(password, salt, expected_hash):
    if not salt or not expected_hash:
        return False
    return secrets.compare_digest(_hash(password, salt), expected_hash)


def password_strength(password):
    """
    Returns (score 0-4, label, list of unmet requirement tips).
    Used to show a live strength meter and to block weak passwords.
    """
    tips = []
    if len(password) < 6:
        tips.append("At least 6 characters")
    if not re.search(r"[A-Za-z]", password):
        tips.append("At least 1 letter")
    if not re.search(r"[0-9]", password):
        tips.append("At least 1 number")

    score = 0
    if len(password) >= 6:
        score += 1
    if len(password) >= 10:
        score += 1
    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 1
    if re.search(r"[0-9]", password):
        score += 1
    if re.search(r"[^A-Za-z0-9]", password):
        score += 1
    score = min(score, 4)

    labels = {0: "Very weak", 1: "Weak", 2: "Fair", 3: "Good", 4: "Strong"}
    return score, labels[score], tips


def _gen_temp_password(length=8):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ================= Students =================

def _all_students_df():
    ws = _cached_worksheet("Students")
    records = _with_retry(ws.get_all_records)
    return pd.DataFrame(records)


def get_all_students_df():
    return _all_students_df()


def get_student_by_phone(phone):
    df = _all_students_df()
    if df.empty:
        return None
    target = _normalize_phone(phone)
    match = df[df["phone"].astype(str).apply(_normalize_phone) == target]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_student_by_id(student_id):
    df = _all_students_df()
    if df.empty:
        return None
    match = df[df["student_id"] == student_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def create_student(name, phone, password, security_question, security_answer):
    phone = _normalize_phone(phone)
    if not phone:
        raise ValueError("Please enter a valid phone number.")
    if get_student_by_phone(phone):
        raise ValueError("This phone number is already registered.")
    ws = _cached_worksheet("Students")
    existing = _with_retry(ws.get_all_records)
    student_id = f"S{len(existing) + 1:04d}"
    pw_hash, salt = hash_password(password)
    ans_hash, _ = hash_password(security_answer.strip().lower(), salt)
    _with_retry(
        ws.append_row,
        [student_id, name.strip(), phone, pw_hash, salt,
         security_question, ans_hash, False, 1, now_bd().strftime("%Y-%m-%d %H:%M:%S")],
        value_input_option=RAW,
    )
    clear_data_caches()
    return student_id


def authenticate_student(phone, password):
    """Returns student dict on success, or raises ValueError with a friendly message."""
    student = get_student_by_phone(phone)
    if not student:
        raise ValueError("No account found with this phone number.")
    if _to_bool(student.get("disabled", False)):
        raise ValueError("Your account has been disabled. Please contact your mentor.")
    if not verify_password(password, student.get("salt", ""), student.get("password_hash", "")):
        raise ValueError("Incorrect password.")
    return student


def _find_student_row_idx(ws, student_id):
    values = _with_retry(ws.get_all_values)
    for i, row in enumerate(values):
        if row and row[0] == student_id:
            return i + 1
    return None


def change_student_password(student_id, new_password):
    """Student changes their own password (or self-service reset). Bumps
    session_version so any other logged-in session gets kicked out."""
    ws = _cached_worksheet("Students")
    row_idx = _find_student_row_idx(ws, student_id)
    if not row_idx:
        raise ValueError("Student not found.")
    student = get_student_by_id(student_id)
    pw_hash, salt = hash_password(new_password)
    new_version = _to_int(student.get("session_version"), 1) + 1
    _with_retry(ws.update, f"D{row_idx}:E{row_idx}", [[pw_hash, salt]], value_input_option=RAW)
    _with_retry(ws.update, f"I{row_idx}", [[new_version]], value_input_option=RAW)
    clear_data_caches()


def reset_password_via_security(phone, security_answer, new_password):
    student = get_student_by_phone(phone)
    if not student:
        raise ValueError("No account found with this phone number.")
    if not verify_password(security_answer.strip().lower(), student.get("salt", ""),
                            student.get("security_answer_hash", "")):
        raise ValueError("Security answer is incorrect.")
    change_student_password(student["student_id"], new_password)


def admin_reset_password(student_id):
    """Mentor-triggered reset -> returns a temp password to hand to the student.
    Not currently exposed in the UI (students self-serve via Forgot Password),
    kept here in case a mentor override is needed via support in the future."""
    temp_password = _gen_temp_password()
    change_student_password(student_id, temp_password)
    return temp_password


def set_student_disabled(student_id, disabled: bool):
    ws = _cached_worksheet("Students")
    row_idx = _find_student_row_idx(ws, student_id)
    if not row_idx:
        raise ValueError("Student not found.")
    _with_retry(ws.update, f"H{row_idx}", [[bool(disabled)]], value_input_option=RAW)
    if disabled:
        # also bump session version so an already-open session gets logged out
        student = get_student_by_id(student_id)
        new_version = _to_int(student.get("session_version"), 1) + 1
        _with_retry(ws.update, f"I{row_idx}", [[new_version]], value_input_option=RAW)
    clear_data_caches()


def get_session_version(student_id):
    student = get_student_by_id(student_id)
    if not student:
        return None
    return _to_int(student.get("session_version"), 1)


# ================= Results =================

def has_submitted(student_id, key_id):
    """Duplicate-submission protection: True if this student already has a
    result recorded for this exam key."""
    df = get_all_results_df()
    if df.empty:
        return False
    match = df[(df["student_id"] == student_id) & (df["key_id"] == key_id)]
    return not match.empty


def append_result(student_id, student_name, key_id, result):
    """
    result is the dict returned by omr_scanner.score_answers():
    total, answered, skipped, correct, wrong_count, wrong,
    wrong_details, accuracy, marks, negative_marking, negative_value
    """
    ws = _cached_worksheet("Results")
    timestamp = now_bd().strftime("%Y-%m-%d %H:%M:%S")
    wrong_str = ",".join(str(q) for q in result.get("wrong", []))
    wrong_details_json = json.dumps(result.get("wrong_details", {}))
    skipped_qs = result.get("skipped_questions")
    if skipped_qs is None:
        skipped_qs = []
    skipped_json = json.dumps(skipped_qs)
    row = [
        timestamp, student_id, student_name, key_id,
        result.get("total", 0), result.get("answered", 0), result.get("skipped", 0),
        result.get("correct", 0), result.get("wrong_count", 0), wrong_str,
        result.get("marks", 0), result.get("accuracy", 0),
        result.get("negative_marking", False), result.get("negative_value", 0.0),
        False, wrong_details_json, skipped_json,
    ]
    _with_retry(ws.append_row, row, value_input_option=RAW)
    clear_data_caches()


def update_result(student_id, key_id, new_marks=None, new_correct=None,
                   new_wrong_count=None, new_wrong=None):
    """Mentor edit/override for a single result row."""
    ws = _cached_worksheet("Results")
    values = _with_retry(ws.get_all_values)
    header = values[0]
    row_idx = None
    for i, row in enumerate(values[1:], start=2):
        rec = dict(zip(header, row))
        if rec.get("student_id") == student_id and rec.get("key_id") == key_id:
            row_idx = i
            break
    if not row_idx:
        raise ValueError("Result not found.")

    updates = {}
    if new_correct is not None:
        updates["correct"] = new_correct
    if new_wrong_count is not None:
        updates["wrong_count"] = new_wrong_count
    if new_wrong is not None:
        updates["wrong"] = ",".join(str(q) for q in new_wrong)
    if new_marks is not None:
        updates["marks"] = new_marks
    updates["edited_by_mentor"] = True

    for col, val in updates.items():
        col_idx = header.index(col) + 1
        col_letter = gspread.utils.rowcol_to_a1(1, col_idx).rstrip("1")
        _with_retry(ws.update, f"{col_letter}{row_idx}", [[val]], value_input_option=RAW)
    clear_data_caches()


def get_all_results_df():
    ws = _cached_worksheet("Results")
    records = _with_retry(ws.get_all_records)
    df = pd.DataFrame(records)
    if not df.empty:
        for col in ["total", "answered", "skipped", "correct", "wrong_count", "marks", "accuracy"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_results_for_student(student_id):
    df = get_all_results_df()
    if df.empty:
        return df
    df = df[df["student_id"] == student_id].copy()
    if df.empty:
        return df
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def get_leaderboard_by_key(key_id):
    """Leaderboard for one specific exam - best marks per student."""
    df = get_all_results_df()
    if df.empty:
        return df
    df = df[df["key_id"] == key_id]
    if df.empty:
        return df
    best = df.sort_values("marks", ascending=False).drop_duplicates("student_id")
    best = best.sort_values("marks", ascending=False).reset_index(drop=True)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best


def get_overall_leaderboard():
    """Ranking by average percentage (total marks / total possible) across all exams."""
    df = get_all_results_df()
    if df.empty:
        return df
    grouped = df.groupby(["student_id", "student"]).agg(
        total_marks=("marks", "sum"),
        total_possible=("total", "sum"),
        exams_taken=("key_id", "nunique"),
    ).reset_index()
    grouped["avg_percent"] = (grouped["total_marks"] / grouped["total_possible"] * 100).round(2)
    grouped = grouped.sort_values("avg_percent", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped


def get_rank_for_student(student_id, key_id=None):
    """Returns (rank, out_of) for either one exam or the overall leaderboard."""
    df = get_leaderboard_by_key(key_id) if key_id else get_overall_leaderboard()
    if df is None or df.empty:
        return None, 0
    match = df[df["student_id"] == student_id]
    if match.empty:
        return None, len(df)
    return int(match.iloc[0]["rank"]), len(df)


# ================= Mentor analytics =================

def get_mentor_analytics():
    students_df = get_all_students_df()
    results_df = get_all_results_df()

    total_students = len(students_df) if not students_df.empty else 0
    active_students = 0
    if not students_df.empty and "disabled" in students_df.columns:
        active_students = int((~students_df["disabled"].apply(_to_bool)).sum())

    total_submissions = len(results_df) if not results_df.empty else 0

    average_score_pct = 0.0
    submissions_today = 0
    if not results_df.empty:
        pct = (results_df["marks"] / results_df["total"].replace(0, pd.NA)) * 100
        average_score_pct = round(pct.mean(skipna=True), 1) if not pct.dropna().empty else 0.0
        today_str = now_bd().strftime("%Y-%m-%d")
        submissions_today = int(results_df["timestamp"].astype(str).str.startswith(today_str).sum())

    active_key = get_active_answer_key()

    return {
        "total_students": total_students,
        "active_students": active_students,
        "total_submissions": total_submissions,
        "average_score_pct": average_score_pct,
        "submissions_today": submissions_today,
        "active_exam": active_key["exam_name"] if active_key else None,
    }


# ================= Export =================

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df, sheet_name="Results"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()
