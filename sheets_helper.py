"""
sheets_helper.py
-----------------
All functions for using Google Sheets as the app's database.

5 worksheets (tabs) are needed inside one Google Sheet:

1. AnswerKeys -> key_id, exam_name, date, start_time, end_time,
                 total_questions, answer_string,
                 negative_marking, negative_marks_value
2. Config     -> calibration_json, mentor_password (key-value store)
3. Results    -> timestamp, student, key_id, total, answered, skipped,
                 correct, wrong_count, wrong, wrong_details, marks,
                 accuracy, negative_marking, negative_value
4. Users      -> user_id, name, password_hash, role, created_at
5. Sessions   -> token, user_id, created_at, expires_at
                 (used for "remember this device" auto-login)

The Sheet name/ID is stored as SHEET_ID in .streamlit/secrets.toml.
"""

import json
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

ANSWERKEYS_HEADER = [
    "key_id", "exam_name", "date", "start_time", "end_time",
    "total_questions", "answer_string", "negative_marking", "negative_marks_value",
]

# "wrong_details" was added so a student's per-question review (what they
# picked vs the correct answer) can be shown later in "Result Analysis",
# not just right after submission. Old rows without this column still
# work fine - _get_all_records_safe() pads missing cells with "".
RESULTS_HEADER = [
    "timestamp", "student", "key_id", "total", "answered", "skipped",
    "correct", "wrong_count", "wrong", "wrong_details", "marks", "accuracy",
    "negative_marking", "negative_value",
]

CONFIG_HEADER = ["config_key", "config_value"]

USERS_HEADER = ["user_id", "name", "password_hash", "role", "created_at"]

SESSIONS_HEADER = ["token", "user_id", "created_at", "expires_at"]

BD_TZ = ZoneInfo("Asia/Dhaka")


def now_bd():
    """
    Streamlit Cloud's server runs on UTC time, but the mentor sets exam
    times based on Bangladesh time. So whenever we need "what time is it
    right now", use this function - it always returns the correct
    Bangladesh time regardless of where the server is located.
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


def _get_all_records_safe(ws, expected_header):
    """
    A drop-in replacement for gspread's ws.get_all_records().

    gspread's own get_all_records() raises GSpreadException if the sheet's
    actual header row (row 1) has duplicate or blank column names - which
    can easily happen after manual edits to the Google Sheet. That crashes
    the whole page.

    This version ignores whatever text is actually in row 1 and instead
    maps every data row (row 2 onwards) POSITIONALLY onto our own known
    `expected_header` column list. Extra/missing cells are padded with "".
    This can never raise a duplicate/blank-header error.
    """
    values = _with_retry(ws.get_all_values)
    if len(values) <= 1:
        return []

    records = []
    width = len(expected_header)
    for row in values[1:]:
        if not any(str(c).strip() for c in row):
            continue  # skip fully blank rows
        row = (row + [""] * width)[:width]  # pad/truncate to expected width
        records.append(dict(zip(expected_header, row)))
    return records


@st.cache_resource(show_spinner=False)
def get_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource(show_spinner=False)
def get_spreadsheet():
    client = get_client()
    return _with_retry(client.open_by_key, st.secrets["SHEET_ID"])


def _get_or_create_worksheet(sh_obj, title, header):
    try:
        ws = _with_retry(sh_obj.worksheet, title)
    except gspread.WorksheetNotFound:
        ws = _with_retry(sh_obj.add_worksheet, title=title, rows=1000, cols=len(header) + 2)
        _with_retry(ws.append_row, header)
        return ws

    values = _with_retry(ws.get_all_values)
    if not values:
        _with_retry(ws.append_row, header)
    else:
        existing_header = values[0]
        if existing_header != header and len(header) > len(existing_header):
            _with_retry(ws.update, "A1", [header])
    return ws


@st.cache_resource(show_spinner=False)
def _cached_worksheet(title):
    header_map = {
        "AnswerKeys": ANSWERKEYS_HEADER,
        "Config": CONFIG_HEADER,
        "Results": RESULTS_HEADER,
        "Users": USERS_HEADER,
        "Sessions": SESSIONS_HEADER,
    }
    sh_obj = get_spreadsheet()
    return _get_or_create_worksheet(sh_obj, title, header_map[title])


def init_sheets():
    _cached_worksheet("AnswerKeys")
    _cached_worksheet("Config")
    _cached_worksheet("Results")
    _cached_worksheet("Users")
    _cached_worksheet("Sessions")
    return get_spreadsheet()


# ---------------- Answer Keys ----------------

@st.cache_data(ttl=15, show_spinner=False)
def _fetch_answerkeys_records():
    ws = _cached_worksheet("AnswerKeys")
    return _get_all_records_safe(ws, ANSWERKEYS_HEADER)


def add_answer_key(exam_name, date_str, start_time_str, end_time_str,
                    total_questions, answer_string,
                    negative_marking=False, negative_marks_value=0.0):
    ws = _cached_worksheet("AnswerKeys")
    existing = _fetch_answerkeys_records()
    key_id = f"K{len(existing) + 1:04d}"
    _with_retry(
        ws.append_row,
        [key_id, exam_name, date_str, start_time_str, end_time_str,
         total_questions, answer_string, negative_marking, negative_marks_value],
    )
    _fetch_answerkeys_records.clear()
    return key_id


def get_all_answer_keys():
    return pd.DataFrame(_fetch_answerkeys_records())


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


# ---------------- Config (generic key-value store) ----------------

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
        _with_retry(ws.update, f"A{row_idx}:B{row_idx}", [[key, json_str]])
    else:
        _with_retry(ws.append_row, [key, json_str])
    get_config_value.clear()


@st.cache_data(ttl=15, show_spinner=False)
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


# ---------------- Calibration ----------------
# Calibration is stored SEPARATELY per sheet layout (100Q vs 40Q), since
# those are physically different printed sheets. Config key is namespaced
# by the question count, e.g. "calibration_100" / "calibration_40".

def save_calibration(calibration_dict, total_questions):
    set_config_value(f"calibration_{total_questions}", calibration_dict)


def load_calibration(total_questions):
    return get_config_value(f"calibration_{total_questions}", default=None)


# ---------------- Mentor invite code ----------------
# Historically this was called the "mentor password" - a single shared
# secret. It's now used as an INVITE CODE: anyone who knows it can create
# a personal Mentor account (see Users below). Existing MENTOR_PASSWORD
# in secrets.toml still works as the default invite code.

def get_mentor_password():
    saved = get_config_value("mentor_password", default=None)
    if saved:
        return saved
    return st.secrets.get("MENTOR_PASSWORD", "")


def set_mentor_password(new_password):
    set_config_value("mentor_password", new_password)


# ---------------- Users (login accounts) ----------------

@st.cache_data(ttl=20, show_spinner=False)
def _fetch_user_records():
    ws = _cached_worksheet("Users")
    return _get_all_records_safe(ws, USERS_HEADER)


def get_all_users_df():
    return pd.DataFrame(_fetch_user_records())


def user_exists(user_id):
    target = str(user_id).strip().lower()
    return any(str(r.get("user_id", "")).strip().lower() == target for r in _fetch_user_records())


def get_user(user_id):
    target = str(user_id).strip().lower()
    for rec in _fetch_user_records():
        if str(rec.get("user_id", "")).strip().lower() == target:
            return rec
    return None


def create_user(user_id, name, password_hash, role):
    ws = _cached_worksheet("Users")
    created_at = now_bd().strftime("%Y-%m-%d %H:%M:%S")
    _with_retry(ws.append_row, [user_id.strip(), name.strip(), password_hash, role, created_at])
    _fetch_user_records.clear()


def update_user_password(user_id, new_password_hash):
    ws = _cached_worksheet("Users")
    values = _with_retry(ws.get_all_values)
    target = str(user_id).strip().lower()
    for i, row in enumerate(values[1:], start=2):
        if row and str(row[0]).strip().lower() == target:
            _with_retry(ws.update, f"C{i}", [[new_password_hash]])
            break
    _fetch_user_records.clear()


# ---------------- Sessions ("remember this device") ----------------

def create_session(token, user_id, expires_at_str):
    ws = _cached_worksheet("Sessions")
    created_at = now_bd().strftime("%Y-%m-%d %H:%M:%S")
    _with_retry(ws.append_row, [token, user_id, created_at, expires_at_str])


def get_session(token):
    ws = _cached_worksheet("Sessions")
    records = _get_all_records_safe(ws, SESSIONS_HEADER)
    for rec in records:
        if rec.get("token") == token:
            try:
                expires_at = datetime.strptime(rec["expires_at"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None
            if expires_at < now_bd():
                delete_session(token)
                return None
            return rec
    return None


def delete_session(token):
    ws = _cached_worksheet("Sessions")
    values = _with_retry(ws.get_all_values)
    for i, row in enumerate(values[1:], start=2):
        if row and row[0] == token:
            _with_retry(ws.delete_rows, i)
            break


# ---------------- Results ----------------

def append_result(student, key_id, result):
    """
    result is the dict returned by omr_scanner.score_answers().
    wrong_details (given vs correct per wrong question) is now saved as
    JSON so students can review it later on the "Result Analysis" page,
    not just right after submitting.
    """
    ws = _cached_worksheet("Results")
    timestamp = now_bd().strftime("%Y-%m-%d %H:%M:%S")
    wrong_str = ",".join(str(q) for q in result.get("wrong", []))
    wrong_details_str = json.dumps(result.get("wrong_details", {}))
    row = [
        timestamp, student, key_id,
        result.get("total", 0), result.get("answered", 0), result.get("skipped", 0),
        result.get("correct", 0), result.get("wrong_count", 0), wrong_str, wrong_details_str,
        result.get("marks", 0), result.get("accuracy", 0),
        result.get("negative_marking", False), result.get("negative_value", 0.0),
    ]
    _with_retry(ws.append_row, row)
    _fetch_results_records.clear()


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_results_records():
    ws = _cached_worksheet("Results")
    return _get_all_records_safe(ws, RESULTS_HEADER)


def get_all_results_df():
    """
    Reads the Results sheet into a DataFrame (short-lived cache so the
    app doesn't hammer the Sheets API on every rerun / widget click).
    """
    df = pd.DataFrame(_fetch_results_records())

    if df.empty:
        return df

    for col in ["total", "answered", "skipped", "correct", "wrong_count", "marks", "accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_leaderboard_by_key(key_id):
    """Leaderboard for one specific exam - best marks per student."""
    df = get_all_results_df()
    required_cols = {"key_id", "marks", "student"}
    if df.empty or not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df = df[df["key_id"] == key_id]
    if df.empty:
        return df

    best = df.sort_values("marks", ascending=False).drop_duplicates("student")
    best = best.sort_values("marks", ascending=False).reset_index(drop=True)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best


def get_overall_leaderboard():
    """Ranking by average percentage (total marks / total possible) across all exams."""
    df = get_all_results_df()
    required_cols = {"student", "marks", "total", "key_id"}
    if df.empty or not required_cols.issubset(df.columns):
        return pd.DataFrame()

    grouped = df.groupby("student").agg(
        total_marks=("marks", "sum"),
        total_possible=("total", "sum"),
        exams_taken=("key_id", "nunique"),
    ).reset_index()

    grouped["avg_percent"] = grouped.apply(
        lambda r: round((r["total_marks"] / r["total_possible"]) * 100, 2) if r["total_possible"] else 0.0,
        axis=1,
    )
    grouped = grouped.sort_values("avg_percent", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped


def get_results_for_student(name):
    """All exam attempts by this student, newest first."""
    df = get_all_results_df()
    if df.empty or "student" not in df.columns:
        return pd.DataFrame()
    mine = df[df["student"] == name].copy()
    if mine.empty:
        return mine
    if "timestamp" in mine.columns:
        mine = mine.sort_values("timestamp", ascending=False)
    return mine.reset_index(drop=True)
