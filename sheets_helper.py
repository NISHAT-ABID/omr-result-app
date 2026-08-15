"""
sheets_helper.py
-----------------
All functions for using Google Sheets as the app's database.

3 worksheets (tabs) are needed inside one Google Sheet:
  1. AnswerKeys -> key_id, exam_name, date, start_time, end_time,
                   total_questions, answer_string,
                   negative_marking, negative_marks_value
  2. Config     -> config_key, config_value (generic key-value store -
                   used for calibration_100 / calibration_40 / mentor_password)
  3. Results    -> timestamp, student, key_id, total, answered, skipped,
                   correct, wrong_count, wrong, marks, accuracy,
                   negative_marking, negative_value

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

RESULTS_HEADER = [
    "timestamp", "student", "key_id", "total", "answered", "skipped",
    "correct", "wrong_count", "wrong", "marks", "accuracy",
    "negative_marking", "negative_value",
]

CONFIG_HEADER = ["config_key", "config_value"]

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
        ws = _with_retry(sh.add_worksheet, title=title, rows=1000, cols=len(header) + 2)
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
    }
    sh = get_spreadsheet()
    return _get_or_create_worksheet(sh, title, header_map[title])


def init_sheets():
    _cached_worksheet("AnswerKeys")
    _cached_worksheet("Config")
    _cached_worksheet("Results")
    return get_spreadsheet()


# ---------------- Answer Keys ----------------

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
    )
    return key_id


def get_all_answer_keys():
    ws = _cached_worksheet("AnswerKeys")
    records = _with_retry(ws.get_all_records)
    return pd.DataFrame(records)


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
# Each sheet layout (100Q / 40Q) is calibrated and stored SEPARATELY, keyed
# by total_questions, e.g. config_key "calibration_100" / "calibration_40".
# This is what makes the calibration auto-switch: whichever total_questions
# the mentor picked for an exam decides which saved calibration is used -
# no manual re-selection needed on the Calibration tab at scan time.

def save_calibration(calibration_dict, total_questions):
    set_config_value(f"calibration_{total_questions}", calibration_dict)


def load_calibration(total_questions):
    return get_config_value(f"calibration_{total_questions}", default=None)


# ---------------- Mentor Password ----------------

def get_mentor_password():
    saved = get_config_value("mentor_password", default=None)
    if saved:
        return saved
    return st.secrets.get("MENTOR_PASSWORD", "")


def set_mentor_password(new_password):
    set_config_value("mentor_password", new_password)


# ---------------- Results ----------------

def append_result(student, key_id, result):
    """
    result is the dict returned by omr_scanner.score_answers():
      total, answered, skipped, correct, wrong_count, wrong,
      wrong_details, skipped_list, skipped_details, accuracy, marks,
      negative_marking, negative_value
    """
    ws = _cached_worksheet("Results")
    timestamp = now_bd().strftime("%Y-%m-%d %H:%M:%S")
    wrong_str = ",".join(str(q) for q in result.get("wrong", []))
    row = [
        timestamp, student, key_id,
        result.get("total", 0), result.get("answered", 0), result.get("skipped", 0),
        result.get("correct", 0), result.get("wrong_count", 0), wrong_str,
        result.get("marks", 0), result.get("accuracy", 0),
        result.get("negative_marking", False), result.get("negative_value", 0.0),
    ]
    _with_retry(ws.append_row, row)


def get_all_results_df():
    ws = _cached_worksheet("Results")
    records = _with_retry(ws.get_all_records)
    df = pd.DataFrame(records)
    if not df.empty:
        for col in ["total", "answered", "skipped", "correct", "wrong_count", "marks", "accuracy"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_leaderboard_by_key(key_id):
    """Leaderboard for one specific exam - best marks per student."""
    df = get_all_results_df()
    if df.empty:
        return df
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
    if df.empty:
        return df
    grouped = df.groupby("student").agg(
        total_marks=("marks", "sum"),
        total_possible=("total", "sum"),
        exams_taken=("key_id", "nunique"),
    ).reset_index()
    grouped["avg_percent"] = (grouped["total_marks"] / grouped["total_possible"] * 100).round(2)
    grouped = grouped.sort_values("avg_percent", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped
