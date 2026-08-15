"""
sheets_helper.py
-----------------
Google Sheets ke database hisebe use korar jonno shob function ekhane.
3 ta worksheet (tab) lagbe ekta Google Sheet er moddhe:
  1. AnswerKeys  -> date, start_time, end_time, answer_string, key_id
  2. Config      -> calibration_json (OMR bubble position data)
  3. Results     -> timestamp, student, key_id, score, total, wrong_questions

Sheet er নাম/ID .streamlit/secrets.toml e SHEET_ID hisebe rakha thakbe.
"""

import json
import time
from datetime import datetime

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

ANSWERKEYS_HEADER = ["key_id", "date", "start_time", "end_time", "answer_string"]
RESULTS_HEADER = ["timestamp", "student", "key_id", "score", "total", "wrong_questions"]
CONFIG_HEADER = ["config_key", "config_value"]


def _with_retry(func, *args, **kwargs):
    """
    Google Sheets API majhe majhe rate-limit (429) error dey jodi
    onek gulo request khub tarataari jai. Eta hole ektu wait kore
    abar try kore - normally 2-3 bar er moddhei kaj kore jay.
    """
    delays = [1, 2, 4, 8]
    last_err = None
    for delay in delays:
        try:
            return func(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            last_err = e
            time.sleep(delay)
    return func(*args, **kwargs) if last_err is None else (_ for _ in ()).throw(last_err)


@st.cache_resource(show_spinner=False)
def get_client():
    """Service account credential diye gspread client toiri kore."""
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
    # ensure header exists
    values = _with_retry(ws.get_all_values)
    if not values:
        _with_retry(ws.append_row, header)
    return ws


@st.cache_resource(show_spinner=False)
def _cached_worksheet(title):
    """
    Worksheet object ekbar khuje pele seta cache kore rakha hoy, tai
    baar baar "eta ache kina" check korar jonno notun API call lage na.
    Eita e main fix jeta APIError (rate limit) thamiye dey.
    """
    header_map = {
        "AnswerKeys": ANSWERKEYS_HEADER,
        "Config": CONFIG_HEADER,
        "Results": RESULTS_HEADER,
    }
    sh = get_spreadsheet()
    return _get_or_create_worksheet(sh, title, header_map[title])


def init_sheets():
    """First time run hole shob worksheet toiri kore dey."""
    _cached_worksheet("AnswerKeys")
    _cached_worksheet("Config")
    _cached_worksheet("Results")
    return get_spreadsheet()


# ---------------- Answer Keys ----------------

def add_answer_key(date_str, start_time_str, end_time_str, answer_string):
    ws = _cached_worksheet("AnswerKeys")
    existing = _with_retry(ws.get_all_records)
    key_id = f"K{len(existing) + 1:04d}"
    _with_retry(ws.append_row, [key_id, date_str, start_time_str, end_time_str, answer_string])
    return key_id


def get_all_answer_keys():
    ws = _cached_worksheet("AnswerKeys")
    records = _with_retry(ws.get_all_records)
    return pd.DataFrame(records)


def get_active_answer_key(now=None):
    """
    Ekhon (now) shomoy onujayi kon answer key active ache seta ber kore.
    start_time / end_time format: 'YYYY-MM-DD HH:MM' (24 hour)
    Return: (key_id, answer_string) othoba (None, None) jodi kono active key na thake.
    """
    if now is None:
        now = datetime.now()
    df = get_all_answer_keys()
    if df.empty:
        return None, None

    for _, row in df.iterrows():
        try:
            start_dt = datetime.strptime(str(row["start_time"]), "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(str(row["end_time"]), "%Y-%m-%d %H:%M")
        except Exception:
            continue
        if start_dt <= now <= end_dt:
            return row["key_id"], row["answer_string"]
    return None, None


# ---------------- Config (generic key-value store) ----------------

def set_config_value(key, value):
    """Config sheet e je kono key-value save/update kore (JSON string hisebe)."""
    ws = _cached_worksheet("Config")
    values = _with_retry(ws.get_all_values)
    json_str = json.dumps(value)

    row_idx = None
    for i, row in enumerate(values):
        if row and row[0] == key:
            row_idx = i + 1  # gspread 1-indexed
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


# ---------------- Calibration (Config er upore banano) ----------------

def save_calibration(calibration_dict):
    set_config_value("calibration", calibration_dict)


def load_calibration():
    return get_config_value("calibration", default=None)


# ---------------- Mentor Password (mentor nijei UI theke change korte parbe) ----------------

def get_mentor_password():
    """
    Sheet e save kora password thakle seta use hobe.
    Prothom bar (Sheet e kichu save kora na thakle) Streamlit Secrets
    er MENTOR_PASSWORD ke "starting password" hisebe use kora hobe.
    """
    saved = get_config_value("mentor_password", default=None)
    if saved:
        return saved
    return st.secrets.get("MENTOR_PASSWORD", "")


def set_mentor_password(new_password):
    set_config_value("mentor_password", new_password)


# ---------------- Results ----------------

def append_result(student, key_id, score, total, wrong_questions):
    ws = _cached_worksheet("Results")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wrong_str = ",".join(str(q) for q in wrong_questions)
    _with_retry(ws.append_row, [timestamp, student, key_id, score, total, wrong_str])


def get_leaderboard():
    ws = _cached_worksheet("Results")
    records = _with_retry(ws.get_all_records)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    # protir student er best/latest score dekhano hocche best score
    best = df.sort_values("score", ascending=False).drop_duplicates("student")
    best = best.sort_values("score", ascending=False).reset_index(drop=True)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best
