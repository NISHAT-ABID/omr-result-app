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


@st.cache_resource(show_spinner=False)
def get_client():
    """Service account credential diye gspread client toiri kore."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource(show_spinner=False)
def get_spreadsheet():
    client = get_client()
    return client.open_by_key(st.secrets["SHEET_ID"])


def _get_or_create_worksheet(sh, title, header):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=len(header) + 2)
        ws.append_row(header)
        return ws
    # ensure header exists
    values = ws.get_all_values()
    if not values:
        ws.append_row(header)
    return ws


def init_sheets():
    """First time run hole shob worksheet toiri kore dey."""
    sh = get_spreadsheet()
    _get_or_create_worksheet(sh, "AnswerKeys", ANSWERKEYS_HEADER)
    _get_or_create_worksheet(sh, "Config", CONFIG_HEADER)
    _get_or_create_worksheet(sh, "Results", RESULTS_HEADER)
    return sh


# ---------------- Answer Keys ----------------

def add_answer_key(date_str, start_time_str, end_time_str, answer_string):
    sh = get_spreadsheet()
    ws = _get_or_create_worksheet(sh, "AnswerKeys", ANSWERKEYS_HEADER)
    existing = ws.get_all_records()
    key_id = f"K{len(existing) + 1:04d}"
    ws.append_row([key_id, date_str, start_time_str, end_time_str, answer_string])
    return key_id


def get_all_answer_keys():
    sh = get_spreadsheet()
    ws = _get_or_create_worksheet(sh, "AnswerKeys", ANSWERKEYS_HEADER)
    records = ws.get_all_records()
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


# ---------------- Config / Calibration ----------------

def save_calibration(calibration_dict):
    sh = get_spreadsheet()
    ws = _get_or_create_worksheet(sh, "Config", CONFIG_HEADER)
    values = ws.get_all_values()
    json_str = json.dumps(calibration_dict)

    # existing row thakle update, na thakle notun row
    row_idx = None
    for i, row in enumerate(values):
        if row and row[0] == "calibration":
            row_idx = i + 1  # gspread 1-indexed
            break

    if row_idx:
        ws.update(f"A{row_idx}:B{row_idx}", [["calibration", json_str]])
    else:
        ws.append_row(["calibration", json_str])


def load_calibration():
    sh = get_spreadsheet()
    ws = _get_or_create_worksheet(sh, "Config", CONFIG_HEADER)
    values = ws.get_all_values()
    for row in values:
        if row and row[0] == "calibration":
            try:
                return json.loads(row[1])
            except Exception:
                return None
    return None


# ---------------- Results ----------------

def append_result(student, key_id, score, total, wrong_questions):
    sh = get_spreadsheet()
    ws = _get_or_create_worksheet(sh, "Results", RESULTS_HEADER)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wrong_str = ",".join(str(q) for q in wrong_questions)
    ws.append_row([timestamp, student, key_id, score, total, wrong_str])


def get_leaderboard():
    sh = get_spreadsheet()
    ws = _get_or_create_worksheet(sh, "Results", RESULTS_HEADER)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    # protir student er best/latest score dekhano hocche best score
    best = df.sort_values("score", ascending=False).drop_duplicates("student")
    best = best.sort_values("score", ascending=False).reset_index(drop=True)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best
