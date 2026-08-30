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
                 session_version, created_at, birth_date, gender

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

IMPORTANT - Phone column kept as plain TEXT, belt-and-suspenders:
RAW input alone stops Sheets from re-parsing NEW writes, but a column
that was ever populated with an auto-parsed number earlier (e.g. before
RAW was added, or via manual editing in the Sheets UI) can be left with
a "Number" cell format. Google Sheets will still *display* a text value
that looks numeric using that stale number format, which can visually
hide/confuse a leading zero. So on every write to the Students sheet we
also explicitly force the Phone column's cell format to plain text
(see `_force_phone_column_text_format`). This makes the leading-zero
bug impossible going forward, independent of what happened to the sheet
in the past. Rows created *before* this fix may still have the old,
already-corrupted value baked in - `format_bd_phone()` below applies a
best-effort cosmetic fix for those legacy rows when displaying them.

IMPORTANT - Concurrent-user duplicate-submission guard:
With multiple students able to use the app at the same time, the OLD
pattern used by the Streamlit layer was: read cached results, check
`has_submitted()`, and only THEN call `append_result()` as two entirely
separate steps. That left a real race window - if the same student
double-clicked Submit, had two tabs open, or two students happened to
submit at nearly the same instant, both requests could pass the
"already submitted?" check before either one had actually written its
row, producing a duplicate result in the Results sheet. Google Sheets'
API has no row-level lock we can take from gspread, so this can't be
made fully atomic - but `append_result_if_not_submitted()` below
collapses the check and the write into one function call, re-reading
the Results sheet fresh (bypassing any Streamlit-level cache) right
before writing. That shrinks the race window from "the whole photo
upload + calibration flow" down to a single network round trip
immediately before the write, which is enough to stop the realistic
causes (double-click, duplicate tab, near-simultaneous different
students) in normal use. Callers should use this function instead of
the separate has_submitted() + append_result() pattern for new
submissions; append_result() is kept only for any code that still needs
the old two-step form.

IMPORTANT - Optional profile fields (birth_date, gender):
These are intentionally NOT collected at signup - creating an account
only ever needs name + phone + password (+ a security question for
password recovery). birth_date/gender are purely optional and can be
added or changed any time afterwards from the student's own Profile
page (see update_student_extra_profile() below). Phone number is NEVER
editable anywhere in this app, by design - it's the student's login
identity - so there is deliberately no function/parameter anywhere that
lets it be changed except by direct database/support access.
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
    "duration_minutes",
]

RESULTS_HEADER = [
    "timestamp", "student_id", "student", "key_id", "total", "answered", "skipped",
    "correct", "wrong_count", "wrong", "marks", "accuracy",
    "negative_marking", "negative_value", "edited_by_mentor",
    "wrong_details_json", "skipped_json",
]

CONFIG_HEADER = ["config_key", "config_value"]

# birth_date / gender are appended at the END of the header (not inserted
# in the middle) so every existing hardcoded column reference elsewhere
# in this file (e.g. "D{row_idx}:E{row_idx}" for password_hash/salt, or
# "H{row_idx}" for disabled) keeps pointing at the same column letters it
# always has - only new rows/edits ever touch the two new columns.
STUDENTS_HEADER = [
    "student_id", "name", "phone", "password_hash", "salt",
    "security_question", "security_answer_hash", "disabled",
    "session_version", "created_at", "birth_date", "gender",
]
PHONE_COL_INDEX = STUDENTS_HEADER.index("phone") + 1  # 1-based, for gspread.format()

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


PHONE_COUNTRY_CODE = "880"


def _normalize_phone(phone):
    """Canonicalizes ANY phone representation we might encounter - a fresh
    '+880 1712345678' style entry, an old-style local '01712345678' row
    saved before the phone-field redesign, or an already-canonical
    '8801712345678' - down to one consistent form: '880' + the 10 local
    digits (no '+', no leading 0). Used everywhere a phone number is
    stored or looked up, so signup/login always agree regardless of which
    era a particular row was created in - no data migration needed."""
    digits = re.sub(r"\D", "", str(phone or ""))
    if digits.startswith(PHONE_COUNTRY_CODE) and len(digits) == 13:
        return digits
    if digits.startswith("0") and len(digits) == 11:
        return PHONE_COUNTRY_CODE + digits[1:]
    if len(digits) == 10:
        return PHONE_COUNTRY_CODE + digits
    return digits


def format_bd_phone(phone):
    """Display form: '+880 1712345678'. Works for any stored variant
    (old or new) since it normalizes first."""
    canon = _normalize_phone(phone)
    if canon.startswith(PHONE_COUNTRY_CODE) and len(canon) == 13:
        return "+880 " + canon[3:]
    return str(phone or "")


def validate_bd_phone_digits(digits):
    """Validates what the user typed into the '+880 [______]' field (i.e.
    everything AFTER the fixed +880 prefix - no '0', no '+880' expected
    from the user at all, which is what removes the whole leading-zero
    class of bug).

    Returns (ok, error_message_or_None, canonical_phone_or_None).
    canonical_phone is ready to pass straight into create_student /
    authenticate_student / get_student_by_phone.
    """
    digits = re.sub(r"\D", "", digits or "")
    if not digits:
        return False, "Please enter your phone number.", None
    if len(digits) > 10:
        return False, (
            f"Too many digits ({len(digits)}). Just type the 10 digits that come "
            f"after +880 - skip the leading 0 and don't type +880 again "
            f"(e.g. for 01712345678, type 1712345678)."
        ), None
    if len(digits) < 10:
        return False, f"Enter all 10 digits after +880 - you've typed {len(digits)} so far.", None
    if digits[0] != "1":
        return False, "A Bangladeshi mobile number starts with 1 right after +880 (e.g. +880 1712345678).", None
    return True, None, PHONE_COUNTRY_CODE + digits


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
        if title == "Students":
            _force_phone_column_text_format(ws)
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
    if title == "Students":
        _force_phone_column_text_format(ws)
    return ws


def _force_phone_column_text_format(ws):
    """Belt-and-suspenders fix for the 'leading zero disappears from phone
    numbers' bug: force the whole Phone column to plain-text cell format,
    so Google Sheets never re-renders a stored value using a stale Number
    format (which is what makes a leading zero vanish visually / on
    export), regardless of how old data in that column got there."""
    try:
        col_letter = gspread.utils.rowcol_to_a1(1, PHONE_COL_INDEX).rstrip("1")
        _with_retry(ws.format, f"{col_letter}:{col_letter}", {"numberFormat": {"type": "TEXT"}})
    except Exception:
        pass  # cosmetic/defensive only - never block app startup on this


def _safe_get_all_records(ws):
    """
    Crash-proof replacement for gspread's ws.get_all_records().

    gspread's own get_all_records() RAISES GSpreadException the moment the
    header row contains any blank or duplicate column name - and that's
    exactly what an ordinary, harmless sheet can end up with: a worksheet
    created with a couple of spare trailing columns (see cols=len(header)+2
    in _get_or_create_worksheet), a header that grew over time (like
    birth_date/gender being appended to Students here), or someone just
    clicking into an empty column in the Sheets UI. Any of those leaves an
    empty-string header cell, and gspread treats two empty-string headers
    as "duplicates" and throws - which previously meant a single messy
    column could crash EVERY page load of the whole app (any read through
    get_all_students_df() / get_all_answer_keys() / get_all_results_df()
    goes through this).

    This reads the exact same raw grid but builds each row's dict by hand
    instead of trusting gspread's strict (and fragile) header validation:
    - a blank header cell becomes "_blank_<column index>" instead of ""
    - a header cell that collides with an earlier one gets "_<n>" appended
    so every column still gets a unique, stable key - no exception, and
    every legitimate (non-blank, non-duplicate) column still reads under
    its normal name exactly as before.
    - short data rows are padded with "" (blank cell) so every column is
      always present, matching gspread's own default_blank='' behavior.
    - data rows longer than the header are truncated to the header's
      width, since there's no header name to hang the extra cells on.
    """
    values = _with_retry(ws.get_all_values)
    if not values:
        return []

    raw_header = values[0]
    seen = {}
    safe_header = []
    for i, h in enumerate(raw_header):
        name = (h or "").strip() or f"_blank_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        safe_header.append(name)

    width = len(safe_header)
    records = []
    for row in values[1:]:
        if not any(cell.strip() for cell in row):
            continue  # skip fully blank rows, same as gspread's default
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        records.append(dict(zip(safe_header, row)))
    return records


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
                    negative_marking=False, negative_marks_value=0.0,
                    duration_minutes=0):
    ws = _cached_worksheet("AnswerKeys")
    # Small collision-retry loop: two mentors saving an exam at almost the
    # exact same instant could otherwise both compute the same next
    # key_id from the same "existing rows" snapshot and end up appending
    # two rows with an identical key_id. This doesn't need a hard lock
    # (mentor accounts are few and this is a rare, low-stakes collision),
    # but re-checking right before the write and retrying with the next
    # number if a clash is spotted costs almost nothing and closes the
    # gap for the common case.
    for _attempt in range(5):
        existing = _safe_get_all_records(ws)
        key_id = f"K{len(existing) + 1:04d}"
        if any(str(r.get("key_id")) == key_id for r in existing):
            continue  # someone else just took this number - recompute and retry
        _with_retry(
            ws.append_row,
            [key_id, exam_name, date_str, start_time_str, end_time_str,
             total_questions, answer_string, negative_marking, negative_marks_value,
             duration_minutes],
            value_input_option=RAW,
        )
        clear_data_caches()
        return key_id
    raise ValueError("Could not save the answer key right now (too many concurrent saves). Please try again.")


def get_all_answer_keys():
    ws = _cached_worksheet("AnswerKeys")
    records = _safe_get_all_records(ws)
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
        "duration_minutes": _to_int(row.get("duration_minutes"), 0),
    }


def get_active_answer_key(now=None):
    """
    Finds which answer key is active right now.
    Returns: dict {key_id, exam_name, answer_string, total_questions,
                   start_dt, end_dt, negative_marking, negative_marks_value,
                   duration_minutes}
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
                "duration_minutes": _to_int(row.get("duration_minutes"), 0),
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
        "duration_minutes": _to_int(best_row.get("duration_minutes"), 0),
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


# ================= Mentor Profile (display name) =================
# There's only ever one mentor login (a single shared password, not a row
# per mentor like Students), so the "profile" is just an optional display
# name stored in the generic Config key-value store - same pattern as the
# mentor password above. Defaults to "Mentor" until someone sets one.

def get_mentor_name():
    saved = get_config_value("mentor_name", default=None)
    return (saved or "").strip() or "Mentor"


def set_mentor_name(new_name):
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("Name cannot be empty.")
    set_config_value("mentor_name", new_name)


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
    records = _safe_get_all_records(ws)
    df = pd.DataFrame(records)
    if not df.empty and "phone" in df.columns:
        # Belt-and-suspenders: get_all_records() can hand back a phone-like
        # column as an int if gspread infers a numeric type for the whole
        # column (e.g. from legacy data). Force it back to a zero-padded
        # string of digits so nothing downstream (login lookup, display,
        # search) ever silently drops a leading zero again in-memory.
        df["phone"] = df["phone"].apply(lambda v: _normalize_phone(v))
    return df


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
    """Creates a new student account. Deliberately asks for the bare
    minimum only - name, phone, password, security question/answer.
    birth_date and gender are NOT collected here; they start blank and
    can be filled in any time afterwards from the Profile page via
    update_student_extra_profile()."""
    phone = _normalize_phone(phone)
    if not phone:
        raise ValueError("Please enter a valid phone number.")
    if get_student_by_phone(phone):
        raise ValueError("This phone number is already registered.")
    ws = _cached_worksheet("Students")
    # Same small collision-retry idea as add_answer_key(): two people
    # signing up at almost the exact same instant could otherwise compute
    # the same next student_id from the same snapshot of existing rows.
    for _attempt in range(5):
        existing = _safe_get_all_records(ws)
        student_id = f"S{len(existing) + 1:04d}"
        if any(str(r.get("student_id")) == student_id for r in existing):
            continue
        pw_hash, salt = hash_password(password)
        ans_hash, _ = hash_password(security_answer.strip().lower(), salt)
        _with_retry(
            ws.append_row,
            [student_id, name.strip(), phone, pw_hash, salt,
             security_question, ans_hash, False, 1, now_bd().strftime("%Y-%m-%d %H:%M:%S"),
             "", ""],
            value_input_option=RAW,
        )
        clear_data_caches()
        return student_id
    raise ValueError("Could not create the account right now (too many concurrent signups). Please try again.")


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


def update_student_name(student_id, new_name):
    """
    Updates a student's display name in the Students sheet, AND
    propagates it to every PAST Results row belonging to that student.

    The Results sheet stores the student's name denormalized in its own
    'student' column (alongside student_id) - this is what lets the
    leaderboard/analytics reads work directly off Results without joining
    back to Students on every read. The tradeoff is that a name change
    would otherwise only show up on brand-new submissions going forward,
    while every past test-history row kept displaying the old, stale
    name - which is exactly what a student changing their name on the
    Profile page would not expect. This function keeps both sheets in
    sync in one call.

    All matching Results rows are updated in a SINGLE batch_update call
    (rather than one API request per row) so a student with a long test
    history doesn't trigger dozens of slow, rate-limit-prone writes.
    """
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("Name cannot be empty.")

    students_ws = _cached_worksheet("Students")
    row_idx = _find_student_row_idx(students_ws, student_id)
    if not row_idx:
        raise ValueError("Student not found.")
    name_col_idx = STUDENTS_HEADER.index("name") + 1
    col_letter = gspread.utils.rowcol_to_a1(1, name_col_idx).rstrip("1")
    _with_retry(students_ws.update, f"{col_letter}{row_idx}", [[new_name]], value_input_option=RAW)

    results_ws = _cached_worksheet("Results")
    values = _with_retry(results_ws.get_all_values)
    if values:
        header = values[0]
        if "student_id" in header and "student" in header:
            sid_idx = header.index("student_id")
            name_idx = header.index("student")
            name_col_letter = gspread.utils.rowcol_to_a1(1, name_idx + 1).rstrip("1")
            batch = []
            for i, row in enumerate(values[1:], start=2):
                if len(row) > sid_idx and row[sid_idx] == student_id and row[name_idx] != new_name:
                    batch.append({"range": f"{name_col_letter}{i}", "values": [[new_name]]})
            if batch:
                _with_retry(results_ws.batch_update, batch, value_input_option=RAW)

    clear_data_caches()


def update_student_extra_profile(student_id, birth_date=None, gender=None):
    """Updates the optional Profile-page fields (birth date, gender) that
    a student can add or change any time after signing up - these are
    deliberately NOT asked for at signup (see create_student()).

    Passing None for either argument leaves that column untouched, so
    callers can update just one field without clobbering the other.
    Passing an empty string clears that field.

    Phone number is intentionally NOT a parameter here (or anywhere in
    this app) - it's the student's login identity and is never editable
    through self-service profile editing.
    """
    ws = _cached_worksheet("Students")
    row_idx = _find_student_row_idx(ws, student_id)
    if not row_idx:
        raise ValueError("Student not found.")

    updates = {}
    if birth_date is not None:
        updates["birth_date"] = birth_date
    if gender is not None:
        updates["gender"] = gender
    if not updates:
        return

    for col, val in updates.items():
        col_idx = STUDENTS_HEADER.index(col) + 1
        col_letter = gspread.utils.rowcol_to_a1(1, col_idx).rstrip("1")
        _with_retry(ws.update, f"{col_letter}{row_idx}", [[val]], value_input_option=RAW)
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
    result recorded for this exam key.

    NOTE: this reads through get_all_results_df(), which the Streamlit
    layer wraps in @st.cache_data - i.e. this can return a slightly STALE
    answer (up to the cache's TTL old). That's fine for deciding whether to
    even SHOW the submit form, but it must never be the only guard right
    before a write - see append_result_if_not_submitted() below, which
    re-checks with a fresh, uncached read at the moment of writing.
    """
    df = get_all_results_df()
    if df.empty:
        return False
    match = df[(df["student_id"] == student_id) & (df["key_id"] == key_id)]
    return not match.empty


def _result_row_values(student_id, student_name, key_id, result):
    timestamp = now_bd().strftime("%Y-%m-%d %H:%M:%S")
    wrong_str = ",".join(str(q) for q in result.get("wrong", []))
    wrong_details_json = json.dumps(result.get("wrong_details", {}))
    skipped_qs = result.get("skipped_questions")
    if skipped_qs is None:
        skipped_qs = []
    skipped_json = json.dumps(skipped_qs)
    return [
        timestamp, student_id, student_name, key_id,
        result.get("total", 0), result.get("answered", 0), result.get("skipped", 0),
        result.get("correct", 0), result.get("wrong_count", 0), wrong_str,
        result.get("marks", 0), result.get("accuracy", 0),
        result.get("negative_marking", False), result.get("negative_value", 0.0),
        False, wrong_details_json, skipped_json,
    ]


def append_result(student_id, student_name, key_id, result):
    """
    result is the dict returned by omr_scanner.score_answers():
    total, answered, skipped, correct, wrong_count, wrong,
    wrong_details, accuracy, marks, negative_marking, negative_value

    NOTE: this does NOT check for an existing submission first - callers
    are responsible for that. New submission code should call
    append_result_if_not_submitted() instead, which does the check and
    the write together against a fresh read. This function is kept for
    any other caller that manages its own duplicate check (e.g. a mentor
    override flow that intentionally wants to add a row).
    """
    ws = _cached_worksheet("Results")
    row = _result_row_values(student_id, student_name, key_id, result)
    _with_retry(ws.append_row, row, value_input_option=RAW)
    clear_data_caches()


def append_result_if_not_submitted(student_id, student_name, key_id, result):
    """
    Duplicate-submission guard + write, collapsed into one call.

    Re-reads the Results sheet FRESH (bypassing any Streamlit-level
    @st.cache_data caching that get_all_results_df()/has_submitted() go
    through) immediately before appending, and only writes if no existing
    row for this (student_id, key_id) pair is found yet. This is what
    should be used for every new student submission instead of the older
    "has_submitted() then append_result()" two-step pattern, which left a
    window where two near-simultaneous requests (double-click, two open
    tabs, or two different students submitting at almost the same moment)
    could both pass the check before either had written.

    This can't be made perfectly atomic (the Sheets API has no row lock),
    but doing the check and the write back-to-back inside one function
    call - right before the network write, with no caching or UI logic in
    between - shrinks the race window from "the whole photo-upload +
    calibration flow" down to a single network round trip. That's enough
    to stop the realistic causes of duplicates in normal use.

    Returns True if the result was saved, False if a submission already
    existed for this (student_id, key_id) and nothing was written.
    """
    ws = _cached_worksheet("Results")
    values = _with_retry(ws.get_all_values)
    if values:
        header = values[0]
        if "student_id" in header and "key_id" in header:
            sid_idx = header.index("student_id")
            key_idx = header.index("key_id")
            for row in values[1:]:
                if (
                    len(row) > max(sid_idx, key_idx)
                    and row[sid_idx] == student_id
                    and row[key_idx] == key_id
                ):
                    return False

    row = _result_row_values(student_id, student_name, key_id, result)
    _with_retry(ws.append_row, row, value_input_option=RAW)
    clear_data_caches()
    return True


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
    records = _safe_get_all_records(ws)
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
    """Ranking by average percentage (total marks / total possible) across
    all exams. Also includes each student's best score, mean accuracy, and
    a month-over-month trend (this month's average % vs last month's) so
    the leaderboard UI can show Tests Taken / Best Score / Average Score /
    Accuracy / Trend columns, not just a plain rank."""
    df = get_all_results_df()
    if df.empty:
        return df

    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    this_month = now_bd().month
    last_month = this_month - 1 or 12

    grouped = df.groupby(["student_id", "student"]).agg(
        total_marks=("marks", "sum"),
        total_possible=("total", "sum"),
        exams_taken=("key_id", "nunique"),
        best_score=("marks", "max"),
        accuracy=("accuracy", "mean"),
    ).reset_index()
    grouped["avg_percent"] = (grouped["total_marks"] / grouped["total_possible"] * 100).round(2)
    grouped["accuracy"] = grouped["accuracy"].round(2)
    grouped["best_score"] = grouped["best_score"].round(2)

    trends = {}
    for student_id, sub in df.groupby("student_id"):
        this_rows = sub[sub["ts"].dt.month == this_month]
        last_rows = sub[sub["ts"].dt.month == last_month]
        if this_rows.empty or last_rows.empty:
            trends[student_id] = None
            continue
        this_avg = (this_rows["marks"] / this_rows["total"]).mean() * 100
        last_avg = (last_rows["marks"] / last_rows["total"]).mean() * 100
        trends[student_id] = round(this_avg - last_avg, 1)
    grouped["trend"] = grouped["student_id"].map(trends)

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
