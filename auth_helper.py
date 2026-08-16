"""
auth_helper.py
---------------
Student account system for the OMR Result App.

Handles:
- Sign up with name + email + password -> emails a 6-digit OTP -> account
  is only activated after the OTP is verified.
- On activation, a unique Student ID (e.g. STU0001) is generated. This ID
  is what identifies the student everywhere in the app (results, exam
  submissions, leaderboard - internally).
- Login with email + password.
- Forgot password: emails an OTP -> verify OTP -> set a new password.

Storage: one more worksheet ("Users") in the same Google Sheet used by
sheets_helper.py.

Users worksheet columns:
    user_id, name, email, password_hash, salt, verified,
    otp_code, otp_expiry, otp_purpose, created_at

Passwords are never stored in plain text - only a salted SHA-256 hash.
"""

import hashlib
import secrets as pysecrets
import string
from datetime import datetime, timedelta

import streamlit as st

import sheets_helper as sh
from sheets_helper import _with_retry, _to_bool

USERS_HEADER = [
    "user_id", "name", "email", "password_hash", "salt", "verified",
    "otp_code", "otp_expiry", "otp_purpose", "created_at",
]

OTP_VALID_MINUTES = 10


# ---------------- Worksheet plumbing ----------------

@st.cache_resource(show_spinner=False)
def _users_ws():
    spreadsheet = sh.get_spreadsheet()
    return sh._get_or_create_worksheet(spreadsheet, "Users", USERS_HEADER)


def init_users_sheet():
    _users_ws()


def _get_all_users():
    ws = _users_ws()
    records = _with_retry(ws.get_all_records)
    return records, ws


def _find_user_row(email):
    """Returns (sheet_row_number, record_dict) for the given email, or (None, None)."""
    records, _ = _get_all_users()
    email = email.strip().lower()
    for i, row in enumerate(records):
        if str(row.get("email", "")).strip().lower() == email:
            return i + 2, row  # +2: header row is row 1, records are 0-indexed
    return None, None


# ---------------- Small helpers ----------------

def _gen_salt():
    return pysecrets.token_hex(8)


def _hash_password(password, salt):
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _gen_otp():
    return "".join(pysecrets.choice(string.digits) for _ in range(6))


def _gen_student_id():
    """Sequential, unique, human-readable student ID e.g. STU0001, STU0002 ..."""
    records, _ = _get_all_users()
    verified_count = sum(1 for r in records if _to_bool(r.get("verified")))
    return f"STU{verified_count + 1:04d}"


def email_exists(email):
    _, row = _find_user_row(email)
    return row is not None


# ---------------- Sign up ----------------

def start_signup(name, email, password):
    """
    Creates (or refreshes) a pending, unverified account row and returns the
    OTP that the caller should email to the user.

    Raises ValueError if the email already belongs to a *verified* account.
    """
    email = email.strip().lower()
    row_idx, row = _find_user_row(email)
    if row and _to_bool(row.get("verified")):
        raise ValueError("এই ইমেইল দিয়ে আগেই account আছে। Login ব্যবহার করো।")

    salt = _gen_salt()
    otp = _gen_otp()
    otp_expiry = (datetime.utcnow() + timedelta(minutes=OTP_VALID_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    password_hash = _hash_password(password, salt)
    ws = _users_ws()

    new_row = ["", name.strip(), email, password_hash, salt, False,
               otp, otp_expiry, "signup", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")]

    if row_idx:
        _with_retry(ws.update, f"A{row_idx}:J{row_idx}", [new_row])
    else:
        _with_retry(ws.append_row, new_row)
    return otp


def verify_signup_otp(email, otp):
    """
    Verifies the signup OTP. On success, assigns a unique student_id and
    activates the account. Returns (user_dict, error_message).
    """
    email = email.strip().lower()
    row_idx, row = _find_user_row(email)
    if not row:
        return None, "এই ইমেইলের জন্য কোনো pending signup পাওয়া যায়নি।"
    if _to_bool(row.get("verified")):
        return None, "এই account আগেই verified। Login করো।"
    if row.get("otp_purpose") != "signup":
        return None, "এই ইমেইলের জন্য কোনো signup verification pending নেই।"
    if str(row.get("otp_code", "")).strip() != str(otp).strip():
        return None, "ভুল OTP।"
    try:
        expiry = datetime.strptime(row["otp_expiry"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        expiry = datetime.min
    if datetime.utcnow() > expiry:
        return None, "OTP-এর মেয়াদ শেষ হয়ে গেছে। আবার Sign Up করো নতুন OTP পেতে।"

    user_id = _gen_student_id()
    ws = _users_ws()
    _with_retry(ws.update, f"A{row_idx}", [[user_id]])
    _with_retry(ws.update, f"F{row_idx}:I{row_idx}", [[True, "", "", ""]])
    return {"user_id": user_id, "name": row["name"], "email": email}, None


# ---------------- Login ----------------

def login(email, password):
    email = email.strip().lower()
    _, row = _find_user_row(email)
    if not row:
        return None, "এই ইমেইলে কোনো account পাওয়া যায়নি।"
    if not _to_bool(row.get("verified")):
        return None, "এই account এখনো verify করা হয়নি। Sign Up ট্যাবে গিয়ে OTP verify করো।"
    if _hash_password(password, row.get("salt", "")) != row.get("password_hash"):
        return None, "ভুল password।"
    return {"user_id": row["user_id"], "name": row["name"], "email": email}, None


# ---------------- Forgot password ----------------

def start_password_reset(email):
    email = email.strip().lower()
    row_idx, row = _find_user_row(email)
    if not row or not _to_bool(row.get("verified")):
        raise ValueError("এই ইমেইলে কোনো verified account পাওয়া যায়নি।")
    otp = _gen_otp()
    otp_expiry = (datetime.utcnow() + timedelta(minutes=OTP_VALID_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    ws = _users_ws()
    _with_retry(ws.update, f"G{row_idx}:I{row_idx}", [[otp, otp_expiry, "reset"]])
    return otp


def reset_password(email, otp, new_password):
    email = email.strip().lower()
    row_idx, row = _find_user_row(email)
    if not row:
        return False, "এই ইমেইলে কোনো account পাওয়া যায়নি।"
    if row.get("otp_purpose") != "reset":
        return False, "এই ইমেইলের জন্য কোনো password reset pending নেই।"
    if str(row.get("otp_code", "")).strip() != str(otp).strip():
        return False, "ভুল OTP।"
    try:
        expiry = datetime.strptime(row["otp_expiry"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        expiry = datetime.min
    if datetime.utcnow() > expiry:
        return False, "OTP-এর মেয়াদ শেষ হয়ে গেছে। আবার নতুন OTP চাও।"

    salt = _gen_salt()
    password_hash = _hash_password(new_password, salt)
    ws = _users_ws()
    _with_retry(ws.update, f"D{row_idx}:E{row_idx}", [[password_hash, salt]])
    _with_retry(ws.update, f"G{row_idx}:I{row_idx}", [["", "", ""]])
    return True, None
