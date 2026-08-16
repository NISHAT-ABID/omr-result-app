"""
auth.py
-------
Login / sign-up / "remember this device" logic for the OMR Result App.

- Passwords are never stored as plain text - only a salted SHA-256 hash
  is saved in the Users sheet.
- "Remember this device" works with a random session token: it's saved
  both in the Sessions worksheet (Google Sheets) and in a browser cookie
  (via streamlit-cookies-controller). On every fresh page load we check
  the cookie -> look the token up in the sheet -> if it's valid and not
  expired, the user is logged back in automatically, no password needed.
  Logging out deletes both the cookie and the sheet row.
"""

import hashlib
import time
import secrets as pysecrets
from datetime import timedelta

import streamlit as st

import sheets_helper as sh

try:
    from streamlit_cookies_controller import CookieController
    _COOKIES_AVAILABLE = True
except Exception:
    _COOKIES_AVAILABLE = False

REMEMBER_DAYS = 30
COOKIE_NAME = "omr_remember_token"


def _pepper():
    # Extra secret mixed into every password hash. Works fine without it,
    # but setting SESSION_SECRET in secrets.toml makes hashes harder to
    # attack if the Google Sheet itself ever leaked.
    return st.secrets.get("SESSION_SECRET", "omr-app-default-pepper")


def hash_password(password: str) -> str:
    salted = f"{password}{_pepper()}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


@st.cache_resource(show_spinner=False)
def _cookie_controller():
    if not _COOKIES_AVAILABLE:
        return None
    try:
        return CookieController()
    except Exception:
        return None


def _get_cookie(name):
    ctrl = _cookie_controller()
    if ctrl is None:
        return None
    try:
        return ctrl.get(name)
    except Exception:
        return None


def _set_cookie(name, value, days):
    ctrl = _cookie_controller()
    if ctrl is None:
        return
    try:
        ctrl.set(name, value, max_age=days * 24 * 60 * 60)
    except Exception:
        pass


def _remove_cookie(name):
    ctrl = _cookie_controller()
    if ctrl is None:
        return
    try:
        ctrl.remove(name)
    except Exception:
        pass


# ---------------- Sign up ----------------

def create_account(user_id, name, password, role):
    """Returns (True, "") on success or (False, "error message in Bangla")."""
    user_id = (user_id or "").strip()
    name = (name or "").strip()
    if not user_id or not name or not password:
        return False, "সব ঘর পূরণ করো।"
    if len(password) < 4:
        return False, "Password কমপক্ষে ৪ অক্ষরের হতে হবে।"
    if sh.user_exists(user_id):
        return False, "এই ID আগে থেকেই ব্যবহার হয়েছে, অন্য একটা ID দাও।"
    sh.create_user(user_id, name, hash_password(password), role)
    return True, ""


# ---------------- Login ----------------

def login(user_id, password):
    """Returns the user record (dict) on success, or None."""
    user = sh.get_user((user_id or "").strip())
    if not user:
        return None
    if not verify_password(password or "", user.get("password_hash", "")):
        return None
    return user


def start_session(user, remember=True):
    """Logs the user into st.session_state and optionally remembers the device."""
    st.session_state["authed"] = True
    st.session_state["user_id"] = user["user_id"]
    st.session_state["name"] = user["name"]
    st.session_state["role"] = user["role"]

    if remember:
        token = pysecrets.token_hex(24)
        expires = sh.now_bd() + timedelta(days=REMEMBER_DAYS)
        sh.create_session(token, user["user_id"], expires.strftime("%Y-%m-%d %H:%M:%S"))
        st.session_state["_session_token"] = token
        _set_cookie(COOKIE_NAME, token, REMEMBER_DAYS)


def try_auto_login():
    """
    Call once near the top of the app, before showing the login page.
    If a valid "remember me" cookie exists, logs the user back in
    silently. Returns True if the user is (now) logged in.
    """
    if st.session_state.get("authed"):
        return True
    if st.session_state.get("_auto_login_checked"):
        return False

    token = _get_cookie(COOKIE_NAME)

    # streamlit-cookies-controller reads cookies via a browser component,
    # which isn't ready on the very first script run. Give it a couple of
    # quick reruns before giving up and showing the login page.
    if token is None:
        retries = st.session_state.get("_auto_login_retries", 0)
        if retries < 2:
            st.session_state["_auto_login_retries"] = retries + 1
            time.sleep(0.2)
            st.rerun()
        st.session_state["_auto_login_checked"] = True
        return False

    st.session_state["_auto_login_checked"] = True

    session = sh.get_session(token)
    if not session:
        return False

    user = sh.get_user(session["user_id"])
    if not user:
        return False

    st.session_state["authed"] = True
    st.session_state["user_id"] = user["user_id"]
    st.session_state["name"] = user["name"]
    st.session_state["role"] = user["role"]
    st.session_state["_session_token"] = token
    return True


def logout():
    token = st.session_state.get("_session_token") or _get_cookie(COOKIE_NAME)
    if token:
        sh.delete_session(token)
    _remove_cookie(COOKIE_NAME)
    for key in ["authed", "user_id", "name", "role", "_session_token",
                "_auto_login_checked", "_auto_login_retries"]:
        st.session_state.pop(key, None)
