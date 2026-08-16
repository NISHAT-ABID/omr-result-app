"""
email_helper.py
----------------
Sends OTP verification / password-reset emails via SMTP.

Needs these keys in .streamlit/secrets.toml (see secrets_example.toml):

    SMTP_HOST = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USER = "youraddress@gmail.com"
    SMTP_PASS = "your-16-character-gmail-app-password"
    SMTP_FROM_NAME = "OMR Result App"

If you use Gmail: you must create an "App Password" from your Google
Account -> Security -> 2-Step Verification -> App passwords. Your normal
Gmail password will NOT work here.
"""

import smtplib
from email.mime.text import MIMEText

import streamlit as st

SUBJECTS = {
    "signup": "Your OMR Result App verification code",
    "reset": "Your OMR Result App password reset code",
}


def send_otp_email(to_email, otp, purpose="signup"):
    host = st.secrets.get("SMTP_HOST")
    port = int(st.secrets.get("SMTP_PORT", 587))
    user = st.secrets.get("SMTP_USER")
    pw = st.secrets.get("SMTP_PASS")
    from_name = st.secrets.get("SMTP_FROM_NAME", "OMR Result App")

    if not host or not user or not pw:
        raise RuntimeError("SMTP is not configured in secrets.toml (SMTP_HOST/SMTP_USER/SMTP_PASS missing).")

    if purpose == "reset":
        body = (
            "You requested to reset your OMR Result App password.\n\n"
            f"Your OTP code is: {otp}\n\n"
            "This code is valid for 10 minutes. If you didn't request this, "
            "you can safely ignore this email."
        )
    else:
        body = (
            "Welcome to OMR Result App!\n\n"
            f"Your verification OTP code is: {otp}\n\n"
            "Enter this code in the app to activate your account. "
            "This code is valid for 10 minutes."
        )

    msg = MIMEText(body)
    msg["Subject"] = SUBJECTS.get(purpose, "Your OTP code")
    msg["From"] = f"{from_name} <{user}>"
    msg["To"] = to_email

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, pw)
        server.sendmail(user, [to_email], msg.as_string())
