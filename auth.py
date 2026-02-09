# auth.py — Firebase Email/Password login + kullanıcıya özel state

import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

# === Firebase Web Config (senin verdiğin) ===
firebase_config = {
    "apiKey": "AIzaSyDYnbR_a6Y3OgoK2FME0OoH7nGJRnLRSo4",
    "authDomain": "nurseplan1.firebaseapp.com",
    "projectId": "nurseplan1",
    "storageBucket": "nurseplan1.firebasestorage.app",
    "messagingSenderId": "710486389137",
    "appId": "1:710486389137:web:5e5b72768f35de85216a30",
}

# === Pyrebase init ===
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# === Firebase Admin (Firestore) ===
if not firebase_admin._apps:
    # Streamlit Cloud'ta servis hesabı json yoksa, test için anon init:
    firebase_admin.initialize_app()
db = firestore.client()

# === Session helpers ===
def is_logged_in() -> bool:
    return "user" in st.session_state and st.session_state.user is not None

def get_user_id() -> str:
    return st.session_state.user["localId"]

# === UI ===
def login_ui():
    st.title("NursePlan Giriş")

    tab1, tab2 = st.tabs(["Giriş", "Kayıt Ol"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Şifre", type="password")
        if st.button("Giriş Yap"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                st.success("Giriş başarılı")
                st.rerun()
            except Exception:
                st.error("Giriş başarısız")

    with tab2:
        email = st.text_input("Yeni email")
        password = st.text_input("Yeni şifre", type="password")
        if st.button("Kayıt Ol"):
            try:
                user = auth.create_user_with_email_and_password(email, password)
                st.success("Kayıt tamam. Şimdi giriş yap.")
            except Exception:
                st.error("Kayıt başarısız")

def logout_button():
    if st.sidebar.button("Çıkış"):
        st.session_state.user = None
        st.rerun()

# === Kullanıcıya özel veri ===
def save_user_state(data: dict):
    uid = get_user_id()
    db.collection("users").document(uid).set(data)

def load_user_state() -> dict:
    uid = get_user_id()
    doc = db.collection("users").document(uid).get()
    if doc.exists:
        return doc.to_dict()
    return {}
