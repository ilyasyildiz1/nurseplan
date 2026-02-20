# app.py — FULL (Shift/Staff rol kuralı + 1-2-3-4 + 0.5 saat + Excel/PDF rapor + ID menu)
import calendar
import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# =========================
# FIREBASE LOGIN + KULLANICIYA ÖZEL KAYIT (Firestore REST)
# - Email/Şifre ile giriş & kayıt
# - Her kullanıcıya özel "state_json" dokümanı
# =========================
import urllib.request

import pyrebase

# ===== YENİ FIREBASE (nurseplan2) =====
FIREBASE_API_KEY = "AIzaSyDpH641YjbdXY2xfGRs3DXRWceX-0Wg7-g"
FIREBASE_AUTH_DOMAIN = "nurseplan2.firebaseapp.com"
FIREBASE_PROJECT_ID = "nurseplan2"
FIREBASE_STORAGE_BUCKET = "nurseplan2.firebasestorage.app"
FIREBASE_MESSAGING_SENDER_ID = "850454086872"
FIREBASE_APP_ID = "1:850454086872:web:388cf1e85e4fe21a4b43ad"

_pyrebase_cfg = {
    "apiKey": FIREBASE_API_KEY,
    "authDomain": FIREBASE_AUTH_DOMAIN,
    "projectId": FIREBASE_PROJECT_ID,
    "storageBucket": FIREBASE_STORAGE_BUCKET,
    "messagingSenderId": FIREBASE_MESSAGING_SENDER_ID,
    "appId": FIREBASE_APP_ID,
    "databaseURL": ""  # boş kalabilir
}

_pb = pyrebase.initialize_app(_pyrebase_cfg)
_auth = _pb.auth()


def is_logged_in() -> bool:
    return "user" in st.session_state and st.session_state.user is not None

def _bearer_token() -> str:
    u = st.session_state.get("fb_user") or {}
    return u.get("idToken") or ""

def _uid() -> str:
    u = st.session_state.get("fb_user") or {}
    return u.get("localId") or ""

def login_ui():
    st.title("NursePlan Giriş")

    t1, t2 = st.tabs(["Giriş", "Kayıt Ol"])

    # 🔵 GİRİŞ TAB
    with t1:
        email = st.text_input("Email")
        pw = st.text_input("Şifre", type="password")

        if st.button("Giriş Yap"):
            try:
                user = _auth.sign_in_with_email_and_password(email, pw)
                st.session_state.user = None
                st.success("Giriş başarılı")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # 🔵 KAYIT TAB
    with t2:
        email = st.text_input("Yeni email", key="reg_email")
        pw = st.text_input("Yeni şifre", type="password", key="reg_pw")
        pw2 = st.text_input("Yeni şifre (tekrar)", type="password", key="reg_pw2")

        if st.button("Kayıt Ol", key="reg_btn"):

            if not email:
                st.error("Email gir")
            elif not pw:
                st.error("Şifre gir")
            elif len(pw) < 6:
                st.error("Şifre en az 6 karakter olmalı.")
            elif pw != pw2:
                st.error("Şifreler uyuşmuyor.")
            else:
                try:
                    _auth.create_user_with_email_and_password(email, pw)
                    st.success("Kayıt tamam ✅ Şimdi Giriş sekmesinden giriş yap.")
                except Exception as e:
                    st.error(str(e))




def logout_button():
    with st.sidebar:
        if st.button("Çıkış", key="logout_btn"):
            st.session_state.fb_user = None
            st.rerun()

def _firestore_doc_url(uid: str) -> str:
    return f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"

def save_user_state(payload: dict):
    uid = _uid()
    token = _bearer_token()
    if not uid or not token:
        return

    state_json = json.dumps(payload, ensure_ascii=False)
    body = {
        "fields": {
            "state_json": {"stringValue": state_json},
            "updated_at": {"timestampValue": datetime.utcnow().isoformat(timespec="seconds") + "Z"},
        }
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _firestore_doc_url(uid),
        data=data,
        method="PATCH",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception:
        pass

def load_user_state() -> dict:
    uid = _uid()
    token = _bearer_token()
    if not uid or not token:
        return {}

    req = urllib.request.Request(
        _firestore_doc_url(uid),
        method="GET",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        raw = urllib.request.urlopen(req, timeout=20).read()
        doc = json.loads(raw.decode("utf-8"))
        fields = doc.get("fields", {})
        sj = fields.get("state_json", {}).get("stringValue")
        if not sj:
            return {}
        return json.loads(sj)
    except Exception:
        return {}

import streamlit.components.v1 as components

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =========================
# OTOMATİK KALICI KAYIT (C)
# =========================
import os
from pathlib import Path

SAVE_DIR = Path.home() / "Documents" / "VardiyaApp"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = SAVE_DIR / "state.json"


def auto_save_state():
    try:
        payload = {
            "staff": [s.__dict__ for s in st.session_state.staff],
            "leaves": {k: [d.isoformat() for d in v] for k, v in st.session_state.leaves.items()},
            "rules": st.session_state.rules,
        }

        tmp_file = STATE_FILE.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        os.replace(tmp_file, STATE_FILE)
    except Exception:
        # sessiz geç (sistemi bozmasın)
        pass


def auto_load_state():
    try:
        if st.session_state.get("_loaded_once"):
            return

        if STATE_FILE.exists():
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)

            staff_list = []
            for item in payload.get("staff", []):
                name = item.get("name") or item.get("Ad Soyad")
                role = item.get("role") or item.get("Ünvan") or "Staff Hemşire"
                target = item.get("target_hours") or item.get("Aylık Saat") or 0
                avail = item.get("availability") or item.get("Uygunluk") or "both"
                night_cap = item.get("night_cap")

                staff_list.append(
                    Staff(
                        name=name,
                        role=role,
                        target_hours=int(target),
                        availability=avail,
                        night_cap=night_cap,
                    )
                )

            st.session_state.staff = staff_list

            leaves_dict = {}
            for name, days in payload.get("leaves", {}).items():
                leaves_dict[name] = set(date.fromisoformat(d) for d in days)
            st.session_state.leaves = leaves_dict

            st.session_state.rules = payload.get("rules", st.session_state.rules)

        st.session_state._loaded_once = True

    except Exception as e:
        # yükleme hatasını sessiz tut, sadece state'e yaz
        st.session_state.last_error = f"AUTOLOAD HATA: {type(e).__name__}: {e}"
        st.session_state._loaded_once = True


# =========================================================
# MODEL
# =========================================================
@dataclass
class Staff:
    name: str
    role: str                 # "Shift Hemşire" | "Staff Hemşire" | "Sorumlu"
    target_hours: int
    availability: str         # "day" | "night" | "both"
    night_cap: Optional[int] = None


# =========================================================
# DEFAULTS / STATE
# =========================================================
def default_rules() -> dict:
    return {
        "requirements": {
            "weekday": {"day": 3, "night": 2},
            "sat": {"day": 3, "night": 2},
            "sun": {"day": 2, "night": 2},
        },
        "shift_hours": {"day": 12.0, "night": 12.0},  # float: 0.5 adım destek
        "no_day_after_night": True,
        "max_consecutive_nights": 2,

        # 2 gece üst üste -> ertesi gün kesin boş (hard)
        "night_rest_after_2": True,

        # Yorgunluk/mesaiye göre blok izin
        "fatigue": {
            "enabled": True,
            "window_days": 4,        # geriye dönük bakılacak gün sayısı
            "rest1_hours": 36.0,     # >= ise 1 gün boş
            "rest2_hours": 48.0,     # >= ise 2 gün boş
            "pattern_ggnn_rest_days": 2,  # GGNN görülürse (4 gün) kaç gün boş
        },

        # Soft rol tercihi (yasak değil)
        "role_soft": {
            "enabled": True,
            "apply_to_day": True,
            "apply_to_night": True,
            # A seçeneği:
            "staff_staff_penalty": 100,  # büyük ceza
            "shift_shift_penalty": 5,    # küçük ceza
        },

        "responsible": {
            "enabled": True,
            "weekday_hours": 8.0,
            "sat_hours": 5.0,
            "sun_off": True,
        },
    }


def _migrate_rules(rules: dict) -> dict:
    if not isinstance(rules, dict):
        return default_rules()

    out = default_rules()

    if "requirements" in rules and isinstance(rules["requirements"], dict):
        out["requirements"] = rules["requirements"]
    elif "requirements_by_role" in rules and isinstance(rules["requirements_by_role"], dict):
        rbr = rules["requirements_by_role"]
        req = {}
        for dtype in ("weekday", "sat", "sun"):
            dd = rbr.get(dtype, {})
            day = int(dd.get("day_shift", 0)) + int(dd.get("day_staff", 0))
            night = int(dd.get("night_shift", 0)) + int(dd.get("night_staff", 0))
            req[dtype] = {"day": day, "night": night}
        out["requirements"] = req

    if "shift_hours" in rules and isinstance(rules["shift_hours"], dict):
        out["shift_hours"]["day"] = float(rules["shift_hours"].get("day", out["shift_hours"]["day"]))
        out["shift_hours"]["night"] = float(rules["shift_hours"].get("night", out["shift_hours"]["night"]))

    out["no_day_after_night"] = bool(rules.get("no_day_after_night", out["no_day_after_night"]))
    out["max_consecutive_nights"] = int(rules.get("max_consecutive_nights", out["max_consecutive_nights"]))
    out["night_rest_after_2"] = bool(rules.get("night_rest_after_2", out["night_rest_after_2"]))

    if "responsible" in rules and isinstance(rules["responsible"], dict):
        out["responsible"]["enabled"] = bool(rules["responsible"].get("enabled", out["responsible"]["enabled"]))
        out["responsible"]["weekday_hours"] = float(rules["responsible"].get("weekday_hours", out["responsible"]["weekday_hours"]))
        out["responsible"]["sat_hours"] = float(rules["responsible"].get("sat_hours", out["responsible"]["sat_hours"]))
        out["responsible"]["sun_off"] = bool(rules["responsible"].get("sun_off", out["responsible"]["sun_off"]))

    if "fatigue" in rules and isinstance(rules["fatigue"], dict):
        out["fatigue"]["enabled"] = bool(rules["fatigue"].get("enabled", out["fatigue"]["enabled"]))
        out["fatigue"]["window_days"] = int(rules["fatigue"].get("window_days", out["fatigue"]["window_days"]))
        out["fatigue"]["rest1_hours"] = float(rules["fatigue"].get("rest1_hours", out["fatigue"]["rest1_hours"]))
        out["fatigue"]["rest2_hours"] = float(rules["fatigue"].get("rest2_hours", out["fatigue"]["rest2_hours"]))
        out["fatigue"]["pattern_ggnn_rest_days"] = int(rules["fatigue"].get("pattern_ggnn_rest_days", out["fatigue"]["pattern_ggnn_rest_days"]))

    if "role_soft" in rules and isinstance(rules["role_soft"], dict):
        out["role_soft"]["enabled"] = bool(rules["role_soft"].get("enabled", out["role_soft"]["enabled"]))
        out["role_soft"]["apply_to_day"] = bool(rules["role_soft"].get("apply_to_day", out["role_soft"]["apply_to_day"]))
        out["role_soft"]["apply_to_night"] = bool(rules["role_soft"].get("apply_to_night", out["role_soft"]["apply_to_night"]))
        out["role_soft"]["staff_staff_penalty"] = int(rules["role_soft"].get("staff_staff_penalty", out["role_soft"]["staff_staff_penalty"]))
        out["role_soft"]["shift_shift_penalty"] = int(rules["role_soft"].get("shift_shift_penalty", out["role_soft"]["shift_shift_penalty"]))

    return out


def init_state():

    # ---- defaults ----
    if "staff" not in st.session_state:
        st.session_state.staff: List[Staff] = []
    if "leaves" not in st.session_state:
        st.session_state.leaves: Dict[str, Set[date]] = {}
    if "rules" not in st.session_state:
        st.session_state.rules = default_rules()
    else:
        st.session_state.rules = _migrate_rules(st.session_state.rules)

    if "auto_pack" not in st.session_state:
        st.session_state.auto_pack = None  # (matrix_df, day_detail_df)
    if "manual_matrix" not in st.session_state:
        st.session_state.manual_matrix = None

    if "status_msg" not in st.session_state:
        st.session_state.status_msg = ""
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    if "upload_success_msg" not in st.session_state:
        st.session_state.upload_success_msg = ""

    if "gen_y" not in st.session_state:
        st.session_state.gen_y = date.today().year
    if "gen_m" not in st.session_state:
        st.session_state.gen_m = date.today().month

    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None
    if "upload_in_progress" not in st.session_state:
        st.session_state.upload_in_progress = False

    if "active_page_id" not in st.session_state:
        st.session_state.active_page_id = "personel"

    # ---- Firebase'ten kullanıcıya özel state yükle (bir kez) ----
    if "fb_state_loaded" not in st.session_state:
        st.session_state.fb_state_loaded = True
        try:
            user_data = load_user_state()
            if isinstance(user_data, dict) and user_data:
                # staff: list[dict] -> List[Staff]
                staff_list: List[Staff] = []
                for item in user_data.get("staff", []) or []:
                    if isinstance(item, Staff):
                        staff_list.append(item)
                        continue
                    if not isinstance(item, dict):
                        continue
                    name = (item.get("name") or item.get("Ad Soyad") or "").strip()
                    if not name:
                        continue
                    role = item.get("role") or item.get("Ünvan") or "Staff Hemşire"
                    target = item.get("target_hours") or item.get("Aylık Saat") or 0
                    avail = item.get("availability") or item.get("Vardiya") or "both"
                    night_cap = item.get("night_cap")
                    try:
                        target_i = int(target)
                    except Exception:
                        target_i = 0
                    try:
                        night_cap_i = None if night_cap in (None, "", 0, "0") else int(night_cap)
                    except Exception:
                        night_cap_i = None
                    staff_list.append(
                        Staff(
                            name=name,
                            role=role,
                            target_hours=target_i,
                            availability=avail,
                            night_cap=night_cap_i,
                        )
                    )
                if staff_list:
                    st.session_state.staff = staff_list

                # leaves: {"isim": ["2026-02-01", ...]}
                leaves_in = user_data.get("leaves", {}) or {}
                leaves_out: Dict[str, Set[date]] = {}
                for k, vals in leaves_in.items():
                    s: Set[date] = set()
                    for v in (vals or []):
                        try:
                            s.add(pd.to_datetime(v).date())
                        except Exception:
                            pass
                    leaves_out[str(k)] = s
                if leaves_out:
                    st.session_state.leaves = leaves_out

                # rules
                st.session_state.rules = user_data.get("rules", st.session_state.rules) or st.session_state.rules
        except Exception:
            pass

    auto_load_state()



# =========================================================
# HELPERS
# =========================================================
def month_days(year: int, month: int) -> List[date]:
    first = date(year, month, 1)
    _, last_day = calendar.monthrange(year, month)
    return [first + timedelta(days=i) for i in range(last_day)]


def day_type(d: date) -> str:
    if d.weekday() == 5:
        return "sat"
    if d.weekday() == 6:
        return "sun"
    return "weekday"


def is_responsible(s: Staff) -> bool:
    return s.role.strip().lower() in {"sorumlu", "sorumlu hemşire", "sorumlu hemsire", "sorumlu hemsire"}


def is_shift_nurse(s: Staff) -> bool:
    return s.role.strip().lower() in {"shift hemşire", "shift hemsire", "shift"}


def is_staff_nurse(s: Staff) -> bool:
    return s.role.strip().lower() in {"staff hemşire", "staff hemsire", "staff"}


def role_tag(s: Staff) -> str:
    if is_shift_nurse(s):
        return "shift"
    if is_staff_nurse(s):
        return "staff"
    if is_responsible(s):
        return "responsible"
    return "other"


def is_available(s: Staff, shift: str) -> bool:
    if s.availability == "both":
        return True
    return (shift == "day" and s.availability == "day") or (shift == "night" and s.availability == "night")


def _cell(v) -> str:
    v = "" if v is None else str(v).strip().upper()
    return v if v in {"", "G", "N"} else ""


def build_need_score(name: str, totals_hours: Dict[str, float], targets: Dict[str, int]) -> float:
    return float(targets.get(name, 0) - totals_hours.get(name, 0.0))


# =========================================================
# EXPORT / IMPORT
# =========================================================
def state_to_json_bytes() -> bytes:
    staff_list = [asdict(s) for s in st.session_state.staff]
    leaves_dict = {name: sorted([d.isoformat() for d in days]) for name, days in st.session_state.leaves.items()}
    payload = {
        "version": 3,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "staff": staff_list,
        "leaves": leaves_dict,
        "rules": st.session_state.rules,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def load_state_from_json_bytes(data: bytes):
    text = data.decode("utf-8-sig")
    payload = json.loads(text)

    for k in ("staff", "leaves", "rules"):
        if k not in payload:
            raise ValueError(f"JSON içinde '{k}' alanı yok.")

    staff_objs: List[Staff] = []
    for item in payload["staff"]:
        staff_objs.append(
            Staff(
                name=str(item["name"]),
                role=str(item.get("role", "Staff Hemşire")),
                target_hours=int(item.get("target_hours", 0)),
                availability=str(item.get("availability", "both")),
                night_cap=item.get("night_cap", None),
            )
        )

    leaves_dict: Dict[str, Set[date]] = {}
    for name, iso_list in payload["leaves"].items():
        leaves_dict[name] = set(date.fromisoformat(x) for x in iso_list)

    st.session_state.staff = staff_objs
    st.session_state.leaves = leaves_dict
    st.session_state.rules = _migrate_rules(payload["rules"])

    st.session_state.auto_pack = None
    st.session_state.manual_matrix = None

    auto_save_state()


def sanitize_sheet_name(name: str) -> str:
    bad = ['\\', '/', '*', '[', ']', ':', '?']
    for b in bad:
        name = name.replace(b, " ")
    name = name.strip()
    return name[:31] if len(name) > 31 else name


def to_excel_bytes(
    matrix_df: pd.DataFrame,
    day_detail_df: pd.DataFrame,
    deficit_df: Optional[pd.DataFrame] = None,
    fairness_df: Optional[pd.DataFrame] = None,
    overtime_df: Optional[pd.DataFrame] = None,
) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        matrix_df.to_excel(writer, index=False, sheet_name="Cizelge_Matris")
        day_detail_df.to_excel(writer, index=False, sheet_name="Gun_Detay")
        if deficit_df is not None:
            deficit_df.to_excel(writer, index=False, sheet_name=sanitize_sheet_name("Eksik_Personel"))
        if fairness_df is not None:
            fairness_df.to_excel(writer, index=False, sheet_name=sanitize_sheet_name("Adalet_Denge"))
        if overtime_df is not None:
            overtime_df.to_excel(writer, index=False, sheet_name=sanitize_sheet_name("Fazla_Eksik"))
    return out.getvalue()


# =========================================================
# PDF (TR font)
# =========================================================
def ensure_tr_font():
    font_name = "TRFont"
    try:
        pdfmetrics.getFont(font_name)
        return font_name
    except Exception:
        pass
    pdfmetrics.registerFont(TTFont(font_name, "/System/Library/Fonts/Supplemental/AppleGothic.ttf"))
    return font_name


def _df_to_table_data(df: Optional[pd.DataFrame], max_rows: int = 35) -> List[List[str]]:
    if df is None or df.empty:
        return [["(Boş)"]]
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    d = d.fillna("")
    return [list(d.columns)] + d.astype(str).values.tolist()


def _styled_table(data: List[List[str]], font_name: str, col_widths: Optional[List[float]] = None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    return t


def matrix_to_pdf_bytes(
    matrix_df: pd.DataFrame,
    title: str,
    deficit_df: Optional[pd.DataFrame] = None,
    fairness_df: Optional[pd.DataFrame] = None,
    overtime_df: Optional[pd.DataFrame] = None,
) -> bytes:
    font_name = ensure_tr_font()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=8 * mm,
        rightMargin=8 * mm,
        topMargin=8 * mm,
        bottomMargin=8 * mm,
    )

    styles = getSampleStyleSheet()
    styles["Title"].fontName = font_name
    styles["Heading2"].fontName = font_name
    styles["Normal"].fontName = font_name

    story = [Paragraph(title, styles["Title"]), Spacer(1, 6)]

    data = [list(matrix_df.columns)] + matrix_df.fillna("").astype(str).values.tolist()
    ncols = len(matrix_df.columns)
    day_cols = ncols - 3  # Ad, Ünvan, days..., Toplam
    col_widths = [55 * mm, 35 * mm] + [6.5 * mm] * day_cols + [22 * mm]
    story.append(_styled_table(data, font_name, col_widths=col_widths))

    story.append(PageBreak())
    story.append(Paragraph("Raporlar", styles["Heading2"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Eksik Personel Uyarıları (ilk 35 satır)", styles["Normal"]))
    story.append(Spacer(1, 4))
    story.append(_styled_table(_df_to_table_data(deficit_df, max_rows=35), font_name))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Adalet / Denge Raporu (ilk 35 satır)", styles["Normal"]))
    story.append(Spacer(1, 4))
    story.append(_styled_table(_df_to_table_data(fairness_df, max_rows=35), font_name))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Fazla / Eksik Mesai Uyarıları (ilk 35 satır)", styles["Normal"]))
    story.append(Spacer(1, 4))
    story.append(_styled_table(_df_to_table_data(overtime_df, max_rows=35), font_name))

    doc.build(story)
    return buffer.getvalue()


# =========================================================
# TOTALS / VALIDATION
# =========================================================
def assignment_hours_for(staff_obj: Staff, rules: dict, d: date, cell_val: str) -> float:
    if cell_val == "":
        return 0.0
    if is_responsible(staff_obj):
        if cell_val != "G":
            return 0.0
        if day_type(d) == "sun" and rules["responsible"].get("sun_off", True):
            return 0.0
        return float(rules["responsible"].get("sat_hours", 5.0)) if day_type(d) == "sat" else float(rules["responsible"].get("weekday_hours", 8.0))
    if cell_val == "G":
        return float(rules["shift_hours"]["day"])
    if cell_val == "N":
        return float(rules["shift_hours"]["night"])
    return 0.0


def validate_manual_matrix(matrix_df: pd.DataFrame, year: int, month: int) -> List[str]:
    rules = st.session_state.rules
    staff_all: List[Staff] = st.session_state.staff
    leaves: Dict[str, Set[date]] = st.session_state.leaves
    staff_by_name = {s.name: s for s in staff_all}
    days = month_days(year, month)
    errors: List[str] = []

    for _, row in matrix_df.iterrows():
        name = row["Ad Soyad"]
        s = staff_by_name.get(name)
        leave_set = leaves.get(name, set())
        for d in days:
            v = _cell(row.get(str(d.day), ""))
            if v in {"G", "N"} and d in leave_set:
                errors.append(f"{name}: {d.isoformat()} izinliyken {v} yazılmış.")
                break
        if s and is_responsible(s):
            for d in days:
                if _cell(row.get(str(d.day), "")) == "N":
                    errors.append(f"{name} (Sorumlu): {d.isoformat()} tarihinde N yazılamaz.")
                    break

    max_consec = int(rules.get("max_consecutive_nights", 2))
    for _, row in matrix_df.iterrows():
        name = row["Ad Soyad"]
        s = staff_by_name.get(name)
        if s and is_responsible(s):
            continue
        streak = 0
        for d in days:
            v = _cell(row.get(str(d.day), ""))
            if v == "N":
                streak += 1
                if streak > max_consec:
                    errors.append(f"{name}: {d.isoformat()} ile birlikte üst üste {streak} gece (max {max_consec}).")
                    break
            else:
                streak = 0

    if rules.get("no_day_after_night", True):
        for _, row in matrix_df.iterrows():
            name = row["Ad Soyad"]
            s = staff_by_name.get(name)
            if s and is_responsible(s):
                continue
            for i in range(len(days) - 1):
                d0 = days[i]
                d1 = days[i + 1]
                if _cell(row.get(str(d0.day), "")) == "N" and _cell(row.get(str(d1.day), "")) == "G":
                    errors.append(f"{name}: {d0.isoformat()} N sonrası {d1.isoformat()} G olamaz.")
                    break

    if rules.get("night_rest_after_2", True):
        for _, row in matrix_df.iterrows():
            name = row["Ad Soyad"]
            s = staff_by_name.get(name)
            if s and is_responsible(s):
                continue
            streak = 0
            for i in range(len(days)):
                d = days[i]
                v = _cell(row.get(str(d.day), ""))
                if v == "N":
                    streak += 1
                else:
                    streak = 0
                if streak == 2 and i + 1 < len(days):
                    d_next = days[i + 1]
                    v_next = _cell(row.get(str(d_next.day), ""))
                    if v_next in {"G", "N"}:
                        errors.append(f"{name}: {d.isoformat()} ikinci N sonrası {d_next.isoformat()} boş olmalı.")
                        break

    return errors


def compute_totals_from_matrix(matrix_df: pd.DataFrame, year: int, month: int) -> pd.Series:
    rules = st.session_state.rules
    staff_all: List[Staff] = st.session_state.staff
    staff_by_name = {s.name: s for s in staff_all}
    days = month_days(year, month)

    totals: List[float] = []
    for _, row in matrix_df.iterrows():
        name = row["Ad Soyad"]
        s = staff_by_name.get(name)
        total = 0.0
        if s is None:
            totals.append(total)
            continue
        for d in days:
            val = _cell(row.get(str(d.day), ""))
            total += assignment_hours_for(s, rules, d, val)
        totals.append(total)

    return pd.Series(totals)


# =========================================================
# AUTO SCHEDULE (Soft Role + Fatigue)
# =========================================================
def generate_schedule(year: int, month: int):
    rules = st.session_state.rules
    req = rules["requirements"]
    staff_all: List[Staff] = st.session_state.staff
    leaves: Dict[str, Set[date]] = st.session_state.leaves

    if not staff_all:
        raise ValueError("Personel listesi boş.")

    responsibles = [s for s in staff_all if is_responsible(s)]
    normals = [s for s in staff_all if not is_responsible(s)]
    if not normals and any(req[d]["night"] > 0 for d in req):
        raise ValueError("Gece ihtiyacı var ama (Sorumlu dışı) personel yok.")

    staff_by_name: Dict[str, Staff] = {s.name: s for s in staff_all}

    totals_hours: Dict[str, float] = {s.name: 0.0 for s in staff_all}
    targets: Dict[str, int] = {s.name: int(s.target_hours) for s in staff_all}
    night_counts: Dict[str, int] = {s.name: 0 for s in staff_all}
    shift_counts: Dict[Tuple[str, str], int] = {}

    last_night_workers: Set[str] = set()
    consec_nights: Dict[str, int] = {s.name: 0 for s in normals}

    rest_days_left: Dict[str, int] = {s.name: 0 for s in normals}
    history: Dict[str, List[str]] = {s.name: [] for s in normals}

    days = month_days(year, month)
    day_assignments: List[dict] = []

    role_soft = rules.get("role_soft", {})
    rs_enabled = bool(role_soft.get("enabled", True))
    rs_day = bool(role_soft.get("apply_to_day", True))
    rs_night = bool(role_soft.get("apply_to_night", True))
    pen_staff = int(role_soft.get("staff_staff_penalty", 100))
    pen_shift = int(role_soft.get("shift_shift_penalty", 5))

    fat = rules.get("fatigue", {})
    fat_enabled = bool(fat.get("enabled", True))
    w_days = int(fat.get("window_days", 4))
    rest1 = float(fat.get("rest1_hours", 36.0))
    rest2 = float(fat.get("rest2_hours", 48.0))
    ggnn_rest = int(fat.get("pattern_ggnn_rest_days", 2))

    def blocked_today_set() -> Set[str]:
        return {n for n, rem in rest_days_left.items() if rem > 0}

    def can_work(s: Staff, d: date, shift: str, blocked_today: Set[str]) -> bool:
        if s.name in blocked_today:
            return False
        if d in leaves.get(s.name, set()):
            return False
        if not is_available(s, shift):
            return False
        if shift == "day" and rules.get("no_day_after_night", True) and s.name in last_night_workers:
            return False
        if shift == "night":
            max_consec = int(rules.get("max_consecutive_nights", 2))
            if consec_nights.get(s.name, 0) >= max_consec:
                return False
            cap = s.night_cap
            if cap is not None and night_counts.get(s.name, 0) >= cap:
                return False
        return True

    def role_penalty(candidate: Staff, current_assigned_names: List[str], shift: str) -> int:
        if not rs_enabled:
            return 0
        if shift == "day" and not rs_day:
            return 0
        if shift == "night" and not rs_night:
            return 0

        cand_role = role_tag(candidate)
        if cand_role not in {"staff", "shift"}:
            return 0

        counts = {"staff": 0, "shift": 0}
        for n in current_assigned_names:
            s0 = staff_by_name.get(n)
            if not s0:
                continue
            rt = role_tag(s0)
            if rt in counts:
                counts[rt] += 1

        if cand_role == "staff":
            return pen_staff * counts["staff"]
        if cand_role == "shift":
            return pen_shift * counts["shift"]
        return 0

    def sort_candidates(names: List[str], shift: str, current_assigned_names: List[str]) -> List[str]:
        def key(nm: str):
            need = build_need_score(nm, totals_hours, targets)
            nightc = night_counts.get(nm, 0)
            shcnt = shift_counts.get((nm, shift), 0)
            pen = role_penalty(staff_by_name[nm], current_assigned_names, shift)
            return (need, -nightc if shift == "night" else 0, -shcnt, -pen)

        names.sort(key=key, reverse=True)
        return names

    def suggest(pool: List[Staff], d: date, shift: str, blocked_today: Set[str], exclude: Set[str], current_assigned: List[str], k: int = 10) -> List[str]:
        names = []
        for s in pool:
            if s.name in exclude:
                continue
            if can_work(s, d, shift, blocked_today):
                names.append(s.name)
        return sort_candidates(names, shift, current_assigned)[:k]

    for d in days:
        dtype = day_type(d)
        blocked_today = blocked_today_set()

        day_need = int(req[dtype]["day"])
        night_need = int(req[dtype]["night"])

        day_assigned: List[str] = []
        night_assigned: List[str] = []

        resp_assigned = 0
        if rules.get("responsible", {}).get("enabled", True) and responsibles:
            for s in responsibles:
                if dtype == "sun" and rules["responsible"].get("sun_off", True):
                    continue
                if d in leaves.get(s.name, set()):
                    continue
                day_assigned.append(s.name)
                resp_assigned += 1
                totals_hours[s.name] += assignment_hours_for(s, rules, d, "G")
                shift_counts[(s.name, "day")] = shift_counts.get((s.name, "day"), 0) + 1

        remaining_day = max(0, day_need - resp_assigned)

        if remaining_day > 0:
            cand = [s.name for s in normals if can_work(s, d, "day", blocked_today)]
            cand = sort_candidates(cand, "day", day_assigned)
            for nm in cand:
                if remaining_day <= 0:
                    break
                if nm in day_assigned:
                    continue
                day_assigned.append(nm)
                totals_hours[nm] += float(rules["shift_hours"]["day"])
                shift_counts[(nm, "day")] = shift_counts.get((nm, "day"), 0) + 1
                remaining_day -= 1

        remaining_night = max(0, night_need)
        if remaining_night > 0:
            cand = [s.name for s in normals if can_work(s, d, "night", blocked_today)]
            cand = sort_candidates(cand, "night", night_assigned)
            for nm in cand:
                if remaining_night <= 0:
                    break
                if nm in night_assigned:
                    continue
                night_assigned.append(nm)
                totals_hours[nm] += float(rules["shift_hours"]["night"])
                night_counts[nm] = night_counts.get(nm, 0) + 1
                shift_counts[(nm, "night")] = shift_counts.get((nm, "night"), 0) + 1
                remaining_night -= 1

        for s in normals:
            consec_nights[s.name] = consec_nights.get(s.name, 0) + 1 if s.name in night_assigned else 0
        last_night_workers = set(night_assigned)

        for nm in list(blocked_today):
            if rest_days_left.get(nm, 0) > 0:
                rest_days_left[nm] -= 1

        if rules.get("night_rest_after_2", True):
            for nm in night_assigned:
                if consec_nights.get(nm, 0) >= 2:
                    rest_days_left[nm] = max(rest_days_left.get(nm, 0), 1)

        if fat_enabled:
            for s in normals:
                v = ""
                if s.name in night_assigned:
                    v = "N"
                elif s.name in day_assigned:
                    v = "G"
                history[s.name].append(v)
                if len(history[s.name]) > w_days:
                    history[s.name] = history[s.name][-w_days:]

            for s in normals:
                h = history.get(s.name, [])
                if not h:
                    continue

                if len(h) == 4 and h[0] == "G" and h[1] == "G" and h[2] == "N" and h[3] == "N":
                    if ggnn_rest > 0:
                        rest_days_left[s.name] = max(rest_days_left.get(s.name, 0), ggnn_rest)

                hours = 0.0
                for v in h:
                    if v == "G":
                        hours += float(rules["shift_hours"]["day"])
                    elif v == "N":
                        hours += float(rules["shift_hours"]["night"])

                if hours >= rest2:
                    rest_days_left[s.name] = max(rest_days_left.get(s.name, 0), 2)
                elif hours >= rest1:
                    rest_days_left[s.name] = max(rest_days_left.get(s.name, 0), 1)

        day_missing = max(0, remaining_day)
        night_missing = max(0, remaining_night)

        day_suggest = []
        night_suggest = []
        if day_missing > 0:
            day_suggest = suggest(normals, d, "day", blocked_today, set(day_assigned), day_assigned, k=10)
        if night_missing > 0:
            night_suggest = suggest(normals, d, "night", blocked_today, set(night_assigned), night_assigned, k=10)

        def role_counts(names: List[str]) -> Tuple[int, int]:
            sc = 0
            stc = 0
            for nm in names:
                so = staff_by_name.get(nm)
                if so is None:
                    continue
                if is_shift_nurse(so):
                    sc += 1
                elif is_staff_nurse(so):
                    stc += 1
            return sc, stc

        d_shift, d_staff = role_counts([n for n in day_assigned if n in staff_by_name and not is_responsible(staff_by_name[n])])
        n_shift, n_staff = role_counts(night_assigned)

        day_assignments.append(
            {
                "date": d,
                "dtype": dtype,
                "day_assigned": list(day_assigned),
                "night_assigned": list(night_assigned),
                "day_missing": day_missing,
                "night_missing": night_missing,
                "day_shift_count": d_shift,
                "day_staff_count": d_staff,
                "night_shift_count": n_shift,
                "night_staff_count": n_staff,
                "gunduz_oneri": ", ".join(day_suggest),
                "gece_oneri": ", ".join(night_suggest),
            }
        )

    matrix_map: Dict[str, Dict[int, str]] = {s.name: {} for s in staff_all}
    for item in day_assignments:
        dom = item["date"].day
        for nm in item["day_assigned"]:
            matrix_map[nm][dom] = "G"
        for nm in item["night_assigned"]:
            matrix_map[nm][dom] = "N"

    last_day = days[-1].day
    rows = []
    for s in staff_all:
        row = {"Ad Soyad": s.name, "Ünvan": s.role}
        for i in range(1, last_day + 1):
            row[str(i)] = matrix_map.get(s.name, {}).get(i, "")
        row["Toplam Saat"] = round(float(totals_hours.get(s.name, 0.0)), 2)
        rows.append(row)
    matrix_df = pd.DataFrame(rows)

    detail_rows = []
    for item in day_assignments:
        detail_rows.append(
            {
                "Tarih": item["date"].isoformat(),
                "Gün Tipi": item["dtype"],
                "Gündüz": ", ".join(item["day_assigned"]),
                "Gece": ", ".join(item["night_assigned"]),
                "Gündüz Eksik": int(item["day_missing"]),
                "Gece Eksik": int(item["night_missing"]),
                "Gündüz (Shift/Staff)": f"{item['day_shift_count']}/{item['day_staff_count']}",
                "Gece (Shift/Staff)": f"{item['night_shift_count']}/{item['night_staff_count']}",
                "Öneri Gündüz": item.get("gunduz_oneri", ""),
                "Öneri Gece": item.get("gece_oneri", ""),
            }
        )
    day_detail_df = pd.DataFrame(detail_rows)

    return matrix_df, day_detail_df


# =========================================================
# REPORTS
# =========================================================
def make_deficit_report(detail_df: pd.DataFrame) -> pd.DataFrame:
    df = detail_df.copy()
    for c in ["Gündüz Eksik", "Gece Eksik"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df = df[(df.get("Gündüz Eksik", 0) > 0) | (df.get("Gece Eksik", 0) > 0)]
    if "Tarih" in df.columns:
        try:
            df["_dt"] = pd.to_datetime(df["Tarih"])
            df = df.sort_values("_dt").drop(columns=["_dt"])
        except Exception:
            pass

    keep = [c for c in ["Tarih", "Gün Tipi", "Gündüz Eksik", "Gece Eksik", "Öneri Gündüz", "Öneri Gece"] if c in df.columns]
    return df[keep]


def make_fairness_report(final_matrix: pd.DataFrame, staff_list: List[Staff], year: int, month: int) -> pd.DataFrame:
    staff_by_name = {s.name: s for s in staff_list}
    days = month_days(year, month)
    day_cols = [str(d.day) for d in days]

    rows = []
    for _, r in final_matrix.iterrows():
        name = r["Ad Soyad"]
        role = r.get("Ünvan", "")
        g = 0
        n = 0
        for c in day_cols:
            v = _cell(r.get(c, ""))
            if v == "G":
                g += 1
            elif v == "N":
                n += 1

        total_hours = float(r.get("Toplam Saat", 0) or 0.0)
        target = int(staff_by_name.get(name).target_hours) if name in staff_by_name else 0
        diff = total_hours - float(target)
        pct = (total_hours / target * 100.0) if target > 0 else None

        rows.append(
            {
                "Ad Soyad": name,
                "Ünvan": role,
                "G Sayısı": g,
                "N Sayısı": n,
                "Toplam Saat": round(total_hours, 2),
                "Hedef Saat": target,
                "Fark (Saat)": round(diff, 2),
                "Hedefe Uyum %": (round(pct, 1) if pct is not None else None),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "Fark (Saat)" in df.columns:
        df = df.sort_values("Fark (Saat)", key=lambda s: s.abs(), ascending=False)
    return df


def make_overtime_report(fair_df: pd.DataFrame, over_thr: float = 10.0, under_thr: float = -10.0) -> pd.DataFrame:
    """
    Hedef saat farkına göre uyarı listesi.
    over_thr: hedefin üstüne +kaç saatten sonra "FAZLA"
    under_thr: hedefin altına -kaç saatten sonra "EKSİK" (negatif)
    """
    if fair_df is None or fair_df.empty:
        return pd.DataFrame(columns=["Ad Soyad", "Ünvan", "Toplam Saat", "Hedef Saat", "Fark (Saat)", "Durum"])

    df = fair_df.copy()
    df["Fark (Saat)"] = pd.to_numeric(df["Fark (Saat)"], errors="coerce").fillna(0.0)

    def _status(x: float) -> str:
        if x >= over_thr:
            return "FAZLA"
        if x <= under_thr:
            return "EKSİK"
        return "OK"

    df["Durum"] = df["Fark (Saat)"].apply(_status)
    df = df[df["Durum"] != "OK"].copy()

    df["_abs"] = df["Fark (Saat)"].abs()
    df = df.sort_values("_abs", ascending=False).drop(columns=["_abs"])

    keep = ["Ad Soyad", "Ünvan", "Toplam Saat", "Hedef Saat", "Fark (Saat)", "Durum"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Vardiya / Nöbet Planlayıcı", layout="wide")
# === LOGIN GATE ===
if not is_logged_in():
    login_ui()
    st.stop()

logout_button()

init_state()

components.html(
    """
    <script>
      (function() {
        try {
          document.documentElement.setAttribute("lang", "tr");
          document.documentElement.setAttribute("dir", "ltr");
        } catch(e) {}
      })();
    </script>
    <style>
      html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"] { visibility: hidden !important; height: 0 !important; }
      [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "Dosyayı buraya sürükleyip bırakın.";
        visibility: visible !important;
        display: block !important;
        height: auto !important;
        margin-top: 8px !important;
      }
      [data-testid="stFileUploaderDropzone"] button[kind="secondary"] { position: relative !important; }
      [data-testid="stFileUploaderDropzone"] button[kind="secondary"] span { visibility: hidden !important; }
      [data-testid="stFileUploaderDropzone"] button[kind="secondary"]::after {
        content: "Dosyalara göz at";
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        white-space: nowrap;
      }
    </style>
    """,
    height=0,
)

st.title("Vardiya / Nöbet Planlayıcı")

# mesajlar
if st.session_state.upload_success_msg:
    st.success(st.session_state.upload_success_msg)
    st.session_state.upload_success_msg = ""

st.markdown("### Şablon")
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    st.download_button(
        "Şablonu indir (JSON)",
        data=state_to_json_bytes(),
        file_name="vardiya_sablon.json",
        mime="application/json",
    )

with c2:
    up = st.file_uploader("Şablon yükle (JSON)", type=["json"], key="uploader_json")
    if up is not None:
        if st.session_state.last_uploaded_name != up.name and not st.session_state.upload_in_progress:
            st.session_state.upload_in_progress = True
            try:
                raw = up.getvalue()
                load_state_from_json_bytes(raw)
                st.session_state.last_uploaded_name = up.name
                st.session_state.last_error = ""
                st.session_state.upload_success_msg = f"Şablon aktarıldı ✅ ({up.name})"
            except Exception as e:
                st.session_state.last_error = f"{type(e).__name__}: {e}"
            finally:
                st.session_state.upload_in_progress = False
            st.rerun()

    if st.session_state.last_error:
        st.error("Yükleme hatası:")
        st.code(st.session_state.last_error)

with c3:
    if st.button("Tüm veriyi sıfırla", type="secondary"):
        st.session_state.clear()
        st.rerun()

st.divider()

PAGE_IDS = ["personel", "izin", "kurallar", "cizelge"]
PAGE_LABELS = {
    "personel": "👤 Personel",
    "izin": "🗓️ İzinler",
    "kurallar": "⚙️ Kurallar",
    "cizelge": "📅 Çizelge",
}

if st.session_state.active_page_id not in PAGE_IDS:
    st.session_state.active_page_id = "personel"

st.session_state.active_page_id = st.radio(
    "Sayfa",
    PAGE_IDS,
    index=PAGE_IDS.index(st.session_state.active_page_id),
    format_func=lambda pid: PAGE_LABELS.get(pid, pid),
    horizontal=True,
    key="active_page_radio_id",
)

st.divider()

# =========================================================
# PAGE - PERSONEL
# =========================================================
if st.session_state.active_page_id == "personel":
    st.subheader("Personel Ekle / Yönet")

    with st.form("add_staff_form", clear_on_submit=True):
        a, b, c, d, e = st.columns([2, 2, 1, 1, 1])
        nm = a.text_input("Ad Soyad")
        rl = b.selectbox("Ünvan", ["Staff Hemşire", "Shift Hemşire", "Sorumlu"], index=0)
        th = c.number_input("Aylık hedef saat", min_value=0, value=180, step=1)
        av = d.selectbox(
            "Çalışabilir vardiya",
            ["both", "day", "night"],
            format_func=lambda x: {"both": "İkisi", "day": "Gündüz", "night": "Gece"}[x],
        )
        nc = e.number_input("Aylık max gece (ops.)", min_value=0, value=0, step=1)
        ok_add_staff = st.form_submit_button("Ekle")

    if ok_add_staff:
        nm = nm.strip()
        if not nm:
            st.error("Ad Soyad boş olamaz.")
        elif any(s.name == nm for s in st.session_state.staff):
            st.error("Bu isim zaten var. (Aynı isim olmasın)")
        else:
            cap = None if int(nc) == 0 else int(nc)
            st.session_state.staff.append(
                Staff(name=nm, role=rl, target_hours=int(th), availability=av, night_cap=cap)
            )
            st.session_state.leaves.setdefault(nm, set())
            auto_save_state()

            # Firebase'e kaydet (senin eklediğin blok varsa burada kalsın)
            try:
                save_user_state({
                    "staff": [s.__dict__ for s in st.session_state.staff],
                    "leaves": {k: [d.isoformat() for d in v] for k, v in st.session_state.leaves.items()},
                    "rules": st.session_state.rules,
                })
            except Exception:
                pass

            st.success(f"Eklendi: {nm}")





    st.divider()
    st.subheader("Personel Listesi")
    if st.session_state.staff:
        df = pd.DataFrame([asdict(s) for s in st.session_state.staff])
        df["availability"] = df["availability"].map({"both": "İkisi", "day": "Gündüz", "night": "Gece"})
        st.dataframe(df, use_container_width=True)

        del_nm = st.selectbox("Silinecek personel", [""] + [s.name for s in st.session_state.staff])
        if st.button("Seçili personeli sil") and del_nm:
            st.session_state.staff = [s for s in st.session_state.staff if s.name != del_nm]
            st.session_state.leaves.pop(del_nm, None)

            auto_save_state()

            # 🔴 FIREBASE KAYDET
            try:
                save_user_state({
                    "staff": [s.__dict__ for s in st.session_state.staff],
                    "leaves": {k: [d.isoformat() for d in v] for k, v in st.session_state.leaves.items()},
                    "rules": st.session_state.rules,
                })
            except Exception:
                pass

            st.success("Silindi.")


# =========================================================
# PAGE - İZİN
# =========================================================
elif st.session_state.active_page_id == "izin":
    st.subheader("İzin Girişi")
    if not st.session_state.staff:
        st.info("Önce personel ekleyin.")
    else:
        c1_, c2_ = st.columns(2)
        y = c1_.number_input("Yıl", min_value=2020, max_value=2100, value=int(st.session_state.gen_y), step=1)
        m = c2_.number_input("Ay", min_value=1, max_value=12, value=int(st.session_state.gen_m), step=1)

        who = st.selectbox("Personel", [s.name for s in st.session_state.staff])
        days = month_days(int(y), int(m))
        picked = st.multiselect("İzin günleri seç", days, format_func=lambda d: d.isoformat())

        if st.button("İzinleri kaydet"):
            st.session_state.leaves[who] = set(picked)
            auto_save_state()
            st.success("İzinler kaydedildi.")

        st.write("Mevcut izinler:")
        st.write(sorted([d.isoformat() for d in st.session_state.leaves.get(who, set())]))


# =========================================================
# PAGE - KURALLAR
# =========================================================
elif st.session_state.active_page_id == "kurallar":
    st.subheader("Kurallar / İhtiyaç")

    rules = st.session_state.rules
    req = rules["requirements"]

    st.markdown("### Gün tipine göre ihtiyaç (toplam kişi)")
    st.caption("Not: Sorumlu hemşire açıksa, gündüz ihtiyacına dahil edilir ve otomatik yazılır.")

    def req_editor(label: str, key: str):
        a, b = st.columns(2)
        req[key]["day"] = int(a.slider(f"{label} - Gündüz kişi", 0, 30, int(req[key]["day"]), 1))
        req[key]["night"] = int(b.slider(f"{label} - Gece kişi", 0, 30, int(req[key]["night"]), 1))

    req_editor("Hafta içi", "weekday")
    req_editor("Cumartesi", "sat")
    req_editor("Pazar", "sun")

    st.markdown("### Vardiya saatleri (0.5 adım)")
    a, b = st.columns(2)
    rules["shift_hours"]["day"] = float(a.number_input("Gündüz saat", min_value=0.0, value=float(rules["shift_hours"]["day"]), step=0.5))
    rules["shift_hours"]["night"] = float(b.number_input("Gece saat", min_value=0.0, value=float(rules["shift_hours"]["night"]), step=0.5))

    st.markdown("### Gece kuralları (hard)")
    rules["max_consecutive_nights"] = int(st.number_input("Üst üste max gece", min_value=1, value=int(rules.get("max_consecutive_nights", 2)), step=1))
    rules["no_day_after_night"] = st.checkbox("Gece sonrası ertesi gün gündüz yazma", value=bool(rules.get("no_day_after_night", True)))
    rules["night_rest_after_2"] = st.checkbox("2 gece sonrası ertesi gün kesin boş", value=bool(rules.get("night_rest_after_2", True)))

    st.markdown("### Soft rol tercihi (yasak değil)")
    rs = rules.get("role_soft", {})
    rs["enabled"] = st.checkbox("Rol karışımı tercih et (Staff-Staff kaçın)", value=bool(rs.get("enabled", True)))
    c1a, c2a, c3a, c4a = st.columns([1, 1, 1, 1])
    rs["apply_to_day"] = c1a.checkbox("Gündüz uygula", value=bool(rs.get("apply_to_day", True)))
    rs["apply_to_night"] = c2a.checkbox("Gece uygula", value=bool(rs.get("apply_to_night", True)))
    rs["staff_staff_penalty"] = int(c3a.number_input("Staff-Staff ceza", min_value=0, value=int(rs.get("staff_staff_penalty", 100)), step=1))
    rs["shift_shift_penalty"] = int(c4a.number_input("Shift-Shift ceza", min_value=0, value=int(rs.get("shift_shift_penalty", 5)), step=1))
    rules["role_soft"] = rs

    st.markdown("### Yorgunluk / Mesai (blok izin)")
    fat = rules.get("fatigue", {})
    fat["enabled"] = st.checkbox("Yorgunluk kurallarını uygula", value=bool(fat.get("enabled", True)))
    cfa, cfb, cfc, cfd = st.columns(4)
    fat["window_days"] = int(cfa.selectbox("Pencere (gün)", [3, 4, 5, 6], index=[3, 4, 5, 6].index(int(fat.get("window_days", 4)))))
    fat["rest1_hours"] = float(cfb.number_input("≥ saat ise 1 gün boş", min_value=0.0, value=float(fat.get("rest1_hours", 36.0)), step=0.5))
    fat["rest2_hours"] = float(cfc.number_input("≥ saat ise 2 gün boş", min_value=0.0, value=float(fat.get("rest2_hours", 48.0)), step=0.5))
    fat["pattern_ggnn_rest_days"] = int(cfd.number_input("GGNN sonrası boş gün", min_value=0, value=int(fat.get("pattern_ggnn_rest_days", 2)), step=1))
    rules["fatigue"] = fat

    st.markdown("### Sorumlu Hemşire (sabit)")
    resp = rules.get("responsible", {})
    resp["enabled"] = st.checkbox("Sorumlu hemşire sabit çalışsın", value=bool(resp.get("enabled", True)))
    a, b, c = st.columns(3)
    resp["weekday_hours"] = float(a.number_input("Hafta içi saat (08-17)", min_value=0.0, value=float(resp.get("weekday_hours", 8.0)), step=0.5))
    resp["sat_hours"] = float(b.number_input("Cumartesi saat (08-13)", min_value=0.0, value=float(resp.get("sat_hours", 5.0)), step=0.5))
    resp["sun_off"] = c.checkbox("Pazar izin", value=bool(resp.get("sun_off", True)))
    rules["responsible"] = resp

    # requirements güncel
    rules["requirements"] = req

    # state her etkileşimde güncel kalsın (migrate ile)
    st.session_state.rules = _migrate_rules(rules)

    if st.button("Kuralları Kaydet", type="primary"):
        auto_save_state()
        st.success("Kurallar kaydedildi ✅")


# =========================================================
# PAGE - ÇİZELGE
# =========================================================
elif st.session_state.active_page_id == "cizelge":
    st.subheader("Çizelge Oluştur / Manuel Düzenle")

    if not st.session_state.staff:
        st.info("Önce personel ekleyin.")
    else:
        st.markdown("### Yeni Çizelge Oluştur")
        with st.form("generate_form"):
            c1__, c2__ = st.columns(2)
            gen_y = c1__.number_input("Yıl", min_value=2020, max_value=2100, value=int(st.session_state.gen_y), step=1)
            gen_m = c2__.number_input("Ay", min_value=1, max_value=12, value=int(st.session_state.gen_m), step=1)
            submitted = st.form_submit_button("Çizelge Oluştur")

        if submitted:
            st.session_state.active_page_id = "cizelge"
            st.session_state.gen_y = int(gen_y)
            st.session_state.gen_m = int(gen_m)
            try:
                matrix_df, day_detail_df = generate_schedule(int(gen_y), int(gen_m))
                st.session_state.auto_pack = (matrix_df, day_detail_df)
                st.session_state.manual_matrix = matrix_df.copy(deep=True)
                st.session_state.status_msg = "Otomatik çizelge üretildi ✅"
                st.session_state.last_error = ""
                st.rerun()
            except Exception as e:
                st.session_state.auto_pack = None
                st.session_state.manual_matrix = None
                st.session_state.status_msg = ""
                st.session_state.last_error = f"{type(e).__name__}: {e}"

        if st.session_state.last_error:
            st.error("Oluşturma hatası:")
            st.code(st.session_state.last_error)

        if st.session_state.status_msg:
            st.success(st.session_state.status_msg)

        if st.session_state.auto_pack is None or st.session_state.manual_matrix is None:
            st.info("Henüz çizelge yok.")
        else:
            y = int(st.session_state.gen_y)
            m = int(st.session_state.gen_m)
            auto_matrix, detail_df = st.session_state.auto_pack

            days_list = month_days(y, m)
            last_day = days_list[-1].day
            day_cols = [str(i) for i in range(1, last_day + 1)]
            col_config = {c: st.column_config.SelectboxColumn(c, options=["", "G", "N"], width="small") for c in day_cols}

            st.markdown("### Manuel Düzenleme (G / N / boş)")
            with st.form("manual_form", clear_on_submit=False):
                edited = st.data_editor(
                    st.session_state.manual_matrix,
                    use_container_width=True,
                    height=520,
                    column_config=col_config,
                    disabled=["Ad Soyad", "Ünvan", "Toplam Saat"],
                    key="editor",
                )
                a, b = st.columns(2)
                save = a.form_submit_button("Kaydet")
                reset = b.form_submit_button("Otomatiğe Sıfırla")

            if reset:
                st.session_state.active_page_id = "cizelge"
                st.session_state.manual_matrix = auto_matrix.copy(deep=True)
                st.session_state.status_msg = "Otomatiğe sıfırlandı ✅"
                st.rerun()

            if save:
                st.session_state.active_page_id = "cizelge"
                for c in day_cols:
                    edited[c] = edited[c].apply(_cell)
                errs = validate_manual_matrix(edited, y, m)
                if errs:
                    st.error("Kaydedilemedi:")
                    for e in errs[:15]:
                        st.write("• " + e)
                    if len(errs) > 15:
                        st.write(f"• (+{len(errs)-15} hata daha)")
                else:
                    edited["Toplam Saat"] = compute_totals_from_matrix(edited, y, m).round(2)
                    st.session_state.manual_matrix = edited.copy(deep=True)
                    st.session_state.status_msg = "Kaydedildi ✅"
                    st.rerun()

            final_matrix = st.session_state.manual_matrix

            st.markdown("### Eksik Personel Uyarıları")
            deficit_df = make_deficit_report(detail_df)
            if deficit_df.empty:
                st.success("Eksik personel yok ✅")
            else:
                st.warning("Eksik olan günler listelendi (öneriler dahil).")
                st.dataframe(deficit_df, use_container_width=True, height=240)

                st.markdown("### Eksik Günler için Detaylı Öneri Listesi")
                for _, r in deficit_df.iterrows():
                    tarih = r.get("Tarih", "")
                    geksik = int(r.get("Gündüz Eksik", 0) or 0)
                    neksik = int(r.get("Gece Eksik", 0) or 0)
                    og = r.get("Öneri Gündüz", "")
                    on = r.get("Öneri Gece", "")

                    with st.expander(f"{tarih} | Gündüz eksik: {geksik} | Gece eksik: {neksik}"):
                        if geksik > 0:
                            st.write("**Gündüz için önerilen adaylar (öncelik sırasıyla):**")
                            st.write(og if og else "Öneri yok (uygun aday bulunamadı).")
                        if neksik > 0:
                            st.write("**Gece için önerilen adaylar (öncelik sırasıyla):**")
                            st.write(on if on else "Öneri yok (uygun aday bulunamadı).")

            st.markdown("### Adalet / Denge Raporu")
            fair_df = make_fairness_report(final_matrix, st.session_state.staff, y, m)
            st.dataframe(fair_df, use_container_width=True, height=320)

            st.markdown("### Fazla / Eksik Mesai Uyarıları")
            ot_df = make_overtime_report(fair_df, over_thr=10.0, under_thr=-10.0)
            if ot_df.empty:
                st.success("Fazla/eksik mesai uyarısı yok ✅")
            else:
                st.dataframe(ot_df, use_container_width=True, height=240)

            st.markdown("### Güncel Matris")
            st.dataframe(final_matrix, use_container_width=True, height=520)

            excel_bytes = to_excel_bytes(final_matrix, detail_df, deficit_df=deficit_df, fairness_df=fair_df, overtime_df=ot_df)
            st.download_button(
                "Excel indir (raporlar dahil)",
                data=excel_bytes,
                file_name=f"vardiya_{y}_{m:02d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            pdf_bytes = matrix_to_pdf_bytes(
                final_matrix,
                f"Vardiya Çizelgesi - {y}-{m:02d}",
                deficit_df=deficit_df,
                fairness_df=fair_df,
                overtime_df=ot_df,
            )
            st.download_button(
                "PDF indir (raporlar dahil)",
                data=pdf_bytes,
                file_name=f"vardiya_{y}_{m:02d}.pdf",
                mime="application/pdf",
            )

            with st.expander("Gün Detayı (ham)"):
                st.dataframe(detail_df, use_container_width=True)
