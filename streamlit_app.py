# streamlit_app_final.py
"""
Integrated final app: merges v5 (prediction, SHAP, PDF, Twilio SMS) and v6 (multi-user, registration,
profiles, feedback, admin, per-user history) into one Streamlit application.

Place this file in the root of your project (e.g., D:\Website\).
Ensure saved_models/rf_pipeline.pkl and vehicle_breakdown_prediction_dataset.xlsx exist.
Set Twilio credentials as environment variables if you want SMS:
  TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO (optional fallback)
Optional libs: shap, fpdf, twilio, bcrypt (app still runs without them).
"""

import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import uuid
import hashlib
import sqlite3
import base64








# ---------- Session State Defaults ----------
defaults = {
    "logged_in": False,
    "username": "",
    "role": "user",
    "phone": "",
    "email": ""
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Optional imports handled safely:
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# Twilio optional import
TWILIO_AVAILABLE = False
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    # Try short-target import if user used pip --target workaround
    # If user used custom path like C:\twilio_pkg, they should have added to sys.path before running.
    TWILIO_AVAILABLE = False

# bcrypt optional
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    BCRYPT_AVAILABLE = False

# Streamlit config
st.set_page_config(page_title="Vehicle Breakdown Predictor - Integrated", layout="wide")

# ----------------- CONFIG PATHS -----------------
PROJECT_ROOT = Path(".")
MODEL_PATH = PROJECT_ROOT / "rf_pipeline.pkl"
DATA_PATH = PROJECT_ROOT / "vehicle_breakdown_prediction_dataset.csv"
USERS_CSV = PROJECT_ROOT / "users.csv"
FEEDBACK_CSV = PROJECT_ROOT / "feedback.csv"
PREDICTION_LOG = PROJECT_ROOT / "prediction_log.csv"
AUTO_ALERT_LOG = PROJECT_ROOT / "auto_alert_log.csv"

SMS_THRESHOLD = float(os.getenv("SMS_THRESHOLD", 0.8))


# ----------------- Mobile-style Dark Mode (paste right after st.set_page_config) -----------------
# Remove any older dark-mode CSS first so there are no conflicts.
def add_bg_from_local(image_file):
    import base64
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpg")





# Minimal polished dark CSS — paste after st.set_page_config
def show_login_register_page():
    st.title("🔐 User Authentication")





# -------------------------------------------------------------------------------------------
DB_PATH = "app.db"

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)
def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            password TEXT,
            role TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()
def add_user(username, name, email, phone, password, role="user"):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
    if cur.fetchone():
        conn.close()
        return False, "Username already exists"

    cur.execute(
        "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
        (username, name, email, phone, hash_password(password), role, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return True, "User registered"

def authenticate_user(username, password):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return False, "User not found"

    if row[4] != hash_password(password):
        return False, "Wrong password"

    # ✅ RETURN USER DICT
    return True, {
        "username": row[0],
        "name": row[1],
        "email": row[2],
        "phone": row[3],
        "role": row[5]
    }

def get_user(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "username": row[0],
        "name": row[1],
        "email": row[2],
        "phone": row[3],
        "role": row[5]
    }
def update_user_profile(username, name, email, phone):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE users SET name=?, email=?, phone=?
        WHERE username=?
    """, (name, email, phone, username))
    conn.commit()
    conn.close()
    return True, "Profile updated"




# Create CSVs if missing with headers


# ----------------- UTILITIES -----------------


def save_feedback(username, rating, comment):
    df = pd.read_csv(FEEDBACK_CSV)
    new = {"username": username, "rating": rating, "comment": comment, "timestamp": datetime.now().isoformat()}
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(FEEDBACK_CSV, index=False)

def log_prediction(username, prediction, probability, inputs):
    df = pd.read_csv(PREDICTION_LOG)
    new = {"username": username, "timestamp": datetime.now().isoformat(),
           "prediction": int(prediction), "probability": float(probability),
           "inputs_json": json.dumps(inputs)}
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(PREDICTION_LOG, index=False)

# Load model safely
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

model = load_model(MODEL_PATH)

# Load dataset (if present) for smart inputs



def load_dataset(path):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        else:
            return None
    except Exception as e:
        st.warning(f"Dataset load failed: {e}")
        return None


df_data = load_dataset(DATA_PATH)
if df_data is not None:
    # drop obvious target/metadata if present
    input_cols = df_data.drop(columns=["Breakdown","Vehicle_ID","Vehicle_Name"], errors='ignore').columns.tolist()
    cat_cols = df_data[input_cols].select_dtypes(include=['object','category']).columns.tolist()
    num_cols = [c for c in input_cols if c not in cat_cols]
else:
    input_cols = ["Vehicle_Type","Manufacturer","Model_Year","Mileage_km","Engine_Hours",
                  "Fuel_Type","Average_Load_%","Driving_Conditions","Previous_Breakdowns",
                  "Brake_Pad_Wear_%","Fuel_Efficiency_kmpl","Average_Speed_kmph","Idle_Time_%"]
    cat_cols = ["Vehicle_Type","Manufacturer","Fuel_Type","Driving_Conditions"]
    num_cols = [c for c in input_cols if c not in cat_cols]

# ----------------- AUTH (session state) -----------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None



def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None

# ----------------- Twilio SMS (secure, user-aware) -----------------
def send_whatsapp_alert_twilio(message, to_phone):
    try:
        ACCOUNT_SID = st.secrets.get("TWILIO_SID")
        AUTH_TOKEN  = st.secrets.get("TWILIO_TOKEN")
        FROM_PHONE = st.secrets.get("TWILIO_FROM")

        if not ACCOUNT_SID or not AUTH_TOKEN or not FROM_PHONE:
            return False, "Twilio WhatsApp credentials missing"

        from twilio.rest import Client
        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        # WhatsApp requires prefix
        if not to_phone.startswith("whatsapp:"):
            to_phone = "whatsapp:" + to_phone

        msg = client.messages.create(
            body=message,
            from_=FROM_PHONE,
            to=to_phone
        )

        return True, msg.sid

    except Exception as e:
        return False, str(e)





# ----------------- PDF helper -----------------
def generate_pdf(vehicle_inputs: dict, prediction: int, proba: float, top_features: dict) -> str:
    """
    Generate a simple PDF report and return file path.
    Requires fpdf installed; if not available, this will raise.
    """
    if not FPDF_AVAILABLE:
        raise RuntimeError("FPDF not installed")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Vehicle Breakdown Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Date: {datetime.now().isoformat()}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, f"Prediction: {'Breakdown' if prediction==1 else 'No Breakdown'}", ln=True)
    pdf.cell(0, 8, f"Probability: {proba:.2f}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, "Inputs:", ln=True)
    for k,v in vehicle_inputs.items():
        pdf.multi_cell(0, 7, f" - {k}: {v}")
    pdf.ln(3)
    pdf.cell(0, 8, "Top feature insights:", ln=True)
    for k,v in top_features.items():
        pdf.multi_cell(0, 7, f" - {k}: {v}")
    out = "prediction_report.pdf"
    pdf.output(out)
    return out

# ----------------- MODEL INSIGHT HELPERS -----------------
def get_feature_importances():
    try:
        rf = model.named_steps.get("classifier", None) if hasattr(model, 'named_steps') else model
        importances = rf.feature_importances_
    except Exception:
        return None
    # try to get feature names from preprocessor
    try:
        pre = model.named_steps.get("preprocessor", None)
        feature_names = pre.get_feature_names_out()
    except Exception:
        # fallback to numeric features with indices
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

# ----------------- UI LAYOUT -----------------
st.title("🚗 Vehicle Breakdown Predictor")

# Sidebar: account actions
with st.sidebar:
    st.header("Account")
    if not st.session_state["logged_in"]:
        choice = st.selectbox("Action", ["Login","Register"])
        if choice == "Login":
            st.subheader("Login")
            lu = st.text_input("Username", key="login_user")
            lp = st.text_input("Password", type="password", key="login_pass")
            if st.button("Log in"):
                ok, result = authenticate_user(lu.strip(), lp.strip())

                if ok:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = result["username"]
                    st.session_state["role"] = result["role"]
                    st.success("Logged in ✅")
                    st.rerun()
                else:
                    st.error(result)
            st.markdown("---")
            st.write("New user? Choose Register from this sidebar.")
        else:
            st.subheader("Register")
            reg_user = st.text_input("Choose username", key="reg_user")
            reg_name = st.text_input("Full name", key="reg_name")
            reg_email = st.text_input("Email", key="reg_email")
            reg_phone = st.text_input("Phone (E.164 e.g. +91...)", key="reg_phone")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            reg_role = st.selectbox("Role", ["user","manager"], index=0)
            if st.button("Register"):
                if not reg_user or not reg_pass or not reg_name:
                    st.error("Username, name and password are required")
                else:
                    ok, msg = add_user(reg_user.strip(), reg_name.strip(), reg_email.strip(), reg_phone.strip(), reg_pass.strip(), reg_role)
                    if ok:
                        st.success("Registered successfully. Please login.")
                    else:
                        st.error(msg)
    else:
        u = get_user(st.session_state["username"])
        st.subheader(u.get("name",""))
        st.write(u.get("email",""))
        st.write(f"Role: {st.session_state.get('role')}")
        if st.button("Logout"):
            logout_user()
            st.rerun()
        st.markdown("---")
        st.subheader("Quick Actions")
        # Test SMS button
        if st.button("Send Test SMS"):
            if st.session_state["logged_in"]:
                user = get_user(st.session_state["username"])
                phone = user.get("phone","")
                if not phone:
                    st.warning("Set phone in profile first.")
                else:
                    ok,info = send_whatsapp_alert_twilio("Test alert from Vehicle App", phone)
                    if ok:
                        st.success(f"Test SMS sent (SID: {info})")
                    else:
                        st.error(f"Test failed: {info}")
        st.markdown("---")
        st.write("App notes: keep Twilio creds in environment variables and phone numbers in E.164 format.")

# ----------------- Tabs: dynamic based on login state -----------------
# Replace the old static tabs_list / tabs creation with this block

# ensure session keys exist
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None

# base tabs visible to everyone
tabs_list = ["Home", "Predict", "Batch Upload", "Analytics & Explainability", "My Profile", "Feedback"]

# unlock additional tabs only after login
if st.session_state.get("logged_in", False):
    # Live Monitor (simulation) visible to any logged-in user
    tabs_list.append("Live Monitor")
    # Admin tab only for admin role
    if st.session_state.get("role") == "admin":
        tabs_list.append("Admin")
else:
    st.sidebar.info("🔒 Log in to unlock Live Monitor and Admin features")
if not st.session_state["logged_in"]:
    show_login_register_page()
    st.stop()


# create tabs
tabs = st.tabs(tabs_list)

# helper: get index by name (returns None if not present)
def tab_index(name):
    try:
        return tabs_list.index(name)
    except ValueError:
        return None

# compute indexes used throughout the app (None if tab not present)
home_idx      = tab_index("Home")
predict_idx   = tab_index("Predict")
batch_idx     = tab_index("Batch Upload")
analytics_idx = tab_index("Analytics & Explainability")
profile_idx   = tab_index("My Profile")
feedback_idx  = tab_index("Feedback")
live_idx      = tab_index("Live Monitor")
admin_idx     = tab_index("Admin")

# ----------------- HOME -----------------
# --- Replace the following block (from with tabs[0]: ... to end of retrain) with this code ---

# HOME
if home_idx is not None:
    with tabs[home_idx]:
        st.header("Welcome")
        st.write("This integrated app supports multi-user login, per-user history, feedback, SHAP explainability, PDF reports, SMS alerts, and batch prediction.")
        st.write("Register in the sidebar and then log in to access profile and get SMS alerts to your phone number.")
        st.markdown("---")
        st.write("Model & dataset status:")
        if model is None:
            st.error(f"Model not found at {MODEL_PATH}. Predictions are disabled.")
        else:
            st.success("Model loaded.")
        if df_data is None:
            st.warning("Dataset not found: smart inputs will use defaults.")

# PREDICT
if predict_idx is not None:
    with tabs[predict_idx]:
        st.header("Single Vehicle Prediction")
        if model is None:
            st.error("Prediction disabled - model missing.")
        else:
            # Build dynamic form
            with st.form("predict_form"):
                inputs = {}
                if df_data is not None:
                    for col in input_cols:
                        if col in cat_cols:
                            opts = df_data[col].dropna().unique().tolist()[:100]
                            opts = opts if len(opts)>0 else ["Unknown"]
                            inputs[col] = st.selectbox(col, opts)
                        elif "%" in col or "Percent" in col or col.lower().endswith("_%"):
                            inputs[col] = st.slider(col, 0, 100, 20)
                        elif "year" in col.lower():
                            miny = int(df_data[col].min()) if col in df_data.columns and pd.notna(df_data[col].min()) else 2000
                            maxy = int(df_data[col].max()) if col in df_data.columns and pd.notna(df_data[col].max()) else datetime.now().year
                            inputs[col] = st.number_input(col, min_value=miny, max_value=maxy, value=maxy)
                        else:
                            default_val = float(df_data[col].median()) if col in df_data.columns and pd.notna(df_data[col].median()) else 0.0
                            inputs[col] = st.number_input(col, value=float(default_val))
                else:
                    for col in input_cols:
                        if col in cat_cols:
                            inputs[col] = st.text_input(col, "")
                        else:
                            inputs[col] = st.number_input(col, value=0.0)
                submitted = st.form_submit_button("Predict Breakdown")
            if submitted:
                inp_df = pd.DataFrame([inputs])
                # try numeric coercion
                for c in inp_df.columns:
                    try:
                        inp_df[c] = pd.to_numeric(inp_df[c])
                    except Exception:
                        pass
                try:
                    pred = model.predict(inp_df)[0]
                    proba = float(model.predict_proba(inp_df)[0][1])

                    # Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba*100,
                        title={'text': "Breakdown Probability (%)"},
                        gauge={'axis': {'range': [0,100]},
                               'bar': {'color': "crimson" if proba>0.5 else "green"},
                               'steps': [{'range':[0,50],'color':'lightgreen'},
                                         {'range':[50,80],'color':'yellow'},
                                         {'range':[80,100],'color':'red'}]}
                    ))
                    st.plotly_chart(fig, use_container_width=True, key="plotly_" + str(uuid.uuid4()))
                    st.success(f"{'🚨 Breakdown Likely' if int(pred)==1 else '✅ No Breakdown Expected'}  | Probability: {proba:.2f}")

                    # SHAP explainability robust
                    top_feature_dict = {}
                    if SHAP_AVAILABLE:
                        try:
                            pre = model.named_steps.get("preprocessor", None) if hasattr(model, 'named_steps') else None
                            clf = model.named_steps.get("classifier", None) if hasattr(model, 'named_steps') else model
                            transformed = pre.transform(inp_df) if pre is not None else inp_df.values
                            explainer = shap.TreeExplainer(clf)
                            shap_vals = explainer.shap_values(transformed)
                            # handle shapes
                            if isinstance(shap_vals, list):
                                shap_arr = shap_vals[-1]
                            else:
                                shap_arr = shap_vals
                            if getattr(shap_arr, "ndim", None) == 3:
                                shap_arr = shap_arr[:, :, -1]
                            try:
                                feature_names = pre.get_feature_names_out() if pre is not None else [f"f{i}" for i in range(shap_arr.shape[1])]
                            except Exception:
                                feature_names = [f"f{i}" for i in range(shap_arr.shape[1])]
                            shap_df = pd.DataFrame(shap_arr, columns=feature_names)
                            top_feats = shap_df.abs().T.sort_values(by=0, ascending=False).head(3)
                            top_feature_dict = {f: f"SHAP influence {round(top_feats.loc[f,0],4)}" for f in top_feats.index}
                            st.subheader("Top contributing features (SHAP)")
                            for f,v in top_feature_dict.items():
                                st.write(f"• **{f}**: {v}")
                        except Exception as e:
                            st.info("SHAP not available for this model or failed: " + str(e))
                    else:
                        st.info("SHAP not installed. To enable, pip install shap.")

                    # PDF report
                    try:
                        if FPDF_AVAILABLE:
                            pdf_path = generate_pdf(inputs, int(pred), proba, top_feature_dict)
                            with open(pdf_path, "rb") as f:
                                st.download_button("Download PDF Report", f, file_name="prediction_report.pdf")
                        else:
                            st.info("FPDF not installed; PDF export not available.")
                    except Exception as e:
                        st.warning("PDF generation failed: " + str(e))

                    # SMS alert to user's profile phone
                    if proba >= SMS_THRESHOLD:
                        if st.session_state.get("logged_in", False):
                            user = get_user(st.session_state["username"])
                            phone = user.get("phone","")
                            if phone:
                                ok,info = send_whatsapp_alert_twilio(f"ALERT: Breakdown risk {proba:.2f}", phone)
                                if ok:
                                    st.info(f"SMS alert sent (SID: {info})")
                                else:
                                    st.warning(f"SMS not sent: {info}")
                            else:
                                st.info("High risk detected. Add phone to profile to receive SMS.")
                        else:
                            st.info("High risk detected. Log in to receive SMS alerts to your profile phone.")

                    # Log prediction
                    username = st.session_state.get("username") if st.session_state.get("logged_in", False) else "anonymous"
                    try:
                        log_prediction(username, pred, proba, inputs)
                    except Exception as e:
                        st.warning("Prediction logging failed: " + str(e))
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

# BATCH UPLOAD
if batch_idx is not None:
    with tabs[batch_idx]:
        st.header("Batch Predictions (CSV)")
        uploaded = st.file_uploader("Upload CSV (columns must match expected inputs)", type=["csv"])
        if uploaded:
            try:
                df_in = pd.read_csv(uploaded, encoding="utf-8", engine="python", sep=None)
                missing = set(input_cols) - set(df_in.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    preds = model.predict(df_in)
                    probas = model.predict_proba(df_in)[:,1]
                    df_in["prediction"] = preds
                    df_in["probability"] = probas
                    st.dataframe(df_in.head(20))
                    csv = df_in.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions CSV", csv, "predictions.csv")
            except Exception as e:
                st.error("Batch prediction failed: " + str(e))

# ANALYTICS & EXPLAINABILITY
if analytics_idx is not None:
    with tabs[analytics_idx]:
        st.header("Model Insights & Feature Importance")
        fi_series = get_feature_importances()
        if fi_series is None:
            st.info("Feature importance unavailable for this model.")
        else:
            top10 = fi_series.sort_values(ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(6,4))
            top10.plot(kind='barh', ax=ax, color='teal')
            ax.set_title("Top 10 Influential Features")
            st.pyplot(fig)
            st.markdown("### Technical Interpretation")
            for feat in top10.index[::-1]:
                if "Previous_Breakdowns" in feat or "Previous" in feat:
                    st.write("• Previous Breakdowns: indicates recurring unresolved issues.")
                elif "Brake" in feat or "Brake_Pad" in feat:
                    st.write("• Brake Wear: high wear correlates with maintenance needs and failures.")
                elif "Engine_Hours" in feat:
                    st.write("• Engine Hours: more hours → thermal and mechanical wear.")
                elif "Load" in feat or "Average_Load" in feat:
                    st.write("• Average Load: heavy loads accelerate drivetrain stress.")
                else:
                    st.write(f"• {feat}: contributes to prediction in non-linear ways captured by Random Forest.")
        st.markdown("---")
        st.header("Prediction History (All)")
        if Path(PREDICTION_LOG).exists():
            hist = pd.read_csv(PREDICTION_LOG)
            st.dataframe(hist.sort_values("timestamp", ascending=False).head(200))
            st.download_button("Download full history", hist.to_csv(index=False).encode('utf-8'), "prediction_history.csv")
        else:
            st.info("No prediction history yet.")

# MY PROFILE
if feedback_idx is not None and predict_idx is not None:
    # Note: keep profile tab index consistent with earlier mapping; profile was tab index 4 in your old layout.
    # We used `feedback_idx` check above; ensure profile index variable exists if you renamed it.
    profile_idx = tab_index("My Profile")
else:
    profile_idx = tab_index("My Profile")

if profile_idx is not None:
    with tabs[profile_idx]:
        st.header("My Profile")
        if not st.session_state.get("logged_in", False):
            st.info("Login to view and edit profile.")
        else:
            user = get_user(st.session_state["username"])
            st.subheader(user.get("name",""))
            st.write(f"Username: {user.get('username')}")
            st.write(f"Email: {user.get('email')}")
            st.write(f"Phone: {user.get('phone')}")
            st.write(f"Role: {user.get('role')}")
            st.markdown("---")
            st.subheader("Edit profile")
            new_name = st.text_input("Full name", value=user.get("name",""))
            new_email = st.text_input("Email", value=user.get("email",""))
            new_phone = st.text_input("Phone (E.164)", value=user.get("phone",""))
            if st.button("Save profile"):
                ok,msg = update_user_profile(user["username"], name=new_name, email=new_email, phone=new_phone)
                if ok:
                    st.success("Profile updated.")
                else:
                    st.error("Invalid username or password")
            st.markdown("---")
            st.subheader("My prediction history")
            hist = pd.read_csv(PREDICTION_LOG)
            myhist = hist[hist["username"] == user["username"]]
            if not myhist.empty:
                st.dataframe(myhist.sort_values("timestamp", ascending=False).head(100))
            else:
                st.info("No personal prediction history yet.")

# FEEDBACK
if feedback_idx is not None:
    with tabs[feedback_idx]:
        st.header("Feedback")
        if not st.session_state.get("logged_in", False):
            st.info("Login to submit feedback.")
        else:
            st.subheader("Leave feedback")
            r = st.slider("Rating (1-5)", 1, 5, 5)
            c = st.text_area("Comments")
            if st.button("Submit feedback"):
                try:
                    save_feedback(st.session_state["username"], r, c)
                    st.success("Thanks — your feedback has been saved.")
                except Exception as e:
                    st.error("Failed to save feedback: " + str(e))
            st.markdown("---")
            st.subheader("My feedback history")
            fdf = pd.read_csv(FEEDBACK_CSV)
            myf = fdf[fdf["username"] == st.session_state.get("username")]
            if not myf.empty:
                st.dataframe(myf.sort_values("timestamp", ascending=False).head(50))
            else:
                st.info("You haven't submitted feedback yet.")

# LIVE SENSOR SIMULATION (only if Live Monitor tab exists)
if live_idx is not None:
    with tabs[live_idx]:
        st.header("Live Sensor Simulation (Demo only)")
        st.write("Simulate a live stream of sensor values and see predictions update.")
        sim_count = st.number_input("Number of simulated updates", min_value=5, max_value=200, value=20)
        sim_interval = st.number_input("Interval between updates (seconds)", min_value=0.1, max_value=10.0, value=0.8, step=0.1)
        start = st.button("Start Simulation")
        if start:
            placeholder = st.empty()
            import time, random
            for i in range(sim_count):
                # create random inputs - adjust ranges to your data
                sim_inputs = {}
                for col in input_cols:
                    if col in cat_cols:
                        # random pick from dataset or fallback value
                        if df_data is not None and col in df_data.columns:
                            sim_inputs[col] = random.choice(df_data[col].dropna().unique().tolist()[:50])
                        else:
                            sim_inputs[col] = "Unknown"
                    else:
                        # numeric simulation
                        sim_inputs[col] = float(np.random.normal(loc=50, scale=20)) if "%" in col or col.lower().endswith("_%") else float(np.random.uniform(0,100))
                inp = pd.DataFrame([sim_inputs])
                try:
                    pred = model.predict(inp)[0]
                    proba = float(model.predict_proba(inp)[0][1])
                except Exception:
                    pred, proba = 0, 0.0
                with placeholder.container():
                    st.subheader(f"Update {i+1}/{sim_count}")
                    st.json(sim_inputs)
                    st.write(f"Predicted probability: {proba:.2f}")
                    # small gauge
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=proba*100, gauge={'axis': {'range': [0,100]}}))
                    st.plotly_chart(fig, use_container_width=True, key="plotly_" + str(uuid.uuid4()))
                time.sleep(sim_interval)
            st.success("Simulation completed.")

# ADMIN (only if Admin tab exists)
if admin_idx is not None:
    with tabs[admin_idx]:
        st.header("Admin Dashboard")
        st.subheader("Users")
        conn = get_db()
        udf = pd.read_sql("SELECT username, name, email, phone, role, created_at FROM users", conn)
        conn.close()
        st.dataframe(udf)

        st.markdown("Create user (admin)")
        a_user = st.text_input("Username (admin create)", key="a_user")
        a_name = st.text_input("Name (admin create)", key="a_name")
        a_email = st.text_input("Email (admin create)", key="a_email")
        a_phone = st.text_input("Phone (admin create)", key="a_phone")
        a_pass = st.text_input("Password (admin create)", key="a_pass")
        a_role = st.selectbox("Role (admin create)", ["user","manager","admin"], index=0, key="a_role")
        if st.button("Create user (admin)"):
            if not a_user or not a_pass:
                st.error("Provide username and password")
            else:
                ok,msg = add_user(a_user, a_name, a_email, a_phone, a_pass, a_role)
                if ok:
                    st.success("User created")
                else:
                    st.error("Invalid username or password")
        st.markdown("---")
        st.subheader("Feedback (all users)")
        fdf = pd.read_csv(FEEDBACK_CSV)
        st.dataframe(fdf.sort_values("timestamp", ascending=False).head(500))
        if st.button("Export feedback CSV"):
            st.download_button("Download feedback", fdf.to_csv(index=False).encode('utf-8'), "feedback_export.csv")
        if st.button("Clear all feedback"):
            pd.DataFrame(columns=["username","rating","comment","timestamp"]).to_csv(FEEDBACK_CSV, index=False)
            st.warning("Feedback cleared.")
        st.markdown("---")
        st.subheader("Prediction logs")
        pdf = pd.read_csv(PREDICTION_LOG)
        st.dataframe(pdf.sort_values("timestamp", ascending=False).head(500))
        if st.button("Export predictions CSV"):
            st.download_button("Download predictions", pdf.to_csv(index=False).encode('utf-8'), "predictions_export.csv")

        # ----------------- ADMIN: Quick Analytics (only inside Admin tab) -----------------
        st.markdown("---")
        st.subheader("Admin: Quick Analytics")

        # Load prediction log
        try:
            pred_df = pd.read_csv(PREDICTION_LOG)
        except Exception:
            pred_df = pd.DataFrame()

        if not pred_df.empty:
            # 1) Breakdown % by Manufacturer (if manufacturer exists in inputs_json)
            def extract_field_from_json(col_json, field):
                try:
                    j = json.loads(col_json)
                    return j.get(field, None)
                except Exception:
                    return None

            # try extract manufacturer column from inputs_json
            pred_df["manufacturer"] = pred_df["inputs_json"].apply(lambda x: extract_field_from_json(x, "Manufacturer"))
            manu_counts = pred_df.groupby("manufacturer")["prediction"].apply(lambda s: (s==1).sum()).rename("breakdowns")
            manu_total = pred_df.groupby("manufacturer").size().rename("total")
            manu = pd.concat([manu_counts, manu_total], axis=1).fillna(0)
            manu["breakdown_rate"] = manu["breakdowns"] / manu["total"]

            manu_plot = manu.sort_values("breakdown_rate", ascending=False).reset_index().head(10)
            if not manu_plot.empty:
                fig = go.Figure([go.Bar(x=manu_plot["manufacturer"], y=manu_plot["breakdown_rate"]*100)])
                fig.update_layout(title="Top Manufacturers by Breakdown Rate (%)", xaxis_title="Manufacturer", yaxis_title="Breakdown %")
                st.plotly_chart(fig, use_container_width=True, key="plotly_" + str(uuid.uuid4()))
            else:
                st.info("Manufacturer info not available in prediction inputs.")

            # 2) Average probability over time (daily)
            try:
                pred_df["timestamp_dt"] = pd.to_datetime(pred_df["timestamp"])
                daily = pred_df.set_index("timestamp_dt").resample("D")["probability"].mean().dropna()
                if not daily.empty:
                    fig2 = go.Figure([go.Scatter(x=daily.index, y=daily.values, mode="lines+markers")])
                    fig2.update_layout(title="Average Predicted Probability (daily)", xaxis_title="Date", yaxis_title="Avg probability")
                    st.plotly_chart(fig, use_container_width=True, key="plotly_" + str(uuid.uuid4()))
            except Exception:
                st.info("Not enough timestamped data to show trend.")
        else:
            st.info("No prediction logs yet — run some predictions to fill analytics.")

        st.markdown("---")
        st.subheader("Model Retraining (Admin)")
        st.write("Upload a clean CSV with same schema as training data. The app will retrain a RandomForest and overwrite saved model.")

        retrain_file = st.file_uploader("Upload CSV to retrain model", type=["csv"])
        if retrain_file is not None:
            if st.button("Retrain model now"):
                try:
                    train_df = pd.read_csv(retrain_file)
                    # Quick preprocess: drop NA rows and separate target if present
                    if "Breakdown" not in train_df.columns:
                        st.error("CSV must contain 'Breakdown' target column.")
                    else:
                        from sklearn.model_selection import train_test_split
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.pipeline import Pipeline
                        from sklearn.preprocessing import OneHotEncoder, StandardScaler
                        from sklearn.compose import ColumnTransformer

                        X = train_df.drop(columns=["Breakdown"])
                        y = train_df["Breakdown"]
                        # simple column splits
                        cat_cols_rt = X.select_dtypes(include=['object','category']).columns.tolist()
                        num_cols_rt = [c for c in X.columns if c not in cat_cols_rt]

                        preproc = ColumnTransformer([
                            ("num", StandardScaler(), num_cols_rt),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_rt)
                        ], remainder="drop")

                        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        pipeline = Pipeline([("preprocessor", preproc), ("classifier", clf)])
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                        pipeline.fit(X_train, y_train)
                        train_acc = pipeline.score(X_train, y_train)
                        val_acc = pipeline.score(X_val, y_val)
                        # save model
                        save_dir = Path("saved_models")
                        save_dir.mkdir(exist_ok=True)
                        joblib.dump(pipeline, save_dir / "rf_pipeline.pkl")
                        st.success(f"Retrain complete. Train acc={train_acc:.3f}, Val acc={val_acc:.3f}. Model saved.")
                except Exception as e:
                    st.error("Retrain failed: " + str(e))
        

# ----------------- Footer -----------------
st.markdown("---")
st.caption("Notes: Passwords are hashed before storage. For production, use a proper DB and hosted auth (Firebase/Auth0). Keep Twilio and other secrets in environment variables.")























