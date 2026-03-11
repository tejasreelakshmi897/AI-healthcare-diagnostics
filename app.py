import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import hashlib
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Healthcare System",
    layout="wide",
    page_icon="🏥"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
}

/* Title */
h1{
text-align:center;
font-size:42px;
font-weight:700;
background: linear-gradient(90deg,#6a5cff,#00c6ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

/* Buttons */
.stButton>button{
background: linear-gradient(90deg,#6a5cff,#00c6ff);
color:white;
border-radius:10px;
height:45px;
font-weight:600;
}

/* Metric cards */
[data-testid="metric-container"]{
background:white;
border-radius:12px;
padding:15px;
box-shadow:0 6px 15px rgba(0,0,0,0.1);
}

/* Sidebar */
section[data-testid="stSidebar"]{
background-color:#0e4c92;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("database.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
username TEXT,
password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS patients (
name TEXT,
age INT,
glucose FLOAT,
bp FLOAT,
bmi FLOAT,
insulin FLOAT,
risk TEXT
)
""")

# ---------------- PASSWORD HASH ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- AUTH FUNCTIONS ----------------
def add_user(username, password):
    c.execute("INSERT INTO users VALUES (?,?)",
              (username, hash_password(password)))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

# ---------------- SIDEBAR LOGIN ----------------
st.sidebar.title("🔐 Authentication")

menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if menu == "Register":

    st.sidebar.subheader("Create Account")

    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Register"):
        add_user(new_user, new_pass)
        st.sidebar.success("Account Created Successfully!")

elif menu == "Login":

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        user = login_user(username, password)

        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.sidebar.error("Invalid Credentials")

# ---------------- MAIN APP ----------------
if st.session_state.logged_in:

    st.sidebar.success(f"Welcome {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    st.title("🏥 AI-Powered Healthcare Diagnostic System")
    st.markdown("### Intelligent Clinical Decision Support")
    st.divider()

    page = st.sidebar.selectbox(
        "Navigate",
        ["Diagnosis", "Dashboard", "Model Accuracy"]
    )

    # ---------------- DIAGNOSIS ----------------
    if page == "Diagnosis":

        st.markdown("### 🧾 Patient Clinical Parameters")

        col1, col2 = st.columns(2)

        with col1:

            name = st.text_input("👤 Patient Name")
            age = st.slider("🎂 Age", 1, 100)
            preg = st.number_input("🤰 Pregnancies", min_value=0)
            glucose = st.number_input("🩸 Glucose Level", min_value=0)

        with col2:

            bp = st.number_input("💓 Blood Pressure", min_value=0)
            insulin = st.number_input("💉 Insulin", min_value=0)
            bmi = st.number_input("⚖ BMI", min_value=0.0)
            dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0)

        st.divider()

        if st.button("🔍 Predict Risk"):

            input_data = np.array([[preg, glucose, bp, insulin, bmi, dpf, age]])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            risk = "High Risk" if prediction == 1 else "Low Risk"

            st.markdown("## 🧠 AI Diagnosis Result")

            col1, col2 = st.columns(2)

            col1.metric("Risk Level", risk)
            col2.metric("Prediction Confidence", f"{round(probability*100,2)}%")

            st.progress(int(probability * 100))

            if risk == "High Risk":
                st.error("⚠ Immediate medical consultation recommended.")
            else:
                st.success("✅ Condition appears stable.")

            c.execute(
                "INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, age, glucose, bp, bmi, insulin, risk)
            )

            conn.commit()

    # ---------------- DASHBOARD ----------------
    elif page == "Dashboard":

        st.subheader("📋 Patient Records")

        df = pd.read_sql_query("SELECT * FROM patients", conn)

        st.dataframe(df, use_container_width=True)

        if not df.empty:

            st.subheader("📊 Risk Distribution")

            st.bar_chart(df["risk"].value_counts())

    # ---------------- MODEL ACCURACY ----------------
    elif page == "Model Accuracy":

        st.subheader("🤖 Model Performance")

        df = pd.read_csv("diabetes.csv")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)

        st.metric("Model Accuracy", f"{round(acc*100,2)}%")

else:
    st.title("🔒 Please Login to Access the System")