import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# WELL DIAGNOSIS - AI HEALTHCARE ASSISTANT
# ============================================================
st.set_page_config(page_title="Well Diagnosis", page_icon="🏥", layout="wide")

# ---------- Helpers ----------
def load_csv(filename):
    try:
        return pd.read_csv(filename)
    except Exception as e:
        st.warning(f"Could not load {filename}: {e}")
        return pd.DataFrame()

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(df, names):
    lower = {str(c).lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for name in names:
        key = name.lower().replace("_", "").replace(" ", "")
        if key in lower:
            return lower[key]
    return None

def get_doctor_records(df):
    if df.empty:
        return []
    name_col = find_column(df, ["name", "doctor", "doctor_name", "doctorname"])
    spec_col = find_column(df, ["speciality", "specialty", "department", "specialization", "specialisation"])
    phone_col = find_column(df, ["phone", "mobile", "contact", "phone_number"])
    exp_col = find_column(df, ["experience", "years_experience", "years"])
    if not name_col:
        return []
    records = []
    for _, row in df.iterrows():
        records.append({
            "name": str(row[name_col]),
            "speciality": str(row[spec_col]) if spec_col else "General",
            "phone": str(row[phone_col]) if phone_col else "Not available",
            "experience": str(row[exp_col]) if exp_col else "Not specified"
        })
    return records

def recommend_doctors(doctors, speciality):
    if not doctors:
        return []
    s = speciality.lower()
    exact = [d for d in doctors if s in d["speciality"].lower() or d["speciality"].lower() in s]
    if exact:
        return exact[:4]
    return doctors[:4]

# ---------- Load datasets ----------
diabetes_df = clean_columns(load_csv("diabetes.csv"))
heart_df = clean_columns(load_csv("heart_disease.csv"))
symptoms_df = clean_columns(load_csv("symptoms.csv"))
doctors_df = clean_columns(load_csv("doctors.csv"))

# ---------- Diabetes model ----------
diabetes_model = None
diabetes_scaler = None
diabetes_accuracy = None

if not diabetes_df.empty and "Outcome" in diabetes_df.columns:
    ddf = diabetes_df.copy()
    for col in ["Glucose", "BloodPressure", "BMI", "Insulin"]:
        if col in ddf.columns:
            ddf[col] = pd.to_numeric(ddf[col], errors="coerce")
            if ddf[col].notna().any():
                ddf[col] = ddf[col].replace(0, np.nan).fillna(ddf[col].median())

    ddf = ddf.dropna()
    if len(ddf) >= 20:
        X = ddf.drop("Outcome", axis=1)
        y = ddf["Outcome"]
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        diabetes_scaler = StandardScaler()
        X_train_s = diabetes_scaler.fit_transform(X_train)
        X_test_s = diabetes_scaler.transform(X_test)
        diabetes_model = RandomForestClassifier(n_estimators=150, random_state=42)
        diabetes_model.fit(X_train_s, y_train)
        diabetes_accuracy = accuracy_score(y_test, diabetes_model.predict(X_test_s))

# ---------- Heart model ----------
heart_model = None
heart_scaler = None
heart_features = []
heart_target = None
heart_accuracy = None

if not heart_df.empty:
    hdf = heart_df.copy()
    # Common heart-disease dataset target names
    target_candidates = [
        "target", "output", "outcome", "condition", "heartdisease",
        "heart_disease", "num", "label", "diagnosis"
    ]
    target_col = find_column(hdf, target_candidates)

    if target_col:
        numeric_cols = hdf.select_dtypes(include=np.number).columns.tolist()
        numeric_features = [c for c in numeric_cols if c != target_col]
        if len(numeric_features) >= 2:
            temp = hdf[numeric_features + [target_col]].copy()
            for c in temp.columns:
                temp[c] = pd.to_numeric(temp[c], errors="coerce")
            temp = temp.dropna()
            if len(temp) >= 20 and temp[target_col].nunique() >= 2:
                heart_features = numeric_features
                heart_target = target_col
                Xh = temp[heart_features]
                yh = temp[heart_target]
                try:
                    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
                        Xh, yh, test_size=0.2, random_state=42, stratify=yh
                    )
                except ValueError:
                    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
                        Xh, yh, test_size=0.2, random_state=42
                    )
                heart_scaler = StandardScaler()
                Xh_train_s = heart_scaler.fit_transform(Xh_train)
                Xh_test_s = heart_scaler.transform(Xh_test)
                heart_model = RandomForestClassifier(n_estimators=150, random_state=42)
                heart_model.fit(Xh_train_s, yh_train)
                heart_accuracy = accuracy_score(yh_test, heart_model.predict(Xh_test_s))

# ---------- Doctors ----------
doctor_records = get_doctor_records(doctors_df)

# ---------- CSS ----------
st.markdown("""
<style>
.main { background-color: #f4f8fb; }
.hero {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    padding: 38px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}
.small-note { color: #555; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("🏥 Well Diagnosis")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Symptoms", "Doctors", "About"]
)

# ============================================================
# HOME
# ============================================================
if menu == "Home":
    st.markdown("""
    <div class="hero">
        <h1>🏥 Well Diagnosis</h1>
        <h3>AI-Powered Healthcare Assistant</h3>
        <p>Early risk screening • Symptom guidance • Doctor discovery • Reports</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🩺 Healthcare Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h3>🩸 Diabetes</h3><p>Machine-learning based diabetes risk screening using the diabetes dataset.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>❤️ Heart Disease</h3><p>Heart-risk screening using the uploaded heart disease dataset when its target and numeric features are compatible.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>🩺 Symptoms</h3><p>Search symptoms from the symptoms dataset and view matching information.</p></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("<div class='card'><h3>👨‍⚕️ Doctors</h3><p>Doctor recommendations are loaded from doctors.csv.</p></div>", unsafe_allow_html=True)
    with c5:
        st.markdown("<div class='card'><h3>📊 Risk Dashboard</h3><p>Visual risk scores and model accuracy are displayed after screening.</p></div>", unsafe_allow_html=True)
    with c6:
        st.markdown("<div class='card'><h3>📄 PDF Report</h3><p>Download a patient screening report after a prediction.</p></div>", unsafe_allow_html=True)

    
# ============================================================
# PREDICTION
# ============================================================
elif menu == "Prediction":
    st.markdown("""
    <div class="hero">
        <h2>🔍 Smart Diagnosis</h2>
        <p>Educational risk screening — not a medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120, 30)

    disease = st.radio(
        "Select screening type",
        ["Diabetes", "Heart Disease", "ENT Disorder", "Critical Condition", "General Surgery"],
        horizontal=True
    )

    result = None
    cause = ""
    treatment = ""
    doctor_speciality = "General"
    risk_value = 0

    # ---------- Diabetes ----------
    if disease == "Diabetes":
        st.subheader("🩸 Diabetes Risk Screening")
        glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
        bp = st.number_input("Blood Pressure", 0.0, 250.0, 120.0)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

        if st.button("Predict Diabetes Risk", type="primary"):
            if diabetes_model is not None:
                feature_order = list(diabetes_df.drop("Outcome", axis=1).columns)
                values = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": bp,
                    "SkinThickness": skin,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age
                }
                try:
                    row = pd.DataFrame([[values.get(c, 0) for c in feature_order]], columns=feature_order)
                    pred = int(diabetes_model.predict(diabetes_scaler.transform(row))[0])
                    if hasattr(diabetes_model, "predict_proba"):
                        probability = float(diabetes_model.predict_proba(diabetes_scaler.transform(row))[0][1])
                    else:
                        probability = 0.75 if pred else 0.25
                    risk_value = int(probability * 100)
                    result = "Higher diabetes risk" if pred else "Lower diabetes risk"
                    cause = "The model identified a higher-risk pattern in the supplied measurements." if pred else "The model identified a lower-risk pattern in the supplied measurements."
                except Exception:
                    score = (glucose > 140)*2 + (bmi > 30)*2 + (age > 45) + (bp > 140)
                    risk_value = min(95, 20 + score*15)
                    result = "Higher diabetes risk" if score >= 4 else "Lower diabetes risk"
                    cause = "Screening score based on glucose, BMI, age and blood pressure."
            else:
                score = (glucose > 140)*2 + (bmi > 30)*2 + (age > 45) + (bp > 140)
                risk_value = min(95, 20 + score*15)
                result = "Higher diabetes risk" if score >= 4 else "Lower diabetes risk"
                cause = "Basic screening score because the diabetes model was unavailable."
            treatment = "Consider professional medical evaluation and maintain healthy lifestyle habits."
            doctor_speciality = "Diabetes"

    # ---------- Heart ----------
    elif disease == "Heart Disease":
        st.subheader("❤️ Heart Disease Risk Screening")
        st.caption("If your heart_disease.csv uses standard numeric clinical columns, the trained model will use them. Otherwise a simple screening fallback is used.")

        age_h = st.number_input("Age", 1, 120, 45, key="heart_age")
        chol = st.number_input("Cholesterol", 80.0, 500.0, 200.0)
        bp = st.number_input("Resting Blood Pressure", 60.0, 250.0, 120.0)
        smoke = st.selectbox("Smoking", ["No", "Yes"])
        chest_pain = st.selectbox("Chest discomfort", ["No", "Yes"])
        heart_rate = st.number_input("Maximum Heart Rate", 50.0, 250.0, 150.0)

        if st.button("Predict Heart Risk", type="primary"):
            if heart_model is not None:
                values = {}
                for col in heart_features:
                    lc = str(col).lower().replace("_", "").replace(" ", "")
                    if lc == "age":
                        values[col] = age_h
                    elif lc in ["chol", "cholesterol"]:
                        values[col] = chol
                    elif lc in ["trestbps", "restingbp", "restingbloodpressure", "bloodpressure"]:
                        values[col] = bp
                    elif lc in ["thalach", "maxheartrate", "maximumheartrate", "heartrate"]:
                        values[col] = heart_rate
                    elif lc in ["smoking", "smoke", "smoker"]:
                        values[col] = int(smoke == "Yes")
                    elif lc in ["cp", "chestpain", "chestpaintype"]:
                        values[col] = int(chest_pain == "Yes")
                    else:
                        values[col] = float(heart_df[col].median()) if pd.api.types.is_numeric_dtype(heart_df[col]) else 0
                try:
                    row = pd.DataFrame([[values[c] for c in heart_features]], columns=heart_features)
                    scaled = heart_scaler.transform(row)
                    pred = heart_model.predict(scaled)[0]
                    if hasattr(heart_model, "predict_proba"):
                        probs = heart_model.predict_proba(scaled)[0]
                        risk_value = int(max(probs) * 100) if len(probs) == 2 and pred != 0 else int((1 - max(probs))*100)
                    else:
                        risk_value = 70 if pred else 25
                    positive = str(pred).lower() not in ["0", "false", "no", "negative"]
                    result = "Higher heart-disease risk" if positive else "Lower heart-disease risk"
                    cause = "The model identified a pattern associated with the target labels in the uploaded dataset."
                except Exception:
                    score = (chol > 240)*2 + (bp > 140)*2 + (age_h > 50) + (smoke == "Yes")*2 + (chest_pain == "Yes")*2
                    risk_value = min(95, 20 + score*10)
                    result = "Higher heart-disease risk" if score >= 4 else "Lower heart-disease risk"
                    cause = "Fallback screening score based on the entered risk factors."
            else:
                score = (chol > 240)*2 + (bp > 140)*2 + (age_h > 50) + (smoke == "Yes")*2 + (chest_pain == "Yes")*2
                risk_value = min(95, 20 + score*10)
                result = "Higher heart-disease risk" if score >= 4 else "Lower heart-disease risk"
                cause = "Basic screening score because the heart-disease dataset could not be trained automatically."
            treatment = "Please consult a qualified clinician for interpretation and appropriate care."
            doctor_speciality = "Cardiology"

    # ---------- ENT ----------
    elif disease == "ENT Disorder":
        st.subheader("👂 ENT Screening")
        temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
        throat = st.selectbox("Throat pain", ["No", "Yes"])
        hearing = st.selectbox("Hearing issue", ["No", "Yes"])
        cold = st.selectbox("Cold", ["No", "Yes"])
        if st.button("Check ENT Risk", type="primary"):
            score = int(temp > 38) + int(throat == "Yes") + int(hearing == "Yes") + int(cold == "Yes")
            risk_value = min(95, 20 + score*18)
            result = "Symptoms need medical evaluation" if score >= 2 else "No major warning pattern detected"
            cause = "Based on the symptoms entered."
            treatment = "Consider medical evaluation if symptoms persist or worsen."
            doctor_speciality = "ENT"

    # ---------- Critical ----------
    elif disease == "Critical Condition":
        st.subheader("🚨 Critical Vital Check")
        oxygen = st.number_input("Oxygen Saturation (%)", 50, 100, 95)
        pulse = st.number_input("Pulse Rate", 40, 180, 80)
        if st.button("Check Critical Status", type="primary"):
            if oxygen < 90 or pulse > 120:
                result = "Potentially critical vital signs"
                risk_value = 95
                cause = "Entered oxygen or pulse value is outside a typical safe range."
            else:
                result = "No immediate critical pattern detected"
                risk_value = 20
                cause = "Entered vital signs did not trigger the screening thresholds."
            treatment = "Seek urgent professional medical care for severe symptoms or worsening condition."
            doctor_speciality = "Critical Care"

    # ---------- Surgery ----------
    else:
        st.subheader("🦴 General Surgery Screening")
        pain = st.slider("Pain Level", 1, 10, 5)
        injury = st.selectbox("Significant injury", ["No", "Yes"])
        if st.button("Check Surgery Risk", type="primary"):
            score = pain + int(injury == "Yes")*2
            risk_value = min(95, 20 + score*7)
            result = "Surgical evaluation may be needed" if score > 8 else "No strong surgical warning from this screen"
            cause = "Based on pain level and injury information."
            treatment = "Consult a qualified clinician for examination and diagnosis."
            doctor_speciality = "General Surgery"

    # ---------- Output ----------
    if result:
        st.divider()
        st.subheader("🩺 Screening Report")
        a, b = st.columns(2)
        with a:
            st.write("**Patient:**", name or "Not provided")
            st.write("**Age:**", age)
            st.write("**Screening:**", disease)
            st.write("**Result:**", result)
            st.write("**Reason:**", cause)
        with b:
            st.write("**Suggested speciality:**", doctor_speciality)
            st.write("**Next step:**", treatment)
            st.progress(min(risk_value, 100) / 100)
            st.write(f"**Screening score: {risk_value}%**")

        fig, ax = plt.subplots()
        ax.bar(["Risk"], [risk_value])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Screening score (%)")
        st.pyplot(fig)

        matches = recommend_doctors(doctor_records, doctor_speciality)
        if matches:
            st.subheader("👨‍⚕️ Suggested Doctors")
            for d in matches:
                st.markdown(f"**{d['name']}** — {d['speciality']}  \n📞 {d['phone']} | Experience: {d['experience']}")
        else:
            st.info("No matching doctors were found in doctors.csv.")

        def create_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            content = [
                Paragraph("Well Diagnosis - Screening Report", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"Patient: {name or 'Not provided'}", styles["Normal"]),
                Paragraph(f"Age: {age}", styles["Normal"]),
                Paragraph(f"Screening: {disease}", styles["Normal"]),
                Paragraph(f"Result: {result}", styles["Normal"]),
                Paragraph(f"Reason: {cause}", styles["Normal"]),
                Paragraph(f"Suggested speciality: {doctor_speciality}", styles["Normal"]),
                Paragraph(f"Next step: {treatment}", styles["Normal"]),
                Paragraph(f"Screening score: {risk_value}%", styles["Normal"]),
                Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
                Spacer(1, 12),
                Paragraph("Educational screening only. This report is not a medical diagnosis or a substitute for professional medical advice.", styles["Normal"])
            ]
            doc.build(content)
            buffer.seek(0)
            return buffer

        st.download_button(
            "📄 Download PDF Report",
            data=create_pdf(),
            file_name="well_diagnosis_report.pdf",
            mime="application/pdf"
        )

# ============================================================
# SYMPTOMS
# ============================================================
elif menu == "Symptoms":
    st.markdown("""
    <div class="hero">
        <h2>🩺 Symptom Explorer</h2>
        <p>Search the symptoms contained in symptoms.csv.</p>
    </div>
    """, unsafe_allow_html=True)

    if symptoms_df.empty:
        st.error("symptoms.csv could not be loaded.")
    else:
        st.write(f"Dataset contains **{len(symptoms_df)}** records.")
        search = st.text_input("Search for a symptom or condition")

        display_df = symptoms_df.copy()
        if search:
            mask = display_df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]

        st.dataframe(display_df, use_container_width=True)

        if search:
            st.success(f"Found {len(display_df)} matching record(s).")

# ============================================================
# DOCTORS
# ============================================================
elif menu == "Doctors":
    st.markdown("""
    <div class="hero">
        <h2>👨‍⚕️ Doctor Directory</h2>
        <p>Doctor information loaded from doctors.csv.</p>
    </div>
    """, unsafe_allow_html=True)

    if doctors_df.empty:
        st.error("doctors.csv could not be loaded.")
    else:
        specialities = sorted([str(x) for x in doctors_df.iloc[:, 0].dropna().unique()])
        st.dataframe(doctors_df, use_container_width=True)
        st.caption(f"{len(doctors_df)} doctor record(s) loaded.")

# ============================================================
# ABOUT
# ============================================================
else:
    st.title("🏥 About Well Diagnosis")
    st.write("""
    ### 🩺 Application Overview
    Well Diagnosis is an educational AI healthcare application for disease-risk
    screening, symptom exploration, doctor discovery and report generation.

    ### 🎯 Objective
    To demonstrate how machine learning and structured healthcare datasets can
    support early risk screening and improve access to organized health information.

    ### ⚠️ Disclaimer
    This application is for educational/project demonstration purposes only.
    It does not diagnose disease, prescribe medicines, or replace a qualified
    healthcare professional. For urgent or severe symptoms, seek professional
    medical care immediately.
    """)

    st.subheader("📁 Dataset Status")
    st.write("Diabetes:", "✅ Loaded" if not diabetes_df.empty else "❌ Missing")
    st.write("Heart Disease:", "✅ Loaded" if not heart_df.empty else "❌ Missing")
    st.write("Symptoms:", "✅ Loaded" if not symptoms_df.empty else "❌ Missing")
    st.write("Doctors:", "✅ Loaded" if not doctors_df.empty else "❌ Missing")

st.sidebar.divider()
st.sidebar.caption("Educational project • Well Diagnosis")
