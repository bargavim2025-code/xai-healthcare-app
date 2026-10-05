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
    exp_col = find_column(df, ["experience", "years_experience", "experience_years", "years"])
    location_col = find_column(df, ["location", "city"])
    consultation_col = find_column(df, ["consultation_type", "consultation", "mode"])
    availability_col = find_column(df, ["availability", "available", "timing", "time", "schedule"])
    if not name_col:
        return []
    records = []
    for _, row in df.iterrows():
        records.append({
            "name": str(row[name_col]),
            "speciality": str(row[spec_col]) if spec_col else "General",
            "phone": str(row[phone_col]) if phone_col else "Not available",
            "experience": str(row[exp_col]) if exp_col else "Not specified",
            "location": str(row[location_col]) if location_col else "Chennai",
            "consultation": str(row[consultation_col]) if consultation_col else "In-person",
            "availability": str(row[availability_col]) if availability_col else "Check hospital schedule"
        })
    return records

def recommend_doctors(doctors, speciality):
    if not doctors:
        return []
    s = speciality.lower()
    aliases = {
        "ent": ["ent", "ear nose throat"],
        "cardiology": ["cardiology", "cardiologist", "heart"],
        "dermatology": ["dermatology", "dermatologist", "skin"],
        "ophthalmology": ["ophthalmology", "ophthalmologist", "eye"],
        "dental": ["dental", "dentistry", "dentist"],
        "gynecology": ["gynecology", "gynaecology", "gynecologist"],
        "diabetology": ["diabetology", "diabetes", "endocrinology"],
        "internal medicine": ["internal medicine", "general medicine"],
        "critical care": ["critical care"],
        "general surgery": ["general surgery", "surgery"],
    }
    wanted = aliases.get(s, [s])
    exact = [
        d for d in doctors
        if any(a in d["speciality"].lower() or d["speciality"].lower() in a for a in wanted)
    ]
    return exact[:4] if exact else []


def normalize_symptom(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def symptom_records_from_dataset(df):
    if df.empty:
        return []
    disease_col = find_column(df, ["disease", "condition", "diagnosis", "illness"])
    if not disease_col:
        return []
    symptom_cols = [c for c in df.columns if normalize_symptom(c).startswith("symptom")]
    if not symptom_cols:
        symptom_cols = [c for c in df.columns if c not in [disease_col, "index"]]
    records = []
    for _, row in df.iterrows():
        symptoms = []
        for c in symptom_cols:
            if pd.notna(row[c]):
                s = normalize_symptom(row[c])
                if s and s != "nan":
                    symptoms.append(s)
        records.append({"disease": str(row[disease_col]), "symptoms": symptoms})
    return records


def match_symptoms(selected, records):
    selected = {normalize_symptom(s) for s in selected if s}
    if not selected:
        return []
    results = []
    for rec in records:
        known = set(rec["symptoms"])
        matched = selected.intersection(known)
        if matched:
            score = len(matched) / len(known) * 100
            results.append({
                "disease": rec["disease"],
                "score": round(score, 1),
                "matched": sorted(matched)
            })
    return sorted(results, key=lambda x: x["score"], reverse=True)[:5]


def humanize_symptom(s):
    return str(s).replace("_", " ").title()


def risk_level(value):
    if value >= 70:
        return "HIGH"
    if value >= 40:
        return "MODERATE"
    return "LOW"

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
symptom_records = symptom_records_from_dataset(symptoms_df)

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
.speciality-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e1e7ed;
    margin-bottom: 10px;
}
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
        <p>Early risk screening • Symptom analysis • Doctor discovery • Reports</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🩺 Healthcare Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h3>🩸 Diabetes</h3><p>Machine-learning based diabetes risk screening.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>❤️ Heart Disease</h3><p>Machine-learning based heart-risk screening.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>🩺 AI Symptoms</h3><p>Match symptoms with possible conditions.</p></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("<div class='card'><h3>👨‍⚕️ Doctors</h3><p>Find doctors by speciality and availability.</p></div>", unsafe_allow_html=True)
    with c5:
        st.markdown("<div class='card'><h3>📊 Risk Dashboard</h3><p>View screening risk and contributing factors.</p></div>", unsafe_allow_html=True)
    with c6:
        st.markdown("<div class='card'><h3>📄 PDF Report</h3><p>Download the AI screening report.</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🏥 Hospital Specialities")
    st.write("Click a speciality to view doctor details and availability.")

    specialities = [
        ("👂", "ENT"),
        ("❤️", "Cardiology"),
        ("🧴", "Dermatology"),
        ("👁️", "Ophthalmology"),
        ("🦷", "Dental"),
        ("👩‍⚕️", "Gynecology"),
    ]

    if "selected_speciality" not in st.session_state:
        st.session_state.selected_speciality = None

    for row in [specialities[:3], specialities[3:]]:
        cols = st.columns(3)
        for col, (icon, speciality) in zip(cols, row):
            with col:
                st.markdown(
                    f"<div class='speciality-card'><h2>{icon}</h2><h3>{speciality}</h3></div>",
                    unsafe_allow_html=True
                )
                if st.button(
                    f"View {speciality} Doctors",
                    key=f"home_spec_{speciality}",
                    use_container_width=True
                ):
                    st.session_state.selected_speciality = speciality

    selected_speciality = st.session_state.selected_speciality

    if selected_speciality:
        st.markdown("---")
        st.markdown(f"## 👨‍⚕️ {selected_speciality} Department")

        matches = recommend_doctors(doctor_records, selected_speciality)

        if matches:
            for d in matches:
                with st.container(border=True):
                    left, right = st.columns([3, 1])
                    with left:
                        st.markdown(f"### 👨‍⚕️ {d['name']}")
                        st.write(f"**Speciality:** {d['speciality']}")
                        st.write(f"**Experience:** {d['experience']} years")
                        st.write(f"**Location:** {d['location']}")
                        st.write(f"**Consultation:** {d['consultation']}")
                        st.write(f"📞 **Contact:** {d['phone']}")
                    with right:
                        st.success("🟢 Available")
                        st.write("**Availability**")
                        st.write(d["availability"])
                        if st.button(
                            "📅 Book Appointment",
                            key=f"book_{selected_speciality}_{d['name']}",
                            use_container_width=True
                        ):
                            st.success(f"Appointment selection started for {d['name']}.")
        else:
            st.info(
                f"No {selected_speciality} doctor is present in the current doctors.csv. "
                "Add verified records for this speciality to display them."
            )

    st.caption(
        "Doctor records and availability are demonstration data unless connected to a verified hospital scheduling system."
    )

# ============================================================
# PREDICTION
# ============================================================
elif menu == "Prediction":
    st.markdown("""
    <div class="hero">
        <h2>🧠 AI Health Assessment</h2>
        <p>Symptoms + health parameters → possible conditions + screening risk</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "Educational screening only. The displayed risk is an AI/model estimate, "
        "not a clinical diagnosis or a guaranteed medical probability."
    )

    st.subheader("👤 Patient Information")
    p1, p2, p3 = st.columns(3)
    with p1:
        name = st.text_input("Patient Name", key="ai_name")
    with p2:
        age = st.number_input("Age", 1, 120, 30, key="ai_age")
    with p3:
        gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Other"], key="ai_gender")

    st.markdown("---")
    st.subheader("🩺 Symptoms")

    all_symptoms = sorted({s for r in symptom_records for s in r["symptoms"]})

    if all_symptoms:
        selected_symptoms = st.multiselect(
            "Select symptoms",
            all_symptoms,
            format_func=humanize_symptom,
            key="ai_symptoms"
        )
        symptom_text = st.text_area(
            "Or describe your symptoms",
            placeholder="Example: headache, dizziness and fatigue",
            key="ai_symptom_text"
        )
        text_norm = normalize_symptom(symptom_text)
        free_matches = [
            s for s in all_symptoms
            if s in text_norm or s.replace("_", " ") in symptom_text.lower()
        ]
        combined_symptoms = list(dict.fromkeys(selected_symptoms + free_matches))
    else:
        selected_symptoms = st.multiselect(
            "Select symptoms",
            ["headache", "fever", "cough", "fatigue", "dizziness", "chest_discomfort", "shortness_of_breath"]
        )
        symptom_text = st.text_area("Describe your symptoms")
        combined_symptoms = selected_symptoms

    if combined_symptoms:
        st.info(
            "Detected: " + ", ".join(humanize_symptom(s) for s in combined_symptoms)
        )

    st.markdown("---")
    st.subheader("📋 Health Parameters")
    a, b, c = st.columns(3)

    with a:
        glucose = st.number_input("Glucose (mg/dL)", 0.0, 400.0, 120.0)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 80.0, 500.0, 200.0)

    with b:
        bp = st.number_input("Blood Pressure (mmHg)", 50.0, 250.0, 120.0)
        heart_rate = st.number_input("Heart Rate (bpm)", 30.0, 220.0, 75.0)
        oxygen = st.number_input("Oxygen Saturation (%)", 50.0, 100.0, 98.0)

    with c:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        family_history = st.selectbox("Family History", ["No", "Yes"])
        temperature = st.number_input("Temperature (°C)", 34.0, 43.0, 37.0)

    with st.expander("🩸 Additional diabetes parameters"):
        d1, d2, d3 = st.columns(3)
        with d1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        with d2:
            insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
        with d3:
            pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    if st.button("🧠 Analyze Health", type="primary", use_container_width=True):

        matches = match_symptoms(combined_symptoms, symptom_records)

        diabetes_ml = None
        heart_ml = None

        # Diabetes ML probability
        if diabetes_model is not None:
            values = {}
            for feature in diabetes_features:
                k = normalize_symptom(feature)
                if k == "pregnancies":
                    values[feature] = pregnancies
                elif k == "glucose":
                    values[feature] = glucose
                elif k == "bloodpressure":
                    values[feature] = bp
                elif k == "skinthickness":
                    values[feature] = float(diabetes_df[feature].median())
                elif k == "insulin":
                    values[feature] = insulin
                elif k == "bmi":
                    values[feature] = bmi
                elif k == "diabetespedigreefunction":
                    values[feature] = pedigree
                elif k == "age":
                    values[feature] = age
                else:
                    values[feature] = float(diabetes_df[feature].median())

            try:
                row = pd.DataFrame([[values[x] for x in diabetes_features]], columns=diabetes_features)
                scaled = diabetes_scaler.transform(row)
                diabetes_ml = float(diabetes_model.predict_proba(scaled)[0][1] * 100)
            except Exception:
                diabetes_ml = None

        # Heart ML probability
        if heart_model is not None:
            values = {}
            for feature in heart_features:
                k = normalize_symptom(feature)
                if k == "age":
                    values[feature] = age
                elif k in ["chol", "cholesterol"]:
                    values[feature] = cholesterol
                elif k in ["trestbps", "restingbp", "restingbloodpressure", "bloodpressure"]:
                    values[feature] = bp
                elif k in ["thalach", "maxheartrate", "maximumheartrate", "heartrate"]:
                    values[feature] = heart_rate
                elif k in ["smoking", "smoke", "smoker"]:
                    values[feature] = int(smoking == "Yes")
                elif k in ["cp", "chestpain", "chestpaintype"]:
                    values[feature] = int(any("chest" in s for s in combined_symptoms))
                else:
                    values[feature] = float(heart_df[feature].median())

            try:
                row = pd.DataFrame([[values[x] for x in heart_features]], columns=heart_features)
                scaled = heart_scaler.transform(row)
                probs = heart_model.predict_proba(scaled)[0]
                classes = list(heart_model.classes_)
                positive = [
                    i for i, cls in enumerate(classes)
                    if str(cls).lower() not in ["0", "false", "no", "negative"]
                ]
                heart_ml = float(probs[positive[-1]] * 100) if positive else float(max(probs) * 100)
            except Exception:
                heart_ml = None

        # Symptom scores
        diabetes_symptoms = {
            "frequent_urination", "excessive_thirst", "fatigue",
            "blurred_vision", "increased_hunger"
        }
        heart_symptoms = {
            "chest_discomfort", "shortness_of_breath",
            "breathing_difficulty", "dizziness"
        }

        diabetes_symptom = len(diabetes_symptoms.intersection(combined_symptoms)) / 5 * 100
        heart_symptom = len(heart_symptoms.intersection(combined_symptoms)) / 4 * 100

        # Simple risk signals
        diabetes_signal = 0
        if glucose >= 126:
            diabetes_signal += 30
        elif glucose >= 100:
            diabetes_signal += 15
        if bmi >= 30:
            diabetes_signal += 15
        if bp >= 140:
            diabetes_signal += 10
        if family_history == "Yes":
            diabetes_signal += 10

        heart_signal = 0
        if cholesterol >= 240:
            heart_signal += 30
        elif cholesterol >= 200:
            heart_signal += 10
        if bp >= 140:
            heart_signal += 20
        if smoking == "Yes":
            heart_signal += 15
        if heart_rate > 100:
            heart_signal += 10

        if diabetes_ml is not None:
            diabetes_risk = diabetes_ml * 0.65 + diabetes_symptom * 0.20 + diabetes_signal * 0.15
        else:
            diabetes_risk = diabetes_symptom * 0.60 + diabetes_signal * 0.40

        if heart_ml is not None:
            heart_risk = heart_ml * 0.65 + heart_symptom * 0.20 + heart_signal * 0.15
        else:
            heart_risk = heart_symptom * 0.60 + heart_signal * 0.40

        condition_risks = {
            "Diabetes": min(max(diabetes_risk, 0), 99),
            "Heart Disease": min(max(heart_risk, 0), 99)
        }

        for match in matches:
            if match["disease"] not in condition_risks:
                condition_risks[match["disease"]] = match["score"]

        ranked = sorted(condition_risks.items(), key=lambda x: x[1], reverse=True)

        if not ranked:
            st.info("Enter at least one symptom or health parameter.")
        else:
            top_condition, top_risk = ranked[0]

            st.markdown("---")
            st.subheader("🧠 AI Screening Result")

            r1, r2 = st.columns([2, 1])
            with r1:
                st.markdown(
                    f"""
                    <div class="card">
                    <h2>Possible condition: {top_condition}</h2>
                    <h3>Estimated screening risk: {top_risk:.1f}%</h3>
                    <p><b>Risk level: {risk_level(top_risk)}</b></p>
                    <p>This is a screening estimate, not a confirmed diagnosis.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with r2:
                st.metric("Risk", f"{top_risk:.1f}%")
                st.progress(int(min(top_risk, 100)))

            st.subheader("📊 Possible Conditions")
            for condition, risk in ranked[:5]:
                x, y = st.columns([4, 1])
                with x:
                    st.write(f"**{condition}**")
                    st.progress(int(min(risk, 100)))
                with y:
                    st.metric("Risk", f"{risk:.1f}%")

            if matches:
                st.subheader("🔎 Symptom Evidence")
                for m in matches[:3]:
                    st.write(
                        f"**{m['disease']}** — {m['score']:.1f}% symptom match"
                    )
                    st.caption(
                        "Matched: " + ", ".join(humanize_symptom(s) for s in m["matched"])
                    )

            st.subheader("📋 Contributing Risk Signals")
            signals = []
            if glucose >= 126:
                signals.append("Glucose is in a high screening range.")
            elif glucose >= 100:
                signals.append("Glucose is above the usual fasting screening range.")
            if bmi >= 30:
                signals.append("BMI is 30 or above.")
            if cholesterol >= 240:
                signals.append("Cholesterol is elevated.")
            if bp >= 140:
                signals.append("Blood pressure is elevated.")
            if smoking == "Yes":
                signals.append("Smoking was reported.")
            if oxygen < 94:
                signals.append("Oxygen saturation is below a typical screening threshold.")
            if temperature >= 38:
                signals.append("Temperature is elevated.")
            if family_history == "Yes":
                signals.append("Family history was reported.")
            if not signals:
                signals.append("No major rule-based risk signal was detected.")

            for s in signals:
                st.write("•", s)

            st.subheader("👨‍⚕️ Suggested Specialist")

            speciality_map = {
                "Diabetes": "Diabetology",
                "Heart Disease": "Cardiology",
                "Hypertension": "Cardiology",
                "Ear Infection": "ENT",
                "Common Cold": "ENT",
                "Sinusitis": "ENT",
                "Allergic Rhinitis": "ENT",
                "Asthma": "Critical Care",
                "GERD": "General Surgery",
            }

            suggested = speciality_map.get(top_condition, "Internal Medicine")
            st.success(f"Recommended department: **{suggested}**")

            suggested_doctors = recommend_doctors(doctor_records, suggested)

            if suggested_doctors:
                for d in suggested_doctors[:3]:
                    st.markdown(
                        f"**{d['name']}** — {d['speciality']} | "
                        f"{d['experience']} years | 📞 {d['phone']} | "
                        f"🟢 {d['availability']}"
                    )
            else:
                st.info("No matching doctor is available in doctors.csv.")

            # PDF
            def create_pdf():
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                content = [
                    Paragraph("Well Diagnosis - AI Health Screening Report", styles["Title"]),
                    Spacer(1, 12),
                    Paragraph(f"Patient: {name or 'Not provided'}", styles["Normal"]),
                    Paragraph(f"Age: {age}", styles["Normal"]),
                    Paragraph(f"Gender: {gender}", styles["Normal"]),
                    Paragraph(f"Possible condition: {top_condition}", styles["Normal"]),
                    Paragraph(f"Estimated screening risk: {top_risk:.1f}%", styles["Normal"]),
                    Paragraph(f"Risk level: {risk_level(top_risk)}", styles["Normal"]),
                    Paragraph(f"Suggested speciality: {suggested}", styles["Normal"]),
                    Paragraph(
                        "Symptoms: " + ", ".join(humanize_symptom(s) for s in combined_symptoms),
                        styles["Normal"]
                    ),
                    Paragraph(f"Glucose: {glucose} mg/dL", styles["Normal"]),
                    Paragraph(f"BMI: {bmi}", styles["Normal"]),
                    Paragraph(f"Blood Pressure: {bp} mmHg", styles["Normal"]),
                    Paragraph(f"Cholesterol: {cholesterol} mg/dL", styles["Normal"]),
                    Paragraph(f"Heart Rate: {heart_rate} bpm", styles["Normal"]),
                    Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
                    Spacer(1, 12),
                    Paragraph(
                        "Educational screening only. This report is not a medical diagnosis "
                        "and does not replace professional medical advice.",
                        styles["Normal"]
                    )
                ]
                doc.build(content)
                buffer.seek(0)
                return buffer

            st.download_button(
                "📄 Download AI Screening Report",
                data=create_pdf(),
                file_name="well_diagnosis_ai_report.pdf",
                mime="application/pdf",
                use_container_width=True
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
