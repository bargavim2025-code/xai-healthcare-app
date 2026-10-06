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

# ============================================================
# DIET RECOMMENDATION ENGINE
# ============================================================
def get_diet_plan(condition, risk, bmi=None, glucose=None, cholesterol=None, bp=None):
    c = str(condition).lower()

    plans = {
        "diabetes": {
            "title": "🩸 Diabetes-Friendly Diet",
            "focus": "Control blood-sugar spikes with balanced meals, high-fiber foods and limited added sugar.",
            "eat": [
                "🥗 Fill about half the plate with non-starchy vegetables such as spinach, cucumber, beans, cauliflower and carrots.",
                "🌾 Choose high-fiber carbohydrates in sensible portions: oats, brown rice, whole-wheat roti, millets and whole grains.",
                "🥚 Include protein with meals: dal, beans, eggs, fish, tofu or skinless chicken.",
                "🍎 Prefer whole fruits in moderate portions rather than fruit juice.",
                "🥜 Choose small portions of unsalted nuts and seeds as snacks.",
            ],
            "avoid": [
                "🥤 Sugary drinks, packaged juices and energy drinks.",
                "🍰 Sweets, cakes, biscuits and foods with large amounts of added sugar.",
                "🍚 Very large portions of white rice or refined carbohydrates.",
                "🍟 Frequent fried and highly processed foods.",
            ],
            "sample": "Breakfast: vegetable oats + boiled egg | Lunch: 1–2 chapati/brown rice + dal + vegetables + curd | Snack: one whole fruit + a few nuts | Dinner: vegetable soup + protein + small whole-grain portion.",
        },
        "heart disease": {
            "title": "❤️ Heart-Healthy Diet",
            "focus": "Support cardiovascular health by reducing sodium, saturated fat and highly processed foods while increasing fiber.",
            "eat": [
                "🥗 Plenty of vegetables and whole fruits.",
                "🌾 Oats, whole grains, brown rice and millets in appropriate portions.",
                "🐟 Fish, beans, dal, tofu and other lean protein sources.",
                "🥜 Unsalted nuts and seeds in small portions.",
                "🫒 Prefer unsaturated plant oils in modest amounts.",
            ],
            "avoid": [
                "🧂 Excess salt, pickles, papad and very salty packaged foods.",
                "🍟 Deep-fried foods and foods high in trans or saturated fat.",
                "🥓 Processed meats and frequent fatty meats.",
                "🥤 Sugary drinks and highly processed snacks.",
            ],
            "sample": "Breakfast: oats + fruit | Lunch: brown rice/chapati + dal + vegetables + curd | Snack: unsalted nuts | Dinner: grilled/steamed protein + vegetables + whole grain.",
        },
        "common cold": {
            "title": "🤧 Recovery-Friendly Diet",
            "focus": "Prioritize hydration, nourishing foods and easy-to-tolerate meals while recovering.",
            "eat": [
                "💧 Water and other non-alcoholic fluids regularly.",
                "🍲 Warm soups and light meals if comfortable.",
                "🍊 Whole fruits and vegetables for nutrients.",
                "🥚 Adequate protein from eggs, dal, beans or other tolerated foods.",
            ],
            "avoid": ["Very spicy or irritating foods if they worsen symptoms.", "Excess sugary drinks and highly processed snacks."],
            "sample": "Breakfast: idli + sambar | Lunch: rice/chapati + dal + vegetables | Snack: fruit | Dinner: light khichdi or soup with protein.",
        },
        "ear infection": {
            "title": "👂 Supportive ENT Diet",
            "focus": "Stay well hydrated and choose nutritious foods; diet does not replace evaluation of an ear infection.",
            "eat": ["💧 Water and adequate fluids.", "🥗 Vegetables and whole fruits.", "🥚 Protein-rich foods such as eggs, dal, beans or fish.", "🍲 Soft, balanced meals if chewing is uncomfortable."],
            "avoid": ["Excessively salty packaged foods.", "Foods that personally worsen discomfort or nausea."],
            "sample": "Breakfast: idli/upma + protein | Lunch: rice/chapati + dal + vegetables | Snack: fruit | Dinner: light balanced meal + fluids.",
        },
        "sinusitis": {
            "title": "👃 Sinus-Supportive Diet",
            "focus": "Hydration and balanced nutrition can support recovery, but persistent or severe symptoms need medical evaluation.",
            "eat": ["💧 Drink fluids regularly.", "🍲 Warm soups if soothing.", "🥗 Plenty of vegetables and whole fruits.", "🥚 Adequate protein."],
            "avoid": ["Excess alcohol.", "Foods that you know trigger your symptoms."],
            "sample": "Breakfast: oats/idli + fruit | Lunch: rice/chapati + dal + vegetables | Snack: fruit | Dinner: soup + protein + whole grain.",
        },
        "allergic rhinitis": {
            "title": "🌿 Balanced Allergy-Friendly Diet",
            "focus": "There is no universal allergy diet; avoid only foods that you know trigger symptoms and maintain balanced nutrition.",
            "eat": ["🥗 Vegetables and whole fruits.", "🌾 Whole grains.", "🥚 Adequate protein.", "💧 Good hydration."],
            "avoid": ["Known personal food triggers.", "Unnecessary restrictive diets without professional advice."],
            "sample": "Choose a balanced plate of vegetables + whole grain + protein at each main meal.",
        },
    }

    key = next((k for k in plans if k in c), None)
    if key is None:
        plan = {
            "title": "🥗 Balanced Recovery Diet",
            "focus": "Because the screening result is not specific enough for a disease-specific diet, focus on a balanced, minimally processed diet and adequate hydration.",
            "eat": [
                "🥗 Half the plate: vegetables and/or salad.",
                "🌾 A sensible portion of whole grains or other high-fiber carbohydrates.",
                "🥚 A protein source such as dal, beans, eggs, fish, tofu or lean chicken.",
                "🍎 Whole fruits in moderate portions.",
                "💧 Drink enough water according to your needs.",
            ],
            "avoid": ["Excess added sugar and sugary drinks.", "Frequent deep-fried and highly processed foods.", "Excess salt."],
            "sample": "Breakfast: oats/idli + protein | Lunch: whole grain + dal/protein + vegetables + curd | Snack: fruit + nuts | Dinner: vegetables + protein + whole grain.",
        }
    else:
        plan = plans[key]

    return plan

# ---------- Sidebar ----------
st.sidebar.title("🏥 Well Diagnosis")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Fix ur diet", "About"]
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
# =========================
# PREDICTION PAGE
# =========================
elif menu == "🧠 Prediction":

    st.title("🧠 AI Health Assessment")
    st.write(
        "Enter the patient's symptoms. The system will dynamically show "
        "health parameters relevant to the selected symptoms."
    )

    # -------------------------
    # Patient details
    # -------------------------
    st.subheader("👤 Patient Details")

    patient_name = st.text_input("Patient Name")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

    # -------------------------
    # Symptoms
    # -------------------------
    st.subheader("🩺 Select Symptoms")

    symptom_list = [
        "Excessive thirst",
        "Frequent urination",
        "Unexplained weight loss",
        "Fatigue",
        "Blurred vision",

        "Chest pain",
        "Shortness of breath",
        "Palpitations",
        "Dizziness",

        "Fever",
        "Cough",
        "Cold",
        "Sore throat",

        "Skin rash",
        "Skin itching",
        "Redness",

        "Eye pain",
        "Blurred vision",
        "Eye redness",

        "Tooth pain",
        "Gum swelling",

        "Abdominal pain",
        "Irregular periods",
        "Pelvic pain"
    ]

    symptoms = st.multiselect(
        "What symptoms does the patient have?",
        symptom_list,
        placeholder="Select one or more symptoms"
    )

    # -------------------------
    # Dynamic health parameters
    # -------------------------

    health_data = {}

    # Diabetes-related symptoms
    diabetes_symptoms = {
        "Excessive thirst",
        "Frequent urination",
        "Unexplained weight loss",
        "Blurred vision"
    }

    # Heart-related symptoms
    heart_symptoms = {
        "Chest pain",
        "Shortness of breath",
        "Palpitations",
        "Dizziness"
    }

    # Infection / respiratory symptoms
    infection_symptoms = {
        "Fever",
        "Cough",
        "Cold",
        "Sore throat"
    }

    # Skin-related symptoms
    skin_symptoms = {
        "Skin rash",
        "Skin itching",
        "Redness"
    }

    # Eye-related symptoms
    eye_symptoms = {
        "Eye pain",
        "Eye redness",
        "Blurred vision"
    }

    # Dental symptoms
    dental_symptoms = {
        "Tooth pain",
        "Gum swelling"
    }

    # Gynecology symptoms
    gyn_symptoms = {
        "Abdominal pain",
        "Irregular periods",
        "Pelvic pain"
    }

    # ---------------------------------
    # Diabetes parameters
    # ---------------------------------
    if diabetes_symptoms.intersection(symptoms):

        st.subheader("🩸 Diabetes-Related Parameters")

        col1, col2 = st.columns(2)

        with col1:
            glucose = st.number_input(
                "Blood Glucose (mg/dL)",
                min_value=50.0,
                max_value=500.0,
                value=100.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=22.0
            )

        with col2:
            insulin = st.number_input(
                "Insulin Level",
                min_value=0.0,
                max_value=1000.0,
                value=80.0
            )

            family_history_diabetes = st.checkbox(
                "Family history of diabetes?"
            )

        health_data["Glucose"] = glucose
        health_data["BMI"] = bmi
        health_data["Insulin"] = insulin
        health_data["Family_History_Diabetes"] = family_history_diabetes


    # ---------------------------------
    # Heart parameters
    # ---------------------------------
    if heart_symptoms.intersection(symptoms):

        st.subheader("❤️ Heart-Related Parameters")

        col1, col2 = st.columns(2)

        with col1:

            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=70,
                max_value=250,
                value=120
            )

            cholesterol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=80,
                max_value=500,
                value=180
            )

        with col2:

            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=30,
                max_value=220,
                value=75
            )

            smoking = st.checkbox(
                "Does the patient smoke?"
            )

        health_data["Systolic_BP"] = systolic_bp
        health_data["Cholesterol"] = cholesterol
        health_data["Heart_Rate"] = heart_rate
        health_data["Smoking"] = smoking


    # ---------------------------------
    # Infection parameters
    # ---------------------------------
    if infection_symptoms.intersection(symptoms):

        st.subheader("🌡️ Infection / Respiratory Parameters")

        col1, col2 = st.columns(2)

        with col1:

            temperature = st.number_input(
                "Body Temperature (°C)",
                min_value=30.0,
                max_value=45.0,
                value=36.8
            )

        with col2:

            oxygen = st.number_input(
                "Oxygen Saturation (SpO₂ %)",
                min_value=50,
                max_value=100,
                value=98
            )

        health_data["Temperature"] = temperature
        health_data["Oxygen"] = oxygen


    # ---------------------------------
    # Skin parameters
    # ---------------------------------
    if skin_symptoms.intersection(symptoms):

        st.subheader("🧴 Skin-Related Parameters")

        skin_duration = st.number_input(
            "How many days has the skin problem been present?",
            min_value=1,
            max_value=365,
            value=3
        )

        health_data["Skin_Duration"] = skin_duration


    # ---------------------------------
    # Eye parameters
    # ---------------------------------
    if eye_symptoms.intersection(symptoms):

        st.subheader("👁️ Eye-Related Parameters")

        eye_duration = st.number_input(
            "How many days has the eye problem been present?",
            min_value=1,
            max_value=365,
            value=2
        )

        health_data["Eye_Duration"] = eye_duration


    # ---------------------------------
    # Dental parameters
    # ---------------------------------
    if dental_symptoms.intersection(symptoms):

        st.subheader("🦷 Dental Parameters")

        dental_duration = st.number_input(
            "How many days has the dental problem been present?",
            min_value=1,
            max_value=365,
            value=2
        )

        health_data["Dental_Duration"] = dental_duration


    # ---------------------------------
    # Gynecology parameters
    # ---------------------------------
    if gyn_symptoms.intersection(symptoms):

        st.subheader("👩‍⚕️ Related Parameters")

        symptom_duration = st.number_input(
            "Duration of the symptoms (days)",
            min_value=1,
            max_value=365,
            value=3
        )

        health_data["Symptom_Duration"] = symptom_duration


    # ---------------------------------
    # No symptoms
    # ---------------------------------
    if len(symptoms) == 0:

        st.info(
            "Please select at least one symptom to display "
            "the relevant health parameters."
        )


    # ---------------------------------
    # Additional description
    # ---------------------------------
    st.subheader("📝 Additional Information")

    description = st.text_area(
        "Describe the patient's symptoms in your own words"
    )


    # ---------------------------------
    # Prediction button
    # ---------------------------------
    if st.button("🔍 Analyze Health", use_container_width=True):

        if not patient_name:
            st.error("Please enter the patient name.")

        elif not symptoms:
            st.error("Please select at least one symptom.")

        else:

            # =========================
            # Symptom-based scoring
            # =========================

            disease_scores = {
                "Diabetes": 0,
                "Heart Disease": 0,
                "Respiratory / Infection": 0,
                "Skin Disorder": 0,
                "Eye Disorder": 0,
                "Dental Problem": 0,
                "Gynecological Condition": 0
            }


            # Diabetes scoring
            diabetes_matches = len(
                diabetes_symptoms.intersection(symptoms)
            )

            disease_scores["Diabetes"] = diabetes_matches * 20

            if "Glucose" in health_data:

                if health_data["Glucose"] >= 126:
                    disease_scores["Diabetes"] += 30

                elif health_data["Glucose"] >= 100:
                    disease_scores["Diabetes"] += 15

                if health_data["BMI"] >= 30:
                    disease_scores["Diabetes"] += 15


            # Heart scoring
            heart_matches = len(
                heart_symptoms.intersection(symptoms)
            )

            disease_scores["Heart Disease"] = heart_matches * 25

            if "Systolic_BP" in health_data:

                if health_data["Systolic_BP"] >= 140:
                    disease_scores["Heart Disease"] += 20

            if "Cholesterol" in health_data:

                if health_data["Cholesterol"] >= 240:
                    disease_scores["Heart Disease"] += 20

            if "Smoking" in health_data and health_data["Smoking"]:
                disease_scores["Heart Disease"] += 10


            # Infection scoring
            infection_matches = len(
                infection_symptoms.intersection(symptoms)
            )

            disease_scores["Respiratory / Infection"] = (
                infection_matches * 25
            )

            if "Temperature" in health_data:

                if health_data["Temperature"] >= 38:
                    disease_scores["Respiratory / Infection"] += 25


            # Skin
            skin_matches = len(
                skin_symptoms.intersection(symptoms)
            )

            disease_scores["Skin Disorder"] = skin_matches * 35


            # Eye
            eye_matches = len(
                eye_symptoms.intersection(symptoms)
            )

            disease_scores["Eye Disorder"] = eye_matches * 35


            # Dental
            dental_matches = len(
                dental_symptoms.intersection(symptoms)
            )

            disease_scores["Dental Problem"] = dental_matches * 40


            # Gynecology
            gyn_matches = len(
                gyn_symptoms.intersection(symptoms)
            )

            disease_scores["Gynecological Condition"] = gyn_matches * 30


            # Keep values within 0–100
            disease_scores = {
                disease: min(score, 100)
                for disease, score in disease_scores.items()
            }


            # Sort conditions
            sorted_conditions = sorted(
                disease_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            possible_condition = sorted_conditions[0][0]
            risk_probability = sorted_conditions[0][1]


            # Risk level
            if risk_probability >= 70:
                risk_level = "HIGH"

            elif risk_probability >= 40:
                risk_level = "MODERATE"

            else:
                risk_level = "LOW"


            # =========================
            # Display result
            # =========================

            st.divider()

            st.subheader("🧠 AI Screening Result")

            st.success(
                f"Possible Condition: {possible_condition}"
            )

            st.metric(
                "Screening Risk Estimate",
                f"{risk_probability}%"
            )

            if risk_level == "HIGH":

                st.error(
                    f"Risk Level: {risk_level}"
                )

            elif risk_level == "MODERATE":

                st.warning(
                    f"Risk Level: {risk_level}"
                )

            else:

                st.info(
                    f"Risk Level: {risk_level}"
                )


            # =========================
            # Other possible conditions
            # =========================

            st.subheader("🔎 Other Possible Conditions")

            for condition, score in sorted_conditions[1:4]:

                if score > 0:

                    st.write(
                        f"**{condition}** — "
                        f"Screening score: {score}%"
                    )


            # =========================
            # Risk factors
            # =========================

            st.subheader("⚠️ Contributing Factors")

            factors = []

            if "Glucose" in health_data:
                if health_data["Glucose"] >= 126:
                    factors.append(
                        "Elevated blood glucose"
                    )

            if "BMI" in health_data:
                if health_data["BMI"] >= 30:
                    factors.append(
                        "High BMI"
                    )

            if "Systolic_BP" in health_data:
                if health_data["Systolic_BP"] >= 140:
                    factors.append(
                        "High blood pressure"
                    )

            if "Cholesterol" in health_data:
                if health_data["Cholesterol"] >= 240:
                    factors.append(
                        "High cholesterol"
                    )

            if "Smoking" in health_data:
                if health_data["Smoking"]:
                    factors.append(
                        "Smoking history"
                    )

            if "Temperature" in health_data:
                if health_data["Temperature"] >= 38:
                    factors.append(
                        "Elevated body temperature"
                    )

            if factors:

                for factor in factors:
                    st.write("•", factor)

            else:

                st.write(
                    "No major risk factor detected from "
                    "the entered parameters."
                )


            # =========================
            # Specialist recommendation
            # =========================

            specialist_map = {

                "Diabetes":
                    "Diabetologist / Endocrinologist",

                "Heart Disease":
                    "Cardiologist",

                "Respiratory / Infection":
                    "General Physician",

                "Skin Disorder":
                    "Dermatologist",

                "Eye Disorder":
                    "Ophthalmologist",

                "Dental Problem":
                    "Dentist",

                "Gynecological Condition":
                    "Gynecologist"
            }

            specialist = specialist_map.get(
                possible_condition,
                "General Physician"
            )

            st.subheader("👨‍⚕️ Recommended Specialist")

            st.info(specialist)


            # =========================
            # Save result for Fix ur diet
            # =========================

            st.session_state.latest_prediction = {
                "patient_name": patient_name,
                "age": age,
                "gender": gender,
                "symptoms": symptoms,
                "condition": possible_condition,
                "risk": risk_probability,
                "risk_level": risk_level,
                "health_data": health_data
            }

            st.success(
                "Assessment completed. You can now open "
                "🥗 Fix ur diet to view personalized general diet guidance."
            )

            st.caption(
                "⚠️ This is an AI-assisted screening estimate for "
                "educational purposes and is not a medical diagnosis."
            )
# ============================================================
# FIX UR DIET
# ============================================================
elif menu == "Fix ur diet":
    st.markdown("""
    <div class="hero">
        <h2>🥗 Fix ur diet</h2>
        <p>Personalized educational diet guidance based on your latest AI screening result</p>
    </div>
    """, unsafe_allow_html=True)

    latest = st.session_state.get("latest_screening")

    if not latest:
        st.info("🧠 First complete a health screening in the Prediction page. Your latest result will appear here with a matching diet plan.")
        st.markdown("### Why this page matters")
        st.write("The diet plan is selected from the condition identified by the screening model and is intended as general educational guidance.")
    else:
        condition = latest["condition"]
        risk = latest["risk"]
        plan = get_diet_plan(
            condition, risk,
            latest.get("bmi"),
            latest.get("glucose"),
            latest.get("cholesterol"),
            latest.get("bp")
        )

        st.success(f"Latest screening result: **{condition}** • Estimated screening risk: **{risk:.1f}%**")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Possible condition", condition)
        with r2:
            st.metric("Screening risk", f"{risk:.1f}%")
        with r3:
            level = "High" if risk >= 70 else "Moderate" if risk >= 40 else "Low"
            st.metric("Risk level", level)

        st.markdown(f"## {plan['title']}")
        st.info(plan["focus"])

        left, right = st.columns(2)
        with left:
            st.markdown("### ✅ What to eat")
            for item in plan["eat"]:
                st.write(item)
        with right:
            st.markdown("### ⚠️ Limit / avoid")
            for item in plan["avoid"]:
                st.write(item)

        st.markdown("### 🍽️ Example one-day meal pattern")
        st.write(plan["sample"])

        # Extra personalized flags based on the measurements entered in Prediction.
        st.markdown("### 🎯 Your screening-based diet focus")
        focus = []
        if latest.get("glucose", 0) >= 126:
            focus.append("Reduce added sugar and keep carbohydrate portions consistent.")
        if latest.get("bmi", 0) >= 30:
            focus.append("Prioritize vegetables, fiber and sensible portions rather than crash dieting.")
        if latest.get("cholesterol", 0) >= 240:
            focus.append("Choose more fiber-rich foods and reduce saturated-fat-heavy foods.")
        if latest.get("bp", 0) >= 140:
            focus.append("Keep sodium intake modest and limit very salty packaged foods.")
        if not focus:
            focus.append("Maintain a balanced plate, regular meal pattern, hydration and minimally processed foods.")
        for item in focus:
            st.write("•", item)

        st.warning(
            "⚠️ This is general educational nutrition guidance, not a personalized medical diet or treatment. "
            "If you have diabetes, heart disease, kidney disease, pregnancy-related needs, food allergies, "
            "or take medicines affected by food, confirm your diet with a qualified doctor or dietitian."
        )

# ============================================================
# ABOUT
# ============================================================
else:
    st.title("🏥 About Well Diagnosis")
    st.write("""
    ### 🩺 Application Overview
    Well Diagnosis is an educational AI healthcare application for disease-risk
    screening, symptom analysis, doctor discovery, diet guidance and report generation.

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
    st.write("Diet guidance:", "✅ Available")

st.sidebar.divider()
st.sidebar.caption("Educational project • Well Diagnosis")
