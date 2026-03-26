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

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ================================
# Page Config
# ================================
st.set_page_config(page_title="Kavi Diagnosis", layout="wide")

# ================================
# Sidebar
# ================================
st.sidebar.title("🏥 Kavi Diagnosis")
menu = st.sidebar.radio("Navigation", ["Home", "Prediction", "About"])

# ================================
# Load Dataset
# ================================
df = pd.read_csv("diabetes.csv")

cols = ['Glucose', 'BloodPressure', 'BMI', 'Insulin']
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

# ================================
# HOME
# ================================
if menu == "Home":

    # ================================
    # PREMIUM CSS (WEBSITE STYLE)
    # ================================
    st.markdown("""
    <style>
    .main {
        background-color: #f4f8fb;
    }

    .hero {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        padding: 40px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: 0.3s;
    }

    .card:hover {
        transform: scale(1.05);
    }

    .section {
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================================
    # HERO SECTION
    # ================================
    st.markdown("""
    <div class='hero'>
        <h1>🏥 Well Diagnosis</h1>
        <h3>Your Trusted Healthcare Partner</h3>
        <p>Advanced AI-powered diagnosis and expert medical care</p>
    </div>
    """, unsafe_allow_html=True)

    # ================================
    # SPECIALITIES
    # ================================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("## 🩺 Our Specialities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='card'>
        <img src='https://cdn-icons-png.flaticon.com/512/2966/2966483.png' width='80'>
        <h4>Critical Care</h4>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
        <img src='https://cdn-icons-png.flaticon.com/512/3774/3774299.png' width='80'>
        <h4>ENT</h4>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
        <img src='https://cdn-icons-png.flaticon.com/512/4320/4320371.png' width='80'>
        <h4>Orthopedics</h4>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class='card'>
        <img src='https://cdn-icons-png.flaticon.com/512/2966/2966334.png' width='80'>
        <h4>General Surgery</h4>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class='card'>
        <img src='https://cdn-icons-png.flaticon.com/512/3209/3209265.png' width='80'>
        <h4>Cardiology</h4>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class='card'>
        <img src='https://cdn-icons-png.flaticon.com/512/3870/3870822.png' width='80'>
        <h4>Diabetes Care</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================================
    # SELECT SPECIALITY
    # ================================
    st.markdown("## 🔍 Select Speciality")

    speciality = st.selectbox(
        "",
        ["Critical Care", "ENT", "Orthopedics", "Cardiology", "General Surgery", "Diabetes Care"]
    )

    doctors = {
        "Critical Care": [("Dr. Karthik Raj", "+91 90123 45678"),
                          ("Dr. Meena Das", "+91 91234 56789")],
        "ENT": [("Dr. Suresh Kumar", "+91 92345 67890"),
                ("Dr. Anjali Verma", "+91 93456 78901")],
        "Orthopedics": [("Dr. Arjun Mehta", "+91 94567 89012"),
                        ("Dr. Vikram Singh", "+91 95678 90123")],
        "Cardiology": [("Dr. Ravi Kumar", "+91 96789 01234"),
                       ("Dr. Neha Sharma", "+91 97890 12345")],
        "General Surgery": [("Dr. Rajesh Patel", "+91 98901 23456"),
                            ("Dr. Deepa Nair", "+91 99012 34567")],
        "Diabetes Care": [("Dr. Priya Sharma", "+91 90111 22334"),
                          ("Dr. Mohan Iyer", "+91 91222 33445")]
    }

    st.markdown(f"### 👨‍⚕️ Doctors - {speciality}")

    colA, colB = st.columns(2)
    doc_list = doctors[speciality]

    with colA:
        st.markdown(f"""
        <div class='card'>
        <h4>{doc_list[0][0]}</h4>
        <p>📞 {doc_list[0][1]}</p>
        <p>Experience: 10+ years</p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class='card'>
        <h4>{doc_list[1][0]}</h4>
        <p>📞 {doc_list[1][1]}</p>
        <p>Experience: 8+ years</p>
        </div>
        """, unsafe_allow_html=True)

    # ================================
    # BOOK APPOINTMENT
    # ================================
    st.markdown("## 📅 Book Appointment")

    pname = st.text_input("Patient Name")
    phone = st.text_input("Phone Number")
    selected_doctor = st.selectbox("Select Doctor", [doc[0] for doc in doc_list])
    date = st.date_input("Date")

    if st.button("Book Appointment"):
        if pname and phone:
            st.success(f"✅ Appointment booked with {selected_doctor} on {date}")
        else:
            st.warning("⚠️ Fill all details")

    # ================================
    # FOOTER
    # ================================
    st.markdown("""
    <hr>
    <center>
    📍 Well Diagnosis, Anna Nagar, Chennai <br>
    📞 +91 98765 43210 | ✉️ welldiagnosis@gmail.com
    </center>
    """, unsafe_allow_html=True)
elif menu == "Prediction":

    st.title("🔍 Multi-Disease Prediction System")

    # ================================
    # SELECT DISEASE
    # ================================
    disease = st.selectbox(
        "Select Disease to Diagnose",
        ["Diabetes", "Heart Disease", "ENT Disorder", "Critical Condition", "General Surgery"]
    )

    name = st.text_input("Patient Name")

    # ================================
    # 1. DIABETES
    # ================================
    if disease == "Diabetes":

        st.subheader("🧪 Diabetes Parameters")

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 200, 120)
            bp = st.number_input("Blood Pressure", 0, 150, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)

        with col2:
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
            dpf = st.number_input("DPF", 0.0, 2.5, 0.5)
            age = st.number_input("Age", 1, 120, 30)

        if st.button("Predict Diabetes"):
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            if prediction[0] == 1:
                st.error(f"⚠️ {name} - High Risk of Diabetes")
            else:
                st.success(f"✅ {name} - Low Risk of Diabetes")

    # ================================
    # 2. HEART DISEASE
    # ================================
    elif disease == "Heart Disease":

        st.subheader("❤️ Heart Parameters")

        age_h = st.number_input("Age", 1, 120, 45)
        chol = st.number_input("Cholesterol", 100, 400, 200)
        bp_h = st.number_input("Blood Pressure", 80, 200, 120)
        hr = st.number_input("Heart Rate", 60, 200, 90)

        if st.button("Predict Heart Disease"):
            risk = (chol + bp_h + age_h + hr) / 4

            if risk > 160:
                st.error(f"⚠️ {name} - High Risk of Heart Disease")
            else:
                st.success(f"✅ {name} - Low Risk of Heart Disease")

    # ================================
    # 3. ENT DISORDER
    # ================================
    elif disease == "ENT Disorder":

        st.subheader("👂 ENT Parameters")

        fever = st.selectbox("Fever", [0, 1])
        throat_pain = st.selectbox("Throat Pain", [0, 1])
        hearing_loss = st.selectbox("Hearing Loss", [0, 1])

        if st.button("Predict ENT Issue"):
            score = fever + throat_pain + hearing_loss

            if score >= 2:
                st.error(f"⚠️ {name} - Possible ENT Disorder")
            else:
                st.success(f"✅ {name} - Normal Condition")

    # ================================
    # 4. CRITICAL CONDITION
    # ================================
    elif disease == "Critical Condition":

        st.subheader("🚑 Critical Care Parameters")

        oxygen = st.number_input("Oxygen Level (%)", 50, 100, 95)
        pulse = st.number_input("Pulse Rate", 40, 150, 80)
        consciousness = st.selectbox("Consciousness Level", ["Normal", "Drowsy", "Unconscious"])

        if st.button("Check Condition"):
            if oxygen < 90 or pulse > 120 or consciousness != "Normal":
                st.error(f"🚨 {name} - Critical Condition! Immediate Care Needed")
            else:
                st.success(f"✅ {name} - Stable Condition")

    # ================================
    # 5. GENERAL SURGERY
    # ================================
    elif disease == "General Surgery":

        st.subheader("🔪 Surgery Evaluation")

        pain_level = st.slider("Pain Level (1-10)", 1, 10, 5)
        swelling = st.selectbox("Swelling", [0, 1])
        injury = st.selectbox("Recent Injury", [0, 1])

        if st.button("Evaluate Surgery Need"):
            score = pain_level + swelling*2 + injury*2

            if score > 8:
                st.error(f"⚠️ {name} - Surgery May Be Required")
            else:
                st.success(f"✅ {name} - No Immediate Surgery Needed")
      

# ================================
# ABOUT
# ================================
elif menu == "About":
    st.title("About")
    st.write("AI healthcare system using machine learning.")