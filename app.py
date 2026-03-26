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

    from io import BytesIO
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    # ================================
    # HOSPITAL STYLE UI
    # ================================
    st.markdown("""
    <style>
    .main {
        background-color: #eef3f9;
    }

    .header {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }

    div[role="radiogroup"] > label {
        background-color: white;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 10px;
        border: 1px solid #ccc;
        cursor: pointer;
    }

    div[role="radiogroup"] > label:hover {
        background-color: #2c5364;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='header'><h2>🔍 Smart Diagnosis - Well Diagnosis</h2></div>", unsafe_allow_html=True)

    # ================================
    # PATIENT DETAILS
    # ================================
    name = st.text_input("Patient Name")
    age_p = st.number_input("Age", 1, 120, 30)

    # ================================
    # DISEASE SELECTION
    # ================================
    st.markdown("## 🏥 Select Disease")

    disease = st.radio(
        "",
        ["Diabetes", "Heart Disease", "ENT Disorder", "Critical Condition", "General Surgery"],
        horizontal=True
    )

    result = ""
    cause = ""
    treatment = ""
    doctor = ""
    medicine = ""

    # ================================
    # DIABETES
    # ================================
   elif menu == "Prediction":

    import numpy as np
    from io import BytesIO
    from datetime import datetime
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    # ================================
    # UI STYLE
    # ================================
    st.markdown("""
    <style>
    .main {background-color: #eef3f9;}
    .header {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        padding: 20px;
        border-radius: 10px;
        color: yellow;
        text-align: center;
    }
    div[role="radiogroup"] > label {
        background-color: white;
        padding: 10px 15px;
        margin: 5px;
        border-radius: 10px;
        border: 1px solid #ccc;
        cursor: pointer;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #2c5364;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='header'><h2>🔍 Smart Diagnosis - Well Diagnosis</h2></div>", unsafe_allow_html=True)

    # ================================
    # PATIENT DETAILS
    # ================================
    name = st.text_input("Patient Name")
    age_p = st.number_input("Age", 1, 120, 30)

    # ================================
    # DISEASE SELECTION
    # ================================
    st.markdown("## 🏥 Select Disease")

    disease = st.radio(
        "",
        ["Diabetes", "Heart Disease", "ENT Disorder", "Critical Condition", "General Surgery"],
        horizontal=True
    )

    result, cause, treatment, doctor, medicine = "", "", "", "", ""

    # ================================
    # DIABETES
    # ================================
    if disease == "Diabetes":

        glucose = st.number_input("Glucose Level", 0, 200, 120)
        bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
        age_d = st.number_input("Age Factor", 1, 120, 30)
        family_history = st.selectbox("Family History", [0, 1])

        if st.button("Predict"):
            doctor = "Dr. Priya Sharma (Diabetologist)"

            score = (glucose > 140)*2 + (bmi > 30)*2 + (age_d > 45)*1 + (family_history == 1)*1

            if score >= 4:
                result = "High Risk of Diabetes"
                cause = "High glucose, obesity, age factor, and genetic history"
                treatment = "Strict diet control, daily exercise"
                medicine = "Metformin / Insulin"
            else:
                result = "Low Risk"
                cause = "Parameters within normal range"
                treatment = "Maintain healthy lifestyle"
                medicine = "None"

    # ================================
    # HEART
    # ================================
    elif disease == "Heart Disease":

        chol = st.number_input("Cholesterol", 100, 400, 200)
        bp = st.number_input("Blood Pressure", 80, 200, 120)
        age_h = st.number_input("Age", 1, 120, 45)
        smoking = st.selectbox("Smoking", [0, 1])

        if st.button("Predict"):
            doctor = "Dr. Ravi Kumar (Cardiologist)"

            score = (chol > 240)*2 + (bp > 140)*2 + (age_h > 50)*1 + (smoking == 1)*2

            if score >= 4:
                result = "High Risk of Heart Disease"
                cause = "High cholesterol, BP, smoking habits, and age"
                treatment = "Low-fat diet, exercise, avoid smoking"
                medicine = "Aspirin, Statins"
            else:
                result = "Low Risk"
                cause = "Normal cardiovascular parameters"
                treatment = "Healthy lifestyle"
                medicine = "None"

    # ================================
    # ENT
    # ================================
    elif disease == "ENT Disorder":

        temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, 37.0)
        throat = st.selectbox("Throat Pain", [0, 1])
        hearing = st.selectbox("Hearing Issue", [0, 1])
        cold = st.selectbox("Cold/Cough", [0, 1])

        if st.button("Predict"):
            doctor = "Dr. Anjali Verma (ENT Specialist)"

            score = int(temp > 38) + throat + hearing + cold

            if score >= 2:
                result = "ENT Infection"
                cause = "Fever, throat pain, or ear-related symptoms"
                treatment = "Steam inhalation, hydration"
                medicine = "Paracetamol, Antibiotics"
            else:
                result = "Normal"
                cause = "No significant symptoms"
                treatment = "Maintain hygiene"
                medicine = "None"

    # ================================
    # CRITICAL
    # ================================
    elif disease == "Critical Condition":

        oxygen = st.number_input("Oxygen Level (%)", 50, 100, 95)
        pulse = st.number_input("Pulse Rate", 40, 150, 80)
        bp_c = st.number_input("Blood Pressure", 80, 200, 120)

        if st.button("Predict"):
            doctor = "Dr. Karthik Raj (Critical Care)"

            if oxygen < 90 or pulse > 120 or bp_c > 180:
                result = "Critical Condition"
                cause = "Low oxygen, high pulse or BP"
                treatment = "Immediate ICU admission"
                medicine = "Oxygen support, emergency drugs"
            else:
                result = "Stable"
                cause = "Vitals within normal range"
                treatment = "Observation"
                medicine = "None"

    # ================================
    # SURGERY
    # ================================
    elif disease == "General Surgery":

        pain = st.slider("Pain Level", 1, 10, 5)
        swelling = st.selectbox("Swelling", [0, 1])
        injury = st.selectbox("Recent Injury", [0, 1])

        if st.button("Predict"):
            doctor = "Dr. Rajesh Patel (Surgeon)"

            score = pain + (swelling*2) + (injury*2)

            if score > 8:
                result = "Surgery Required"
                cause = "Severe pain, swelling, or injury"
                treatment = "Immediate surgical consultation"
                medicine = "Painkillers"
            else:
                result = "No Surgery Needed"
                cause = "Mild symptoms"
                treatment = "Rest and medication"
                medicine = "Ibuprofen"

    # ================================
    # OUTPUT + PDF
    # ================================
    
    if result != "":
        st.subheader("🩺 Diagnosis Report")

        st.write(f"**Patient:** {name}")
        st.write(f"**Age:** {age_p}")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Result:** {result}")
        st.write(f"**Cause:** {cause}")
        st.write(f"**Treatment:** {treatment}")
        st.write(f"**Doctor:** {doctor}")
        st.write(f"**Medicines:** {medicine}")

        # PDF
        def create_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph("Well Diagnosis Hospital", styles['Title']))
            content.append(Spacer(1, 12))

            content.append(Paragraph(f"Patient: {name}", styles['Normal']))
            content.append(Paragraph(f"Age: {age_p}", styles['Normal']))
            content.append(Paragraph(f"Disease: {disease}", styles['Normal']))
            content.append(Spacer(1, 12))

            content.append(Paragraph(f"Result: {result}", styles['Normal']))
            content.append(Paragraph(f"Cause: {cause}", styles['Normal']))
            content.append(Paragraph(f"Treatment: {treatment}", styles['Normal']))
            content.append(Paragraph(f"Doctor: {doctor}", styles['Normal']))
            content.append(Paragraph(f"Medicines: {medicine}", styles['Normal']))
            content.append(Paragraph(f"Date: {datetime.now()}", styles['Normal']))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf = create_pdf()

        st.download_button(
            "📄 Download Patient Report",
            data=pdf,
            file_name="report.pdf",
            mime="application/pdf"
        )
# ================================
# ABOUT
# ================================
elif menu == "About":
    st.title("About")
    st.write("AI healthcare system using machine learning.")