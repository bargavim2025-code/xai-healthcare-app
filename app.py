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

    st.markdown("<h1 style='text-align:center; color:#6a1b9a;'>🏥 Well Diagnosis</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Advanced Multi-Speciality Hospital</h4>", unsafe_allow_html=True)

    # ================================
    # Banner Image
    # ================================
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1f",
        use_container_width=True
    )

    # ================================
    # SPECIALITIES SECTION
    # ================================
    st.markdown("## 🩺 Specialities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966483.png", width=100)
        st.markdown("### Critical Care")

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
        st.markdown("### ENT")

    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/4320/4320371.png", width=100)
        st.markdown("### Orthopedics")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=100)
        st.markdown("### General Surgery")

    with col5:
        st.image("https://cdn-icons-png.flaticon.com/512/3209/3209265.png", width=100)
        st.markdown("### Cardiology")

    with col6:
        st.image("https://cdn-icons-png.flaticon.com/512/3870/3870822.png", width=100)
        st.markdown("### Diabetes Care")

    # ================================
    # ADDRESS
    # ================================
    st.markdown("## 📍 Address")
    st.info("""
    Well Diagnosis  
    Anna Nagar, Chennai  
    Phone: +91 98765 43210  
    Email: welldiagnosis@gmail.com  
    """)

    # ================================
    # BOOK APPOINTMENT SECTION
    # ================================
    st.markdown("## 📅 Book Appointment")

    with st.form("appointment_form"):
        pname = st.text_input("Patient Name")
        phone = st.text_input("Phone Number")
        doctor = st.selectbox("Select Doctor", [
            "Dr. Ravi Kumar (Cardiologist)",
            "Dr. Priya Sharma (Diabetologist)",
            "Dr. Arjun Mehta (General Physician)"
        ])
        date = st.date_input("Appointment Date")

        submit = st.form_submit_button("Book Appointment")

        if submit:
            st.success(f"✅ Appointment booked for {pname} with {doctor} on {date}")

            st.write("📞 Our team will contact you shortly.")
elif menu == "Prediction":

    st.title("🔍 Diabetes Prediction")

    name = st.text_input("Patient Name")

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

    if st.button("Predict"):

        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        result = "High Risk" if prediction[0] == 1 else "Low Risk"

        if prediction[0] == 1:
            st.error(f"⚠️ {name} - High Risk of Diabetes")
        else:
            st.success(f"✅ {name} - Low Risk of Diabetes")

        st.write("Probability:", probability)

        # Feature Importance
        st.subheader("Feature Importance")
        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        st.pyplot(fig)

        # ================================
        # FIXED PDF (IN-MEMORY)
        # ================================
        def create_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []

            content.append(Paragraph("Kavi Diagnosis", styles['Title']))
            content.append(Spacer(1, 12))

            content.append(Paragraph(f"Patient Name: {name}", styles['Normal']))
            content.append(Paragraph(f"Date: {datetime.now()}", styles['Normal']))
            content.append(Spacer(1, 12))

            content.append(Paragraph(f"Result: {result}", styles['Normal']))
            content.append(Paragraph(f"Probability: {probability}", styles['Normal']))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf_file = create_pdf()

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="report.pdf",
            mime="application/pdf"
        )

# ================================
# ABOUT
# ================================
elif menu == "About":
    st.title("About")
    st.write("AI healthcare system using machine learning.")