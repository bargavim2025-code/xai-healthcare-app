import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================================
# Page Config
# ================================
st.set_page_config(page_title="AI Healthcare App", layout="wide")

st.markdown("<h1 style='text-align: center;'>🏥 Explainable AI Healthcare System</h1>", unsafe_allow_html=True)

# ================================
# Load Dataset
# ================================
df = pd.read_csv("diabetes.csv")

columns = ['Glucose', 'BloodPressure', 'BMI', 'Insulin']
for col in columns:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Train Model
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ================================
# Model Accuracy
# ================================
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
st.write(f"### 📊 Model Accuracy: {accuracy:.2f}")

# ================================
# Patient Info
# ================================
st.subheader("👤 Patient Information")
name = st.text_input("Patient Name")

# ================================
# Input Fields
# ================================
st.subheader("🧾 Enter Medical Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 30)

# ================================
# Prediction
# ================================
if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    result = "High Risk" if prediction[0] == 1 else "Low Risk"

    # Result Display
    if prediction[0] == 1:
        st.error(f"⚠️ {name}, You have High Risk of Diabetes")
    else:
        st.success(f"✅ {name}, You have Low Risk of Diabetes")

    st.write("### 📊 Probability:")
    st.write(probability)

    # ================================
    # Feature Importance (Explanation)
    # ================================
    st.write("### 🔍 Explanation (Feature Importance)")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # ================================
    # Data Visualization
    # ================================
    st.subheader("📈 Glucose Distribution")
    st.line_chart(df["Glucose"])

    # ================================
    # Download Report
    # ================================
    st.subheader("📄 Download Report")

    report = f"""
    AI Healthcare Report
    ---------------------
    Name: {name}
    Date: {datetime.now()}

    Prediction: {result}
    Probability: {probability}

    Input Values:
    Pregnancies: {pregnancies}
    Glucose: {glucose}
    Blood Pressure: {bp}
    Skin Thickness: {skin}
    Insulin: {insulin}
    BMI: {bmi}
    DPF: {dpf}
    Age: {age}
    """

    st.download_button("📥 Download Report", report, file_name="health_report.txt")