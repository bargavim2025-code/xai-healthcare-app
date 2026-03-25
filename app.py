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
st.set_page_config(page_title="AI Healthcare", layout="wide")

# ================================
# Custom CSS (COLORFUL UI)
# ================================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #4facfe, #00f2fe);
}

.main {
    background-color: #f5f7fa;
    padding: 20px;
    border-radius: 10px;
}

h1 {
    color: #2c3e50;
    text-align: center;
}

.stButton>button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stTextInput>div>div>input {
    border-radius: 10px;
}

.stNumberInput input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Title
# ================================
st.markdown("<h1>🏥 AI Healthcare Decision System</h1>", unsafe_allow_html=True)

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
# Accuracy Card
# ================================
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

st.markdown(f"""
<div style='background: #ffffff; padding: 15px; border-radius: 10px; text-align:center;'>
<h3>📊 Model Accuracy</h3>
<h2 style='color: green;'>{accuracy:.2f}</h2>
</div>
""", unsafe_allow_html=True)

# ================================
# Patient Info
# ================================
st.subheader("👤 Patient Details")
name = st.text_input("Enter Patient Name")

# ================================
# Inputs in Columns
# ================================
st.subheader("🧾 Medical Inputs")

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
# Prediction Button
# ================================
if st.button("🔍 Predict Now"):

    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    result = "High Risk" if prediction[0] == 1 else "Low Risk"

    # Result Card
    if prediction[0] == 1:
        st.markdown(f"""
        <div style='background:#ffcccc; padding:20px; border-radius:10px;'>
        <h2 style='color:red;'>⚠️ {name} - High Risk of Diabetes</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#ccffcc; padding:20px; border-radius:10px;'>
        <h2 style='color:green;'>✅ {name} - Low Risk of Diabetes</h2>
        </div>
        """, unsafe_allow_html=True)

    st.write("### 📊 Probability")
    st.write(probability)

    # ================================
    # Feature Importance Chart
    # ================================
    st.subheader("🔍 Feature Importance")

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
    # Extra Visualization
    # ================================
    st.subheader("📈 Glucose Trend")
    st.line_chart(df["Glucose"])

    # ================================
    # Download Report
    # ================================
    report = f"""
    AI Healthcare Report
    Name: {name}
    Date: {datetime.now()}

    Result: {result}
    Probability: {probability}
    """

    st.download_button("📄 Download Report", report, file_name="report.txt")