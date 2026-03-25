import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ================================
# Load Dataset
# ================================
df = pd.read_csv("diabetes.csv")

# Preprocessing
columns = ['Glucose', 'BloodPressure', 'BMI', 'Insulin']
for col in columns:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# ================================
# Streamlit UI
# ================================
st.title("🏥 Explainable AI Healthcare App")
st.write("Predict Diabetes Risk with Explanation")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # Output
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.write("### Probability:")
    st.write(probability)

    # ================================
    # SHAP Explanation
    # ================================
    shap_values = explainer.shap_values(input_scaled)

    st.write("### 🔍 Explanation (SHAP)")
    
    fig, ax = plt.subplots()
    shap.bar_plot(shap_values[1][0], feature_names=X.columns)
    st.pyplot(fig)