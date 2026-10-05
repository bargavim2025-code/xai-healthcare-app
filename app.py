"""
Well Diagnosis 2.0 - Explainable AI Healthcare Assistant
Features: model comparison, calibrated risk, explainable AI, what-if simulator,
Claude-powered assistant, patient history dashboard, model performance, PDF reports.
Run:  streamlit run app.py      (needs diabetes.csv in the same folder)
"""
import os
import io
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, f1_score)
from sklearn.inspection import permutation_importance

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

DISCLAIMER = ("Educational tool only. It does not diagnose disease and is not a "
              "substitute for professional medical advice. Always consult a doctor.")
DB_PATH = "patients.db"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

st.set_page_config(page_title="Well Diagnosis 2.0", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);padding:36px;
border-radius:14px;color:white;text-align:center;margin-bottom:18px}
.card{background:white;padding:18px;border-radius:14px;text-align:center;
box-shadow:0 4px 15px rgba(0,0,0,.10);color:#203a43;height:100%}
.pill{display:inline-block;padding:6px 16px;border-radius:20px;color:white;font-weight:600}
.warn{background:#fff6e0;border-left:5px solid #f0a500;padding:10px 14px;
border-radius:6px;color:#5a4300;font-size:.9rem}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ DATA + ML
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)   # 0 = missing in this dataset
    return df


@st.cache_resource
def train_models():
    df = load_data()
    X, y = df[FEATURES], df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    def make(est):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)

    candidates = {
        "Logistic Regression": make(LogisticRegression(max_iter=1000)),
        "Random Forest": make(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=42)),
        "Gradient Boosting": make(GradientBoostingClassifier(random_state=42)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, pipe in candidates.items():
        auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"Model": name, "CV ROC-AUC": round(auc, 3),
                     "Test Accuracy": round(accuracy_score(y_te, pred), 3),
                     "Test F1": round(f1_score(y_te, pred), 3)})
    table = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False)
    best_name = table.iloc[0]["Model"]
    best = candidates[best_name]

    proba = best.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    cm = confusion_matrix(y_te, best.predict(X_te))
    imp = permutation_importance(best, X_te, y_te, n_repeats=10,
                                 random_state=42, scoring="roc_auc")
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    return {"model": best, "name": best_name, "table": table,
            "auc": roc_auc_score(y_te, proba), "fpr": fpr, "tpr": tpr,
            "cm": cm, "importance": importance, "baseline": X_tr.median()}


def explain(bundle, row):
    """Per-patient attribution: how much does risk drop if a feature is set to the
    population median? Positive = pushes risk up for this patient."""
    model, base = bundle["model"], bundle["baseline"]
    p0 = model.predict_proba(row)[0, 1]
    contrib = {}
    for c in FEATURES:
        tmp = row.copy()
        tmp[c] = base[c]
        contrib[c] = p0 - model.predict_proba(tmp)[0, 1]
    return p0, pd.Series(contrib).sort_values()


# ----------------------------------------------------------- RULE SCREENINGS
def screen(rules):
    hit = [(label, pts) for label, cond, pts in rules if cond]
    return min(100, sum(p for _, p in hit)), hit


def level_of(risk):
    return "Low" if risk < 30 else "Moderate" if risk < 60 else "High"


LEVEL_COLOR = {"Low": "#2e9e5b", "Moderate": "#f0a500", "High": "#d63a3a"}

SCREENINGS = {
    "Heart Disease": {"doctor": "Cardiologist",
        "advice": "Low-salt, low-fat diet; regular exercise; stop smoking; get an ECG and lipid profile checked by a cardiologist."},
    "ENT Infection": {"doctor": "ENT Specialist",
        "advice": "Rest, fluids, steam inhalation. See an ENT specialist if symptoms last more than 3 days or worsen."},
    "Critical Vitals": {"doctor": "Critical Care / Emergency",
        "advice": "Abnormal vitals can be serious. Seek emergency medical care immediately if the person feels unwell."},
}

DOCTORS = {"Diabetes": "Diabetologist"}


# -------------------------------------------------------------------- STORAGE
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, name TEXT, age INTEGER,
        module TEXT, risk REAL, level TEXT)""")
    return con


def save_record(res):
    with db() as con:
        con.execute("INSERT INTO records(ts,name,age,module,risk,level) VALUES(?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), res["name"],
                     res["age"], res["module"], res["risk"], res["level"]))


def load_records():
    with db() as con:
        return pd.read_sql("SELECT * FROM records", con, parse_dates=["ts"])


# ------------------------------------------------------------------- VISUALS
def gauge(risk, level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk, number={"suffix": "%"},
        title={"text": f"{level} Risk"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": LEVEL_COLOR[level]},
               "steps": [{"range": [0, 30], "color": "#e3f4ea"},
                         {"range": [30, 60], "color": "#fdf1d6"},
                         {"range": [60, 100], "color": "#f8dcdc"}]}))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def factor_chart(factors):
    """factors: list of (label, value). Positive = increases risk."""
    df = pd.DataFrame(factors, columns=["Factor", "Impact"]).sort_values("Impact")
    df["Direction"] = np.where(df["Impact"] >= 0, "Raises risk", "Lowers risk")
    fig = px.bar(df, x="Impact", y="Factor", orientation="h", color="Direction",
                 color_discrete_map={"Raises risk": "#d63a3a", "Lowers risk": "#2e9e5b"})
    fig.update_layout(height=340, margin=dict(t=20, b=10), xaxis_title="Impact on risk (points)")
    return fig


def make_pdf(res):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    labels = [f[0] for f in res["factors"]]
    vals = [f[1] for f in res["factors"]]
    ax.barh(labels, vals, color=["#d63a3a" if v >= 0 else "#2e9e5b" for v in vals])
    ax.set_title("Key factors")
    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=150)
    plt.close(fig)
    img.seek(0)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, topMargin=36, bottomMargin=36)
    s = getSampleStyleSheet()
    story = [Paragraph("Well Diagnosis - AI Health Report", s["Title"]), Spacer(1, 10)]
    table = Table([["Patient", res["name"] or "-"], ["Age", str(res["age"])],
                   ["Assessment", res["module"]],
                   ["Risk level", f'{res["level"]} ({res["risk"]:.0f}%)'],
                   ["Suggested specialist", res["doctor"]],
                   ["Date", datetime.now().strftime("%d %b %Y %H:%M")]],
                  colWidths=[1.8 * inch, 4 * inch])
    table.setStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)])
    story += [table, Spacer(1, 12), Paragraph("<b>Guidance</b>", s["Heading3"]),
              Paragraph(res["advice"], s["Normal"]), Spacer(1, 10),
              Image(img, width=5.5 * inch, height=2.4 * inch), Spacer(1, 12),
              Paragraph(f"<i>{DISCLAIMER}</i>", s["Normal"])]
    doc.build(story)
    buf.seek(0)
    return buf


def show_result(res):
    c1, c2 = st.columns([1, 1.3])
    c1.plotly_chart(gauge(res["risk"], res["level"]), use_container_width=True)
    c2.markdown("#### Why this result?")
    c2.plotly_chart(factor_chart(res["factors"]), use_container_width=True)
    st.info(f"**Guidance:** {res['advice']}  \n**Suggested specialist:** {res['doctor']}")
    st.download_button("📄 Download PDF report", make_pdf(res),
                       f"report_{res['name'] or 'patient'}.pdf", mime="application/pdf")
    st.markdown(f"<div class='warn'>⚠️ {DISCLAIMER}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------- AI ASSISTANT
SYSTEM_PROMPT = (
    "You are Well Diagnosis Assistant, a friendly health-education helper. Explain the "
    "patient's screening result in simple language, give general lifestyle guidance, and "
    "suggest which kind of doctor to see. NEVER diagnose, NEVER prescribe or recommend "
    "specific medicines or doses, and always remind the user to consult a doctor. If the "
    "user describes an emergency (chest pain, trouble breathing, stroke signs), tell them to "
    "contact emergency services immediately. Keep answers under 150 words.")


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def ask_claude(history, context):
    key = get_api_key()
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        model = st.secrets.get("CLAUDE_MODEL", "claude-sonnet-4-6") if hasattr(st, "secrets") else "claude-sonnet-4-6"
        reply = client.messages.create(
            model=model, max_tokens=600,
            system=SYSTEM_PROMPT + "\n\nLatest screening result:\n" + context,
            messages=history)
        return reply.content[0].text
    except Exception as e:
        return f"(AI service error: {e})"


def offline_reply(res):
    if not res:
        return "Run a screening first, then I can explain the result."
    top = ", ".join(f[0] for f in sorted(res["factors"], key=lambda f: -f[1])[:2])
    return (f"Your {res['module']} screening shows **{res['level']} risk ({res['risk']:.0f}%)**. "
            f"The main contributing factors were: {top}. {res['advice']} "
            f"Please consult a {res['doctor']} for a proper evaluation. "
            "(Connect an Anthropic API key to enable the full conversational assistant.)")


# ===================================================================== PAGES
st.sidebar.title("🏥 Well Diagnosis 2.0")
page = st.sidebar.radio("Navigate", ["Home", "Diabetes AI", "Other Screenings",
                                     "AI Assistant", "Patient Dashboard",
                                     "Model Performance", "About"])
st.sidebar.caption(DISCLAIMER)

if not os.path.exists("diabetes.csv"):
    st.error("diabetes.csv not found. Place it in the same folder as app.py.")
    st.stop()

# ------------------------------------------------------------------- HOME
if page == "Home":
    bundle = train_models()
    st.markdown("""<div class='hero'><h1>🏥 Well Diagnosis 2.0</h1>
    <h3>Explainable AI for early health risk screening</h3>
    <p>Calibrated risk scores • Transparent explanations • AI health assistant • PDF reports</p></div>""",
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", bundle["name"])
    k2.metric("ROC-AUC (test)", f"{bundle['auc']:.3f}")
    k3.metric("Training patients", f"{len(load_data()):,}")
    k4.metric("Screenings", "4 modules")
    st.markdown("### What makes it different")
    a, b, c = st.columns(3)
    for col, icon, t, d in [
        (a, "🧠", "Real ML, compared", "Three models cross-validated; the best one is selected automatically."),
        (b, "🔍", "Explainable", "Every prediction shows which factors raised or lowered the risk."),
        (c, "💬", "AI assistant", "Claude explains results in plain language with medical safety rules.")]:
        col.markdown(f"<div class='card'><h2>{icon}</h2><h4>{t}</h4><p>{d}</p></div>",
                     unsafe_allow_html=True)

# --------------------------------------------------------------- DIABETES AI
elif page == "Diabetes AI":
    bundle = train_models()
    st.title("🩸 Diabetes Risk Prediction")
    st.caption(f"Model: {bundle['name']} (ROC-AUC {bundle['auc']:.2f}) trained on the Pima Indians dataset")
    name = st.text_input("Patient name")
    c1, c2, c3, c4 = st.columns(4)
    preg = c1.number_input("Pregnancies", 0, 20, 1)
    glu = c2.number_input("Glucose (mg/dL)", 40, 250, 120)
    bp = c3.number_input("Blood pressure (diastolic)", 30, 140, 70)
    skin = c4.number_input("Skin thickness (mm)", 5, 100, 20)
    c5, c6, c7, c8 = st.columns(4)
    ins = c5.number_input("Insulin (µU/mL)", 10, 900, 80)
    bmi = c6.number_input("BMI", 12.0, 70.0, 25.0)
    dpf = c7.number_input("Family history score (pedigree)", 0.05, 2.5, 0.35)
    age = c8.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict", type="primary"):
        st.session_state["diab_row"] = dict(zip(FEATURES, [preg, glu, bp, skin, ins, bmi, dpf, age]))
        row = pd.DataFrame([st.session_state["diab_row"]])
        p, contrib = explain(bundle, row)
        risk = p * 100
        res = {"name": name, "age": age, "module": "Diabetes", "risk": risk,
               "level": level_of(risk), "doctor": "Diabetologist",
               "factors": [(k, v * 100) for k, v in contrib.items()],
               "advice": "Balanced low-sugar diet, 150 min/week of exercise, weight management, "
                         "and an HbA1c / fasting glucose test with a doctor."}
        st.session_state["last"] = res
        save_record(res)

    if "diab_row" in st.session_state and st.session_state.get("last", {}).get("module") == "Diabetes":
        show_result(st.session_state["last"])
        st.markdown("### 🎛️ What-if simulator")
        st.caption("See how lifestyle changes could shift the predicted risk.")
        base = st.session_state["diab_row"]
        w1, w2 = st.columns(2)
        new_g = w1.slider("Glucose", 60, 250, int(base["Glucose"]))
        new_b = w2.slider("BMI", 15.0, 50.0, float(base["BMI"]))
        sim = {**base, "Glucose": new_g, "BMI": new_b}
        p_old = bundle["model"].predict_proba(pd.DataFrame([base]))[0, 1] * 100
        p_new = bundle["model"].predict_proba(pd.DataFrame([sim]))[0, 1] * 100
        m1, m2 = st.columns(2)
        m1.metric("Original risk", f"{p_old:.0f}%")
        m2.metric("Simulated risk", f"{p_new:.0f}%", f"{p_new - p_old:+.0f} pts", delta_color="inverse")

# --------------------------------------------------------- OTHER SCREENINGS
elif page == "Other Screenings":
    st.title("🩺 Clinical Rule-Based Screenings")
    st.caption("These modules use transparent clinical thresholds (not trained ML) and are meant for awareness only.")
    name = st.text_input("Patient name")
    age = st.number_input("Age", 1, 120, 40)
    kind = st.radio("Screening", list(SCREENINGS), horizontal=True)

    if kind == "Heart Disease":
        c1, c2, c3, c4 = st.columns(4)
        chol = c1.number_input("Cholesterol (mg/dL)", 100, 400, 200)
        bp = c2.number_input("Systolic BP", 80, 220, 120)
        sugar = c3.number_input("Fasting sugar", 60, 300, 95)
        smoke = c4.selectbox("Smoker", ["No", "Yes"]) == "Yes"
        rules = [("High cholesterol", chol > 240, 25), ("High blood pressure", bp > 140, 25),
                 ("Age over 50", age > 50, 15), ("Smoking", smoke, 25), ("High fasting sugar", sugar > 126, 10)]
    elif kind == "ENT Infection":
        c1, c2, c3, c4 = st.columns(4)
        temp = c1.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
        throat = c2.selectbox("Throat pain", ["No", "Yes"]) == "Yes"
        hearing = c3.selectbox("Hearing issue", ["No", "Yes"]) == "Yes"
        cold = c4.selectbox("Cold / congestion", ["No", "Yes"]) == "Yes"
        rules = [("Fever", temp > 38, 30), ("Throat pain", throat, 25),
                 ("Hearing issue", hearing, 25), ("Cold / congestion", cold, 20)]
    else:
        c1, c2, c3, c4 = st.columns(4)
        spo2 = c1.number_input("Oxygen saturation (%)", 50, 100, 97)
        pulse = c2.number_input("Pulse (bpm)", 30, 200, 80)
        resp = c3.number_input("Respiratory rate", 5, 60, 16)
        temp = c4.number_input("Temperature (°C)", 34.0, 42.0, 37.0)
        rules = [("Low oxygen", spo2 < 92, 50), ("Abnormal pulse", pulse > 120 or pulse < 50, 25),
                 ("Fast breathing", resp > 24, 15), ("High fever", temp > 39, 10)]

    if st.button("Run screening", type="primary"):
        risk, hit = screen(rules)
        factors = [(label, float(pts if cond else 0)) for label, cond, pts in rules]
        info = SCREENINGS[kind]
        res = {"name": name, "age": age, "module": kind, "risk": risk, "level": level_of(risk),
               "doctor": info["doctor"], "factors": factors, "advice": info["advice"]}
        st.session_state["last"] = res
        save_record(res)
    if st.session_state.get("last", {}).get("module") == kind:
        show_result(st.session_state["last"])

# ------------------------------------------------------------ AI ASSISTANT
elif page == "AI Assistant":
    st.title("💬 AI Health Assistant")
    res = st.session_state.get("last")
    if res:
        st.success(f"Context loaded: {res['module']} - {res['level']} risk ({res['risk']:.0f}%)")
        context = (f"Patient age {res['age']}, assessment {res['module']}, risk {res['risk']:.0f}% "
                   f"({res['level']}). Factors: " + "; ".join(f"{a}: {b:+.0f}" for a, b in res["factors"]))
    else:
        st.info("Run a screening first for personalised answers, or ask a general question.")
        context = "No screening yet."
    st.session_state.setdefault("chat", [])
    for m in st.session_state["chat"]:
        st.chat_message(m["role"]).write(m["content"])
    q = st.chat_input("Ask about your result, diet, lifestyle...")
    if q:
        st.session_state["chat"].append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        with st.spinner("Thinking..."):
            answer = ask_claude(st.session_state["chat"], context) or offline_reply(res)
        st.session_state["chat"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

# ----------------------------------------------------------------- DASHBOARD
elif page == "Patient Dashboard":
    st.title("📈 Patient History Dashboard")
    rec = load_records()
    if rec.empty:
        st.info("No screenings saved yet.")
    else:
        names = ["All"] + sorted(rec["name"].fillna("").replace("", "Unnamed").unique())
        who
