# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---  ---
# PAGE CONFIG
# ---  ---
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---  ---
# GLOBAL CSS  - dark glassmorphism theme
# ---  ---
st.markdown("""
<style>
/* -- Base -- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e2e8f0;
}

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: .05em; }

/* -- Gradient Header -- */
.hero-header {
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(99,102,241,.35);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-title { font-size: 2rem; font-weight: 800; color: #fff; margin: 0; position: relative; }
.hero-subtitle { font-size: 0.95rem; color: rgba(255,255,255,0.75); margin: 6px 0 0; position: relative; }

/* -- MOBILE RESPONSIVE -- */
@media (max-width: 768px) {
    /* Hero header */
    .hero-header { padding: 18px 16px; margin-bottom: 14px; }
    .hero-title  { font-size: 1.25rem; }
    .hero-subtitle { font-size: 0.78rem; }

    /* KPI cards: 3 per row on mobile instead of 5 */
    .kpi-wrap {
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 14px;
    }
    .kpi-value { font-size: 1.1rem; }
    .kpi-label { font-size: 0.65rem; letter-spacing: 0; }
    .kpi-card  { padding: 10px 6px; }

    /* Glass cards */
    .glass-card { padding: 14px 12px; margin-bottom: 10px; }

    /* Prediction badges */
    .pred-title-high, .pred-title-low { font-size: 0.95rem; }
    .pred-high, .pred-low { padding: 12px 14px; }

    /* Summary grid - keep 2 col but tighter */
    .summary-grid { gap: 6px; }
    .summary-value { font-size: 0.82rem; }
    .summary-label { font-size: 0.65rem; }

    /* Factor bars */
    .factor-name { font-size: 0.75rem; }
    .factor-val  { font-size: 0.68rem; width: 30px; }

    /* Rec/risk items */
    .rec-item, .risk-item { font-size: 0.78rem; padding: 8px 10px; }

    /* Section titles */
    .section-title { font-size: 0.68rem; }

    /* Hide hover effect on mobile (no hover) */
    .glass-card:hover { transform: none; }

    /* Streamlit block container padding */
    .block-container { padding: 0.5rem 0.6rem 1rem !important; }

    /* Streamlit columns - force full width stack */
    [data-testid="column"] { min-width: 100% !important; width: 100% !important; }
}

@media (max-width: 480px) {
    .kpi-wrap { grid-template-columns: repeat(3, 1fr); gap: 5px; }
    .kpi-value { font-size: 1rem; }
    .hero-title { font-size: 1.1rem; }
}

/* -- Glass Cards -- */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 20px 22px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,.25);
    margin-bottom: 16px;
    transition: transform .2s, box-shadow .2s;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,.35); }

/* -- KPI Cards -- */
.kpi-wrap { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 20px; }
.kpi-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
    backdrop-filter: blur(8px);
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border-radius: 12px 12px 0 0;
}
.kpi-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.kpi-value { font-size: 1.6rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
.kpi-bar { height: 3px; background: rgba(255,255,255,.1); border-radius: 2px; margin-top: 10px; }
.kpi-bar-fill { height: 3px; border-radius: 2px; background: linear-gradient(90deg, #6366f1, #a855f7); }

/* -- Section title -- */
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* -- Prediction badge -- */
.pred-high {
    background: linear-gradient(135deg, rgba(239,68,68,.2), rgba(239,68,68,.05));
    border: 1px solid rgba(239,68,68,.4);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.pred-low {
    background: linear-gradient(135deg, rgba(16,185,129,.2), rgba(16,185,129,.05));
    border: 1px solid rgba(16,185,129,.4);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.pred-title-high { font-size: 1.15rem; font-weight: 800; color: #f87171; margin: 0; }
.pred-title-low  { font-size: 1.15rem; font-weight: 800; color: #34d399; margin: 0; }
.pred-sub { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

/* -- Factor bars -- */
.factor-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.factor-dir { font-size: 1rem; width: 18px; text-align: center; }
.factor-name { font-size: 0.82rem; color: #e2e8f0; flex: 1; }
.factor-bar-bg { flex: 2; height: 6px; background: rgba(255,255,255,.08); border-radius: 4px; overflow: hidden; }
.factor-bar-pos { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #f87171, #ef4444); }
.factor-bar-neg { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #34d399, #10b981); }
.factor-val { font-size: 0.75rem; color: #64748b; width: 38px; text-align: right; }

/* -- Summary grid -- */
.summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.summary-item { display: flex; flex-direction: column; }
.summary-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.summary-value { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; margin-top: 2px; }

/* -- Recommendation -- */
.rec-item {
    background: rgba(16,185,129,.08);
    border: 1px solid rgba(16,185,129,.2);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #a7f3d0;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.risk-item {
    background: rgba(239,68,68,.07);
    border: 1px solid rgba(239,68,68,.2);
    border-radius: 8px;
    padding: 9px 14px;
    margin-bottom: 7px;
    font-size: 0.83rem;
    color: #fca5a5;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* -- Streamlit widget overrides -- */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stSlider .stSlider > div { color: #6366f1; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,.04);
    border-radius: 10px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,.08);
}
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ---  ---
# LOAD MODEL & SCALER
# ---  ---
@st.cache_resource
def load_artifacts():
    import os; base = os.path.dirname(os.path.abspath(__file__)); model  = joblib.load(os.path.join(base, "models", "telco_churn_model.pkl"))
    scaler = joblib.load(os.path.join(base, "models", "scaler.pkl"))
    return model, scaler

try:
    model, scaler = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)


# ---  ---
# FEATURE ENGINEERING  (mirrors notebook exactly)
# ---  ---
# Exact 28 columns the scaler/model was trained on (reproduced from notebook)
# Binary cols (LabelEncoder): Senior Citizen, Partner, Dependents, Paperless Billing
#   LabelEncoder alphabetical order: No=0, Yes=1
# Multi cols (get_dummies, drop_first=True, base category = first alphabetically):
#   Multiple Lines      base=No        -> _No phone service, _Yes
#   Internet Service    base=DSL       -> _Fiber optic, _No
#   Online Security     base=No        -> _No internet service, _Yes
#   Online Backup       base=No        -> _No internet service, _Yes
#   Device Protection   base=No        -> _No internet service, _Yes
#   Tech Support        base=No        -> _No internet service, _Yes
#   Streaming TV        base=No        -> _No internet service, _Yes
#   Streaming Movies    base=No        -> _No internet service, _Yes
#   Contract            base=Month-to-month -> _One year, _Two year
#   Payment Method      base=Bank transfer  -> _Credit card, _Electronic check, _Mailed check

EXPECTED_COLS = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Paperless Billing",
    "Monthly Charges",
    "Total Charges",
    "Multiple Lines_No phone service",
    "Multiple Lines_Yes",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Online Security_No internet service",
    "Online Security_Yes",
    "Online Backup_No internet service",
    "Online Backup_Yes",
    "Device Protection_No internet service",
    "Device Protection_Yes",
    "Tech Support_No internet service",
    "Tech Support_Yes",
    "Streaming TV_No internet service",
    "Streaming TV_Yes",
    "Streaming Movies_No internet service",
    "Streaming Movies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "Payment Method_Credit card (automatic)",
    "Payment Method_Electronic check",
    "Payment Method_Mailed check",
]

def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {col: 0 for col in EXPECTED_COLS}

    # --- Binary (LabelEncoder: No=0, Yes=1) ---
    row["Senior Citizen"]  = 1 if inputs["Senior Citizen"]  == "Yes" else 0
    row["Partner"]         = 1 if inputs["Partner"]         == "Yes" else 0
    row["Dependents"]      = 1 if inputs["Dependents"]      == "Yes" else 0
    row["Paperless Billing"] = 1 if inputs["Paperless Billing"] == "Yes" else 0

    # --- Numeric ---
    row["Tenure Months"]   = inputs["Tenure Months"]
    row["Monthly Charges"] = inputs["Monthly Charges"]
    row["Total Charges"]   = inputs["Total Charges"]

    # --- Multiple Lines (base = No) ---
    if inputs["Multiple Lines"] == "No phone service":
        row["Multiple Lines_No phone service"] = 1
    elif inputs["Multiple Lines"] == "Yes":
        row["Multiple Lines_Yes"] = 1

    # --- Internet Service (base = DSL) ---
    if inputs["Internet Service"] == "Fiber optic":
        row["Internet Service_Fiber optic"] = 1
    elif inputs["Internet Service"] == "No":
        row["Internet Service_No"] = 1

    # --- Online Security (base = No) ---
    if inputs["Online Security"] == "No internet service":
        row["Online Security_No internet service"] = 1
    elif inputs["Online Security"] == "Yes":
        row["Online Security_Yes"] = 1

    # --- Online Backup (base = No) ---
    if inputs["Online Backup"] == "No internet service":
        row["Online Backup_No internet service"] = 1
    elif inputs["Online Backup"] == "Yes":
        row["Online Backup_Yes"] = 1

    # --- Device Protection (base = No) ---
    if inputs["Device Protection"] == "No internet service":
        row["Device Protection_No internet service"] = 1
    elif inputs["Device Protection"] == "Yes":
        row["Device Protection_Yes"] = 1

    # --- Tech Support (base = No) ---
    if inputs["Tech Support"] == "No internet service":
        row["Tech Support_No internet service"] = 1
    elif inputs["Tech Support"] == "Yes":
        row["Tech Support_Yes"] = 1

    # --- Streaming TV (base = No) ---
    if inputs["Streaming TV"] == "No internet service":
        row["Streaming TV_No internet service"] = 1
    elif inputs["Streaming TV"] == "Yes":
        row["Streaming TV_Yes"] = 1

    # --- Streaming Movies (base = No) ---
    if inputs["Streaming Movies"] == "No internet service":
        row["Streaming Movies_No internet service"] = 1
    elif inputs["Streaming Movies"] == "Yes":
        row["Streaming Movies_Yes"] = 1

    # --- Contract (base = Month-to-month) ---
    if inputs["Contract"] == "One year":
        row["Contract_One year"] = 1
    elif inputs["Contract"] == "Two year":
        row["Contract_Two year"] = 1

    # --- Payment Method (base = Bank transfer (automatic)) ---
    if inputs["Payment Method"] == "Credit card (automatic)":
        row["Payment Method_Credit card (automatic)"] = 1
    elif inputs["Payment Method"] == "Electronic check":
        row["Payment Method_Electronic check"] = 1
    elif inputs["Payment Method"] == "Mailed check":
        row["Payment Method_Mailed check"] = 1

    return pd.DataFrame([row])[EXPECTED_COLS]


# ---  ---
# RECOMMENDATION ENGINE
# ---  ---
def generate_recommendation(internet, contract, payment, tenure, tech_support, online_security):
    reasons, recommendations = [], []
    if internet == "Fiber optic":
        reasons.append("Fiber Optic Internet")
    if contract == "Month-to-month":
        reasons.append("Month-to-Month Contract")
        recommendations.append("Offer a one-year or two-year contract with a loyalty discount.")
    if payment == "Electronic check":
        reasons.append("Electronic Check Payment")
        recommendations.append("Encourage switching to automatic payments for improved retention.")
    if tenure < 12:
        reasons.append(f"New Customer (Tenure: {tenure} months)")
        recommendations.append("Provide personalised onboarding support and welcome offers.")
    if tech_support == "No":
        reasons.append("No Tech Support")
        recommendations.append("Offer a discounted Tech Support package.")
    if online_security == "No":
        reasons.append("No Online Security")
        recommendations.append("Promote the Online Security add-on to boost engagement.")
    if not reasons:
        reasons.append("No major churn risk factors identified.")
    if not recommendations:
        recommendations.append("Continue monitoring satisfaction and engagement levels.")
    return reasons, recommendations


# ---  ---
# FEATURE IMPORTANCE  (LR coefficients mapped)
# ---  ---
FEATURE_LABELS = {
    "Contract_Two year":                       "Two-Year Contract",
    "Contract_One year":                       "One-Year Contract",
    "Tenure Months":                           "Long Tenure",
    "Tech Support_Yes":                        "Tech Support",
    "Tech Support_No internet service":        "No Internet (Tech Support)",
    "Online Security_Yes":                     "Online Security",
    "Online Security_No internet service":     "No Internet (Security)",
    "Internet Service_Fiber optic":            "Fiber Optic Internet",
    "Internet Service_No":                     "No Internet Service",
    "Payment Method_Electronic check":         "Electronic Check",
    "Payment Method_Mailed check":             "Mailed Check",
    "Payment Method_Credit card (automatic)":  "Credit Card (Auto)",
    "Paperless Billing":                       "Paperless Billing",
    "Monthly Charges":                         "High Monthly Charges",
    "Multiple Lines_Yes":                      "Multiple Lines",
    "Multiple Lines_No phone service":         "No Phone Service",
    "Online Backup_Yes":                       "Online Backup",
    "Device Protection_Yes":                   "Device Protection",
    "Streaming TV_Yes":                        "Streaming TV",
    "Streaming Movies_Yes":                    "Streaming Movies",
    "Senior Citizen":                          "Senior Citizen",
    "Partner":                                 "Has Partner",
    "Dependents":                              "Has Dependents",
}


# ---  ---
# SIDEBAR - INPUT FORM
# ---  ---
with st.sidebar:
    st.markdown("## 🎛️ Customer Profile")
    st.markdown("---")

    st.markdown("**👤 Demographics**")
    senior   = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner  = st.selectbox("Partner",         ["No", "Yes"])
    dependents = st.selectbox("Dependents",    ["No", "Yes"])

    st.markdown("---")
    st.markdown("**📱 Services**")
    internet  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    multi     = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
    sec       = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
    backup    = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
    dev_prot  = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
    tech_sup  = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
    stream_tv = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
    stream_mv = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("---")
    st.markdown("**📋 Billing & Contract**")
    contract  = st.selectbox("Contract",  ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment   = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown("---")
    st.markdown("**💰 Charges**")
    tenure   = st.slider("Tenure (months)", 1, 72, 12)
    monthly  = st.number_input("Monthly Charges (₹)", min_value=0.0, value=65.0, step=0.5, format="%.2f")
    total    = st.number_input("Total Charges (₹)",   min_value=0.0, value=float(round(monthly * tenure, 2)), step=10.0, format="%.2f")

    predict_btn = st.button("🔍 Predict Churn", use_container_width=True, type="primary")


# ---  ---
# HEADER
# ---  ---
st.markdown("""
<div class="hero-header">
  <div class="hero-title">📊 Customer Churn Prediction Dashboard</div>
  <div class="hero-subtitle">Powered by Logistic Regression · Telco Dataset · Built with Streamlit</div>
</div>
""", unsafe_allow_html=True)

# Mobile tip — only visible on small screens via CSS
st.markdown("""
<div class="mobile-tip">
  ☰ Tap the <strong>arrow</strong> (top-left) to open the input panel
</div>
<style>
.mobile-tip {
    display: none;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.8rem;
    color: #a5b4fc;
    margin-bottom: 12px;
    text-align: center;
}
@media (max-width: 768px) {
    .mobile-tip { display: block; }
}
</style>
""", unsafe_allow_html=True)


# ---  ---
# MODEL STATUS & KPI ROW
# ---  ---
if not MODEL_LOADED:
    st.error(f"⚠️ Could not load model: `{MODEL_ERROR}`. Make sure `telco_churn_model.pkl` and `scaler.pkl` are in the same directory.")
    st.stop()

# Static KPI values from your tuned LR
kpis = [("Accuracy", 79.74), ("Precision", 67.82), ("Recall", 58.34), ("F1 Score", 62.74), ("ROC-AUC", 84.23)]

kpi_html = '<div class="kpi-wrap">'
for label, val in kpis:
    fill = val
    kpi_html += f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val:.2f}%</div>
        <div class="kpi-bar"><div class="kpi-bar-fill" style="width:{fill}%"></div></div>
    </div>"""
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)
st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)


# ---  ---
# PREDICTION LOGIC
# ---  ---
inputs = {
    "Senior Citizen": senior, "Partner": partner, "Dependents": dependents,
    "Tenure Months": tenure, "Multiple Lines": multi,
    "Internet Service": internet, "Online Security": sec,
    "Online Backup": backup, "Device Protection": dev_prot,
    "Tech Support": tech_sup, "Streaming TV": stream_tv,
    "Streaming Movies": stream_mv, "Contract": contract,
    "Paperless Billing": paperless, "Payment Method": payment,
    "Monthly Charges": monthly, "Total Charges": total,
}

X   = build_feature_row(inputs)
Xs  = scaler.transform(X)
prob = float(model.predict_proba(Xs)[0][1])
pred = prob >= 0.5

reasons, actions = generate_recommendation(internet, contract, payment, tenure, tech_sup, sec)


# ---  ---
# MAIN LAYOUT  - responsive (stacks on mobile via CSS)
# ---  ---
left, right = st.columns([1, 1], gap="medium")

# --- LEFT: Gauge + Prediction + Customer Summary ---
with left:
    # Gauge chart
    gauge_color = "#ef4444" if pred else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": gauge_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(16,185,129,0.15)"},
                {"range": [40, 65], "color": "rgba(251,191,36,0.15)"},
                {"range": [65, 100],"color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#facc15", "width": 3},
                "thickness": 0.85,
                "value": 50
            },
        },
        title={"text": "Churn Probability", "font": {"color": "#94a3b8", "size": 14}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Prediction badge
    if pred:
        st.markdown("""
        <div class="pred-high">
          <p class="pred-title-high">⚠️ HIGH RISK OF CHURN</p>
          <p class="pred-sub">This customer is likely to leave. Act now.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="pred-low">
          <p class="pred-title-low">✅ Customer Likely to Stay</p>
          <p class="pred-sub">Low churn risk detected. Keep up engagement.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Customer Summary
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Customer Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-grid">
      <div class="summary-item"><span class="summary-label">Senior Citizen</span><span class="summary-value">{senior}</span></div>
      <div class="summary-item"><span class="summary-label">Partner</span><span class="summary-value">{partner}</span></div>
      <div class="summary-item"><span class="summary-label">Dependents</span><span class="summary-value">{dependents}</span></div>
      <div class="summary-item"><span class="summary-label">Tenure</span><span class="summary-value">{tenure} months</span></div>
      <div class="summary-item"><span class="summary-label">Contract</span><span class="summary-value">{contract}</span></div>
      <div class="summary-item"><span class="summary-label">Internet</span><span class="summary-value">{internet}</span></div>
      <div class="summary-item"><span class="summary-label">Monthly</span><span class="summary-value">₹{monthly:.0f}</span></div>
      <div class="summary-item"><span class="summary-label">Total Charges</span><span class="summary-value">₹{total:.0f}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- RIGHT: Feature Contribution + Recommendation ---
with right:
    # Feature Contribution
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Top Churn Factors</div>', unsafe_allow_html=True)

    coef_series = pd.Series(model.coef_[0], index=EXPECTED_COLS)
    top_positive = coef_series.nlargest(5)
    top_negative = coef_series.nsmallest(4)
    top_features = pd.concat([top_positive, top_negative])

    max_abs = coef_series.abs().max()

    for feat, coef in top_features.items():
        label = FEATURE_LABELS.get(feat, feat)
        direction = "↑" if coef > 0 else "↓"
        color_cls = "factor-bar-pos" if coef > 0 else "factor-bar-neg"
        bar_pct = abs(coef) / max_abs * 100
        st.markdown(f"""
        <div class="factor-row">
          <span class="factor-dir">{direction}</span>
          <span class="factor-name">{label}</span>
          <div class="factor-bar-bg"><div class="{color_cls}" style="width:{bar_pct:.0f}%"></div></div>
          <span class="factor-val">{coef:+.2f}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:.72rem;color:#475569;margin-top:6px;'>↑ increases churn risk &nbsp;·&nbsp; ↓ reduces churn risk</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Recommendation
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 AI Recommendation</div>', unsafe_allow_html=True)

    if reasons[0] != "No major churn risk factors identified.":
        st.markdown("<div style='font-size:.8rem;color:#94a3b8;margin-bottom:8px;'>High-Risk Characteristics Detected:</div>", unsafe_allow_html=True)
        for r in reasons:
            st.markdown(f'<div class="risk-item">⚡ {r}</div>', unsafe_allow_html=True)
        st.markdown("<div style='font-size:.8rem;color:#94a3b8;margin:10px 0 8px;'>Recommended Actions:</div>", unsafe_allow_html=True)
        for a in actions:
            st.markdown(f'<div class="rec-item">✅ {a}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rec-item">✅ No major risk factors identified. Continue monitoring customer satisfaction.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.75rem;color:#475569;padding:8px 0;'>"
    "📊 Customer Churn Predictor · Logistic Regression · Telco Dataset"
    "</div>",
    unsafe_allow_html=True
)