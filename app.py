import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Salary Predictor Pro", page_icon="💰", layout="centered")

# ==============================
# Enhanced Custom CSS
# ==============================
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d3436 100%);
        color: white;
    }

    /* Modern Title */
    .title-text {
        font-family: 'Helvetica Neue', sans-serif;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 50px;
        text-align: center;
        margin-bottom: 0px;
    }

    /* Animated Gradient Button */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #00dbde 0%, #fc00ff 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-weight: bold;
        font-size: 20px;
        border-radius: 12px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        width: 100%;
        margin-top: 20px;
    }

    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(252, 0, 255, 0.4);
        border: none;
        color: white;
    }

    /* Result Box Styling */
    .result-card {
        background: rgba(0, 210, 255, 0.1);
        border-left: 5px solid #00d2ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 25px;
    }

    /* Label Styling for better visibility */
    .stSelectbox label, .stNumberInput label {
        color: #00d2ff !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load Model & Data
# ==============================
@st.cache_resource
def load_assets():
    # Ensure these files are in the same directory as app.py
    model = load_model("salary_model.keras")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

try:
    model, scaler, columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")

# ==============================
# UI Components
# ==============================
st.markdown('<h1 class="title-text">Salary AI Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#aaa; margin-bottom:40px;">Precision Neural Network Analysis</p>', unsafe_allow_html=True)

# Input Layout
col1, col2 = st.columns(2)

with col1:
    education = st.selectbox(
        "🎓 Education Level",
        ["Bachelor's", "Master's", "PhD"]
    )
    
with col2:
    experience = st.number_input(
        "📅 Years of Experience",
        min_value=0.0,
        max_value=50.0,
        step=0.5
    )

job_title = st.selectbox(
    "💼 Job Title",
    ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director"]
)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================
# Prediction Logic
# ==============================
if st.button("🚀 CALCULATE ESTIMATED SALARY"):
    # 1. Prepare input data
    new_data = pd.DataFrame({
        "Education Level": [education],
        "Job Title": [job_title],
        "Years of Experience": [experience]
    })

    # 2. Preprocess (One-hot encoding and scaling)
    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=columns, fill_value=0)
    new_scaled = scaler.transform(new_data)

    # 3. Predict
    prediction = model.predict(new_scaled)
    predicted_salary = prediction[0][0]

    # 4. Display Result (Styled Card)
    st.markdown(f"""
        <div class="result-card">
            <p style="margin:0; font-size:16px; color:#aaa;">ESTIMATED ANNUAL GROSS</p>
            <h2 style="margin:0; color:#00d2ff; font-size:40px;">₹ {round(predicted_salary, 2):,}</h2>
        </div>
    """, unsafe_allow_html=True)

# Note: st.balloons() has been removed as per your request.