import streamlit as st
import pickle
import numpy as np

# Page setup
st.set_page_config(
    page_title="Fetal Age Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open('fetal_age_model.pkl', 'rb'))

model = load_model()

# Session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

def switch_page(page: str):
    st.session_state.current_page = page

# Sidebar navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Overview"):
    switch_page("Overview")
if st.sidebar.button("Predict Fetal Age"):
    switch_page("Predict")
if st.sidebar.button("About"):
    switch_page("About")
st.sidebar.markdown("---")
st.sidebar.write("App built using Streamlit")

# ---------------- OVERVIEW PAGE ----------------
if st.session_state.current_page == "Overview":
    st.title("👩‍🍼 Overview – Fetal Age Prediction")
    st.markdown("""
    **Fetal Age Prediction** is a medical estimation technique used to assess the gestational age of a fetus based on various biometric and maternal health inputs.
    
    In this application, we use a machine learning model trained on features such as:
    
    - **Maternal Age**
    - **Hemoglobin Level**
    - **Biparietal Diameter (BPD)**
    - **Femur Length (FFL)**
    - **Head Circumference (HC)**
    - **Abdominal Circumference (AC)**
    - **Estimated Fetal Weight (EFW)**
    - **Days Since Last Menstrual Period (LMP)**
    
    This prediction helps healthcare providers estimate how far along a pregnancy is and guide prenatal care decisions.
    """)

# ---------------- PREDICT PAGE ----------------
elif st.session_state.current_page == "Predict":
    st.title("Fetal Age Prediction App")
    st.markdown("Enter the 8 clinical and ultrasound values below to predict fetal gestational age (in days).")

    st.header("Enter Inputs")

    
use_sample = st.button("📋 Use Sample Data")

    
    col1, col2 = st.columns(2)
    with col1:
        age_input = st.text_input("Maternal Age (years)", value="28" if use_sample else "", help="Age of the mother")
        hem_input = st.text_input("Hemoglobin Level (g/dL)", value="12.5" if use_sample else "", help="Hemoglobin level of the mother")
        bpd_input = st.text_input("Biparietal Diameter (mm)", value="85.0" if use_sample else "", help="Width of fetal skull")
        ffl_input = st.text_input("Femur Length (mm)", value="65.0" if use_sample else "", help="Length of fetal femur")

    with col2:
        hc_input = st.text_input("Head Circumference (mm)", value="320.0" if use_sample else "", help="Fetal head circumference")
        ac_input = st.text_input("Abdominal Circumference (mm)", value="280.0" if use_sample else "", help="Fetal abdomen circumference")
        efw_input = st.text_input("Estimated Fetal Weight (g)", value="1500.0" if use_sample else "", help="Weight of the fetus in grams")
        lmp_input = st.text_input("Days Since Last Menstrual Period (LMP)", value="200.0" if use_sample else "", help="Days since the last menstrual period")


    if st.button("Predict"):
        try:
            features = np.array([[
                float(age_input), float(hem_input), float(bpd_input), float(ffl_input),
                float(hc_input), float(ac_input), float(efw_input), float(lmp_input)
            ]])
            prediction = model.predict(features)
            st.success(f"🧒 Predicted Gestational Age: **{prediction[0]:.0f} days**")
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# ---------------- ABOUT PAGE ----------------
elif st.session_state.current_page == "About":
    st.title("About This App")
    st.markdown("""
    This app uses a machine learning model trained on real ultrasound data and maternal health inputs to predict the fetal gestational age (in days).
    
    It uses 8 input features, including maternal age, biometric scan data, and estimated fetal weight.
    
    """)
