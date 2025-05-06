import streamlit as st
import pickle
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Fetal Age Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open('fetal_age_model.pkl', 'rb'))

model = load_model()

# Navigation state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Predict"

def switch_page(page: str):
    st.session_state.current_page = page

# Sidebar navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Predict Fetal Age"):
    switch_page("Predict")
if st.sidebar.button("About"):
    switch_page("About")
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit")

# --- Predict Page ---
if st.session_state.current_page == "Predict":
    st.title("Fetal Age Prediction App")
    st.markdown("Enter the 8 clinical and ultrasound measurements below to predict fetal gestational age in **days**.")

    st.header("Enter Input Features")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Maternal Age (years)", min_value=10.0, max_value=60.0)
        hem = st.number_input("Hemoglobin Level (g/dL)", min_value=5.0, max_value=20.0)
        bpd = st.number_input("Biparietal Diameter (mm)", min_value=0.0, max_value=100.0)
        ffl = st.number_input("Femur Length (mm)", min_value=0.0, max_value=100.0)

    with col2:
        hc = st.number_input("Head Circumference (mm)", min_value=0.0, max_value=500.0)
        ac = st.number_input("Abdominal Circumference (mm)", min_value=0.0, max_value=500.0)
        efw = st.number_input("Estimated Fetal Weight (g)", min_value=0.0, max_value=1000.0)
        lmp = st.number_input("Days Since Last Menstrual Period (LMP)", min_value=0.0, max_value=300.0)

    if st.button("Predict"):
        features = np.array([[age, hem, bpd, ffl, hc, ac, efw, lmp]])
        prediction = model.predict(features)
        st.success(f"🍼 Predicted Gestational Age: **{prediction[0]:.0f} days**")

# --- About Page ---
elif st.session_state.current_page == "About":
    st.title("About This App")
    st.markdown("""
    This app predicts **fetal gestational age (in days)** based on 8 ultrasound and maternal clinical features:

    - **Age**: Maternal Age (years)  
    - **Hem**: Hemoglobin Level  
    - **BPD**: Biparietal Diameter  
    - **FFL**: Femur Length  
    - **HC**: Head Circumference  
    - **AC**: Abdominal Circumference  
    - **EFW**: Estimated Fetal Weight  
    - **LMP**: Days Since Last Menstrual Period

    The model used is a trained `RandomForestRegressor.

    """)
