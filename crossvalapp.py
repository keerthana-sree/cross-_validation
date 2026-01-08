import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_model():


# Load model
    with open("wine_quality_linear_regression.pkl", "rb") as file:
       model = pickle.load(file)
       return model
model = load_model()

st.set_page_config(page_title="Wine Quality Prediction", layout="centered")
st.title("🍷 Wine Quality Prediction App")

st.write("Enter wine chemical properties to predict **quality score**")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 4.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 1.9)
chlorides = st.number_input("Chlorides", 0.01, 0.3, 0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1.0, 80.0, 11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 6.0, 300.0, 34.0)
density = st.number_input("Density", 0.9900, 1.0050, 0.9978, format="%.4f")
ph = st.number_input("pH", 2.5, 4.5, 3.51)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)
input_data = np.array([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide
]])

# Predict
if st.button("Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, free_sulfur_dioxide,
                             total_sulfur_dioxide, density, ph, sulphates, alcohol]])
    
    prediction = model.predict(input_data)
    st.success(f"🍷 Predicted Wine Quality Score: {prediction[0]:.2f}")
