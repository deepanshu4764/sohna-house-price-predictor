import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("sohna_price_model.pkl")

# Title of the web app
st.title("Sohna House Price Predictor 🏡")

# Input fields for user to enter property details
area = st.number_input("Area (sq. ft.)", min_value=500, max_value=5000, step=50)
bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
location_score = st.slider("Location Score (1-10)", 1.0, 10.0, 5.0, step=0.1)
age_of_house = st.number_input("Age of House (years)", min_value=0, max_value=50, step=1)

# Predict button
if st.button("Predict Price"):
    # Prepare input data
    features = np.array([[area, bedrooms, bathrooms, location_score, age_of_house]])
    
    # Make prediction
    predicted_price = model.predict(features)[0]
    
    # Display the result
    st.success(f"Estimated House Price: ₹{predicted_price:,.2f}")

# Footer
st.markdown("Developed by [Your Name]")