import streamlit as st
import pickle
import numpy as np

st.title("🏡 House Price Prediction")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Input fields
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1200)
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)

if st.button("Predict Price"):
    features = np.array([[area, rooms, bathrooms]])
    price = model.predict(features)[0]
    st.success(f"Estimated Price: ₹ {price:,.2f}")
