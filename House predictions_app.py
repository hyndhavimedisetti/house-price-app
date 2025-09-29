import streamlit as st
import pickle

# Title
st.title("🏡 House Price Prediction")

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input fields for all 8 features
medinc = st.number_input("Median Income", value=5.0)
house_age = st.number_input("House Age", value=30)
ave_rooms = st.number_input("Average Rooms", value=6.0)
ave_bedrms = st.number_input("Average Bedrooms", value=2.0)
population = st.number_input("Population", value=1000)
ave_occup = st.number_input("Average Occupancy", value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

# Combine inputs into feature array
features = [[medinc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]]

# Predict button
if st.button("Predict"):
    price = model.predict(features)[0]
    st.write(f"Predicted House Price: {price}")


