import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
temp_model = joblib.load("models/temp_model.pkl")
co2_model = joblib.load("models/co2_model.pkl")
sea_model = joblib.load("models/sea_model.pkl")

st.title("🌍 Climate Change Prediction Dashboard")
st.write("Predict Temperature, CO₂ Levels, and Sea Level Rise (2025–2050)")

year = st.slider("Select Year", min_value=2025, max_value=2050, value=2030)
X = pd.DataFrame([[year]], columns=["Year"])

# Predictions
temp_pred = temp_model.predict(X)[0]
co2_pred = co2_model.predict(X)[0]
sea_pred = sea_model.predict(X)[0]

st.subheader(f"📅 Predictions for Year {year}")
st.write(f"🌡️ **Temperature anomaly:** {temp_pred:.3f} °C")
st.write(f"🟩 **CO₂ concentration:** {co2_pred:.2f} ppm")
st.write(f"🌊 **Sea level rise:** {sea_pred:.2f} mm")

# Graphs
st.subheader("📈 Future Trend (2025–2050)")

future_years = np.arange(2025, 2051).reshape(-1, 1)
future_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Temperature": temp_model.predict(future_years),
    "CO2": co2_model.predict(future_years),
    "Sea Level": sea_model.predict(future_years)
})

st.line_chart(future_df.set_index("Year"))
