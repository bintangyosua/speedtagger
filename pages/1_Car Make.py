import streamlit as st
import joblib
import pandas as pd

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write("This app predicts the **car make** based on engine performance and price.")

df = pd.read_csv("car_make_one_hot.csv")

clf_scaler = joblib.load("clf_scaler.pkl")
clf_model = joblib.load("clf_model.pkl")
clf_label_encoder = joblib.load("car_make_label_encoder.pkl")

st.title("🔍 Car Make Prediction")
st.caption("🏷️ Predict the car's manufacturer based on specifications and price")

st.markdown("---")
st.subheader("🛠️ Input Features")

col1, col2 = st.columns(2)

with col1:
    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=1.0,
        max_value=9.0,
        value=2.0,
        step=0.1,
        help="Enter engine size in liters (L)"
    )
    horsepower = st.number_input(
        "Horsepower",
        min_value=100.0,
        max_value=2000.0,
        value=df['horsepower'].median(),
        step=10.0,
        help="Total horsepower produced by the engine"
    )
    year = st.number_input(
        "Year",
        min_value=1900,
        max_value=2025,
        value=2021,
        help="Year the car was manufactured"
    )

with col2:
    torque = st.number_input(
        "Torque (lb-ft)",
        min_value=100.0,
        max_value=1500.0,
        value=df['torque_(lb-ft)'].median(),
        help="Engine torque measured in pound-feet"
    )
    zero_to_sixty = st.number_input(
        "0-60 mph Time (seconds)",
        min_value=0.0,
        max_value=7.0,
        value=df['0-60_mph_time_(seconds)'].median(),
        step=0.1,
        help="Time taken to accelerate from 0 to 60 mph"
    )
    price = st.number_input(
        "Price (in USD)",
        min_value=10000.0,
        max_value=6000000.0,
        value=df['price_(in_usd)'].median(),
        step=1000.0,
        help="Estimated price of the car in USD"
    )

st.markdown("---")

if st.button("🔍 Predict Car Make"):
    # Buat dictionary input
    input_dict = {
        'year': [year],
        'engine_size_(l)': [engine_size],
        'horsepower': [horsepower],
        'torque_(lb-ft)': [torque],
        '0-60_mph_time_(seconds)': [zero_to_sixty],
        'price_(in_usd)': [price],  # Ditaruh paling akhir
    }

    # Buat DataFrame input
    input_df = pd.DataFrame(input_dict)

    # Scaling input
    input_scaled = clf_scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=clf_model.feature_names_in_)

    # Prediksi label
    pred = clf_model.predict(input_scaled_df)

    # Decode label
    pred_label = clf_label_encoder.inverse_transform(pred)[0]

    # Tampilkan hasil
    st.success(f"🏷️ **Predicted Car Make:** {pred_label}")

