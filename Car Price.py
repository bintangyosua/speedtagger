import streamlit as st
import joblib

import pandas as pd

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write("This app predicts **car prices** based on performance and engine specifications.")

df = pd.read_csv('car_make_one_hot.csv')

car_make_label_encoder = joblib.load("car_make_label_encoder.pkl")
car_make_options = list(car_make_label_encoder.classes_)

reg_scaler = joblib.load("reg_scaler.pkl")
y_reg_scaler = joblib.load("y_reg_scaler.pkl")
reg_model = joblib.load("reg_model.pkl")

st.title("🚗 Car Price Prediction")
st.caption("💰 Predict car prices based on engine and performance specs")

st.markdown("---")
st.subheader("🛠️ Input Features")

col1, col2 = st.columns(2)

with col1:
    car_make = st.selectbox(
        "Car Make",
        car_make_options,  # Ganti sesuai dataset
        help="Select the manufacturer of the car"
    )
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

with col2:
    year = st.number_input(
        "Year",
        min_value=1900,
        max_value=2025,
        value=2021,
        help="Year the car was manufactured"
    )
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

st.markdown("---")

if st.button("🔮 Predict Price"):
    # One-hot encoding manual untuk car_make
    all_makes = reg_model.feature_names_in_[5:]  # Ambil nama-nama merek mobil dari model
    car_make_ohe = [1 if make == car_make else 0 for make in all_makes]

    # Buat dictionary input lengkap
    input_dict = {
        'year': [year],
        'engine_size_(l)': [engine_size],
        'horsepower': [horsepower],
        'torque_(lb-ft)': [torque],
        '0-60_mph_time_(seconds)': [zero_to_sixty],
    }
    for i, make in enumerate(all_makes):
        input_dict[make] = [car_make_ohe[i]]

    # Buat DataFrame input
    input_df = pd.DataFrame(input_dict)

    # Scaling input
    input_scaled = reg_scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=reg_model.feature_names_in_)

    # Prediksi harga (dalam bentuk yang masih discaled)
    pred = reg_model.predict(input_scaled_df)

    # Inverse transform agar kembali ke satuan harga sebenarnya
    inverse_pred = y_reg_scaler.inverse_transform(pred.reshape(1, -1))

    # Tampilkan hasilnya
    st.success(f"💵 **Estimated Car Price:** ${inverse_pred[0][0]:,.2f}")
