import streamlit as st
import pandas as pd
import joblib


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/car_price_model.pkl"
    )


model = load_model()


# ============================================
# TITLE
# ============================================

st.title("🚗 Car Price Prediction")

st.markdown(
    """
    ### Machine Learning Based Car Price Predictor

    Enter the details of a used car and the model
    will estimate its selling price.
    """
)


# ============================================
# INPUTS
# ============================================

st.sidebar.header("Car Details")


present_price = st.sidebar.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    max_value=1000.0,
    value=8.5,
    step=0.1
)


driven_kms = st.sidebar.number_input(
    "Driven Kilometers",
    min_value=0,
    max_value=1000000,
    value=30000,
    step=1000
)


year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2018,
    step=1
)


fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "CNG"
    ]
)


selling_type = st.sidebar.selectbox(
    "Selling Type",
    [
        "Dealer",
        "Individual"
    ]
)


transmission = st.sidebar.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic"
    ]
)


owner = st.sidebar.selectbox(
    "Previous Owners",
    [
        0,
        1,
        2,
        3
    ]
)


# ============================================
# FEATURE ENGINEERING
# ============================================

CURRENT_YEAR = 2026

car_age = CURRENT_YEAR - year


# ============================================
# INPUT DATAFRAME
# ============================================

input_data = pd.DataFrame({
    "Present_Price": [present_price],
    "Driven_kms": [driven_kms],
    "Fuel_Type": [fuel_type],
    "Selling_type": [selling_type],
    "Transmission": [transmission],
    "Owner": [owner],
    "Car_Age": [car_age]
})


# ============================================
# DISPLAY INPUT
# ============================================

st.subheader("Car Information")

st.dataframe(
    input_data,
    use_container_width=True
)


# ============================================
# PREDICTION
# ============================================

if st.button(
    "🚗 Predict Car Price",
    use_container_width=True
):

    prediction = model.predict(
        input_data
    )

    predicted_price = prediction[0]

    st.success(
        f"### Estimated Selling Price: ₹{predicted_price:.2f} Lakhs"
    )

    st.info(
        f"Car Age: {car_age} years"
    )


# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.caption(
    "Car Price Prediction | Machine Learning Project"
)