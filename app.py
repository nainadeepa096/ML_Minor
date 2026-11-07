"""Streamlit web app.  Run with:  streamlit run app.py"""
import streamlit as st

from src.predict import load_bundle, predict_price

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Predictor")
st.caption("Estimate the resale price of a used car (prices in lakhs of ₹).")

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

uses_present = "present_price" in bundle["features"]

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", bundle["brands"])
    year = st.number_input("Manufacturing year", 1990, bundle["reference_year"],
                           bundle["reference_year"] - 5)
    kms = st.number_input("Kilometres driven", 0, 500000, 40000, step=1000)
    owner = st.selectbox("Previous owners", [0, 1, 2, 3])
with col2:
    fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
    seller = st.selectbox("Seller type", ["Individual", "Dealer"])
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])
    present = (st.number_input("Current showroom price (lakhs)", 1.0, 100.0, 8.0, step=0.5)
               if uses_present else None)

if st.button("Predict price", type="primary"):
    price = predict_price(bundle, brand=brand, year=year, kms_driven=kms,
                          fuel_type=fuel, seller_type=seller, transmission=trans,
                          owner=owner, present_price=present)
    st.success(f"Estimated resale price: **₹{price:.2f} lakhs** (≈ ₹{price * 1e5:,.0f})")
    st.caption(f"Model: {bundle['model_name']} · Test R² = {bundle['metrics']['r2']:.3f} "
               f"· MAE = ₹{bundle['metrics']['mae_lakhs']:.2f} lakhs")
