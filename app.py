# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details and click Predict. The app loads a pre-trained sklearn pipeline `logistic_model.pkl`.")

@st.cache_resource
def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.header("Passenger information")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger class (1 = upper)", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male","female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0, format="%.2f")

embarked = st.selectbox("Port of Embarkation", ["S","C","Q"])

# Build a DataFrame that matches the pipeline's expected input columns
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

st.write("### Preview input")
st.dataframe(input_df)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # probability of class 1
    if pred == 1:
        st.success(f"Likely SURVIVED — probability = {prob:.2%}")
    else:
        st.error(f"Likely DID NOT SURVIVE — probability = {prob:.2%}")

st.write("---")
st.caption("Model: logistic regression pipeline (preprocessing included). Retrain if you change features.")
