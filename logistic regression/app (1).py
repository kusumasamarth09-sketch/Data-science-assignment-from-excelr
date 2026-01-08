import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('logistic_model.pkl', 'rb'))

# Create the Streamlit app
st.title('Titanic Survival Prediction')

# Input features
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
fare = st.number_input('Fare', min_value=0.0, value=10.0)
embarked_q = st.checkbox('Embarked from Queenstown')
embarked_s = st.checkbox('Embarked from Southampton')

# Create a button to make predictions
if st.button('Predict'):
    # Convert sex to binary
    sex_encoded = 0 if sex == 'Male' else 1

    # Create a DataFrame with input features
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'Fare': [fare],
        'Embarked_Q': [1 if embarked_q else 0],
        'Embarked_S': [1 if embarked_s else 0]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display the prediction
    if prediction == 1:
        st.success('Passenger is likely to survive.')
    else:
        st.error('Passenger is likely to not survive.')

