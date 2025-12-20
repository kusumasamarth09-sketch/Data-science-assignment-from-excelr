
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

@st.cache_data(show_spinner=False)
def load_model(path='model.pkl'):
    data = joblib.load(path)
    return data['pipeline'], data['features']

pipeline, features = load_model('model.pkl')

st.title("Titanic Survival Predictor")
st.write("Enter passenger details and get predicted survival probability (Logistic Regression).")

def build_inputs(features):
    inputs = {}
    if 'Pclass' in features:
        inputs['Pclass'] = st.selectbox("Pclass", options=[1,2,3], index=0)
    if 'Age' in features:
        inputs['Age'] = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    if 'SibSp' in features:
        inputs['SibSp'] = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    if 'Parch' in features:
        inputs['Parch'] = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    if 'Fare' in features:
        inputs['Fare'] = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
    if 'HasCabin' in features:
        inputs['HasCabin'] = st.selectbox("Has Cabin info?", options=[0,1], index=0)
    if any(f.startswith('Sex_') for f in features):
        inputs['Sex'] = st.selectbox("Sex", options=['male','female'])
    if any(f.startswith('Embarked_') for f in features):
        inputs['Embarked'] = st.selectbox("Embarked", options=['C','Q','S'])
    other_features = [f for f in features if f not in ['Pclass','Age','SibSp','Parch','Fare','HasCabin'] and not f.startswith('Sex_') and not f.startswith('Embarked_')]
    for f in other_features:
        try:
            inputs[f] = st.number_input(f, value=0.0)
        except:
            inputs[f] = st.text_input(f, value="")
    return inputs

with st.form("input_form"):
    user_inputs = build_inputs(features)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {}
    for f in features:
        if f in ['Pclass','Age','SibSp','Parch','Fare','HasCabin']:
            row[f] = float(user_inputs.get(f, 0))
        elif f.startswith('Sex_'):
            base = f.split('_',1)[1]
            row[f] = 1.0 if user_inputs.get('Sex','') == base else 0.0
        elif f.startswith('Embarked_'):
            base = f.split('_',1)[1]
            row[f] = 1.0 if user_inputs.get('Embarked','') == base else 0.0
        else:
            try:
                row[f] = float(user_inputs.get(f, 0))
            except:
                row[f] = 0.0
    X = pd.DataFrame([row], columns=features)
    prob = pipeline.predict_proba(X)[:,1][0]
    pred = pipeline.predict(X)[0]
    st.write("Predicted survival probability:", round(prob, 4))
    st.write("Predicted class (0 = not survived, 1 = survived):", int(pred))
    with st.expander("Show input features"):
        st.write(X.T)
