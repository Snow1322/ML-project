import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('happiness_model_gboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the input fields (based on the 2019 dataset)
feature_names = [
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom to make life choices',
    'Generosity',
    'Perceptions of corruption'
]

st.title("🌍 World Happiness Score Predictor")
st.write("Enter the following features to predict the happiness score:")

# Collect user input
user_input = {}
for feature in feature_names:
    user_input[feature] = st.slider(
        label=feature,
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.01
    )

# Convert input to DataFrame for prediction
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict Happiness Score"):
    prediction = model.predict(input_df)
    st.success(f"🎉 Predicted Happiness Score: {prediction[0]:.2f}")
