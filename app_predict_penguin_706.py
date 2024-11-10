
import streamlit as st
import pickle
import pandas as pd

# Load the model and encoders
with open('model_penguin_706.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app layout
st.title('Penguin Species Prediction')

# Create user input fields
st.sidebar.header('Input Features')

# User inputs
species = st.sidebar.selectbox('Species', species_encoder.classes_)
island = st.sidebar.selectbox('Island', island_encoder.classes_)
sex = st.sidebar.selectbox('Sex', sex_encoder.classes_)

# Slider for numeric inputs
bill_length_mm = st.sidebar.slider('Bill Length (mm)', 30.0, 60.0, 45.0)
bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 10.0, 25.0, 18.0)
flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 170.0, 240.0, 200.0)
body_mass_g = st.sidebar.slider('Body Mass (g)', 2500.0, 6000.0, 4000.0)

# Prepare the input data (ensure columns are in the same order as expected by the model)
input_data = pd.DataFrame({
    'species': [species],
    'island': [island],
    'sex': [sex],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g]
})

# Apply encoding to categorical features (check column names here!)
input_data['species'] = species_encoder.transform(input_data['species'])
input_data['island'] = island_encoder.transform(input_data['island'])
input_data['sex'] = sex_encoder.transform(input_data['sex'])

# Ensure the columns are in the correct order
expected_columns = ['species', 'island', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
input_data = input_data[expected_columns]

# Make prediction
prediction = model.predict(input_data)

# Show the result
st.write(f'Predicted Species: {species_encoder.inverse_transform(prediction)}')

