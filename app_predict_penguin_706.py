
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and encoders from the pickle file
@st.cache_resource
def load_model():
    with open('model_penguin_706.pkl', 'rb') as file:
        # Load the entire model and encoders
        model, species_encoder, island_encoder, sex_encoder = pickle.load(file)
    return model, species_encoder, island_encoder, sex_encoder

# Load the model and encoders only once
model, species_encoder, island_encoder, sex_encoder = load_model()

# Streamlit title and description
st.title("Penguin Species Prediction")
st.write("Enter the penguin characteristics below to predict its species.")

# Input fields for user input
island = st.selectbox('Island', ['Torgersen', 'Dream', 'Biscoe'])
culmen_length = st.number_input('Culmen Length (mm)', min_value=0.0, step=0.1, value=37.0)
culmen_depth = st.number_input('Culmen Depth (mm)', min_value=0.0, step=0.1, value=19.3)
flipper_length = st.number_input('Flipper Length (mm)', min_value=0.0, step=0.1, value=192.3)
body_mass = st.number_input('Body Mass (g)', min_value=0.0, step=1.0, value=3750.0)
sex = st.selectbox('Sex', ['MALE', 'FEMALE'])

# Prepare the new data for prediction
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length],
    'culmen_depth_mm': [culmen_depth],
    'flipper_length_mm': [flipper_length],
    'body_mass_g': [body_mass],
    'sex': [sex]
})

# Encode categorical features using the loaded encoders
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Scale numerical features
scaler = StandardScaler()
x_new[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']] = scaler.fit_transform(
    x_new[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
)

# Make predictions using the model
y_pred_new = model.predict(x_new)

# Decode the prediction back to the original species
result = species_encoder.inverse_transform(y_pred_new)

# Display the predicted species
if st.button('Predict'):
    st.write(f"Predicted Species: {result[0]}")

