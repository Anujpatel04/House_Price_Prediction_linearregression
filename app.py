import numpy as np
import pickle
import streamlit as st

# Load the model
model = pickle.load(open(r'C:\Users\a\VSCODE_NAREDH-IT\MACHINE-LEARNING\Home-selling-price-prediction\model.pkl', 'rb'))

# Streamlit app title
st.title('Home Selling Price Prediction')

# User input for square feet
st.write('Enter the square feet living area to predict the price:')
sq_feet = st.number_input(
    'Enter the square feet living area',
    min_value=0.0,
    max_value=20000.0,  # Adjust max value if necessary
    value=0.0,
    step=0.5
)

# Predict button
if st.button('Predict Price'):
    input_data = np.array([[sq_feet]])
    price = model.predict(input_data)
    st.success(f"The predicted selling price for a home with {sq_feet} square feet is ${price[0]:,.2f}")

# Footer
st.write('Thank you for using the app! Made by Anuj Patel.')
