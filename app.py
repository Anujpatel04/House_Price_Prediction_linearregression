import numpy as np
import pickle
import streamlit as st

model = pickle.load(open(r'C:\Users\a\VSCODE_NAREDH-IT\MACHINE-LEARNING\Home-selling-price-prediction\model.pkl', 'rb'))

st.title('Home Selling Price Prediction')

st.write('Enter the square feet living area to predict the price:')
sq_feet = st.number_input(
    'Enter the square feet living area',
    min_value=0.0,
    max_value=20000.0,  
    value=0.0,
    step=0.5
)

if st.button('Predict Price'):
    input_data = np.array([[sq_feet]])
    price = model.predict(input_data)
    st.success(f"The predicted selling price for a home with {sq_feet} square feet is ${price[0]:,.2f}")

st.write('Thank you for using the app! Made by Anuj Patel.')
