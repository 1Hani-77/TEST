import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title('🏠 ABHA Real Estate Price Predictor')

st.info('This app predicts real estate prices in ABHA using machine learning!')

# Load dataset
url = 'https://raw.githubusercontent.com/SLW-20/ProjectMIS/master/abha%20real%20estate.csv'
df = pd.read_csv(url)

with st.expander('Raw Data'):
    st.write('**Property Data**')
    st.write(df)
    
    st.write('**Features (X)**')
    X_raw = df.drop('price_sar', axis=1)
    st.write(X_raw)
    
    st.write('**Target (y)**')
    y_raw = df['price_sar']
    st.write(y_raw)

with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x='area_sq_m', y='price_sar', color='property_type')

# Sidebar for user input
with st.sidebar:
    st.header('Property Specifications')
    
    property_type = st.selectbox('Property Type', df['property_type'].unique())
    bedrooms = st.slider('Bedrooms', 
                        min_value=int(df['bedrooms'].min()), 
                        max_value=int(df['bedrooms'].max()),
                        value=int(df['bedrooms'].median()))
    bathrooms = st.slider('Bathrooms', 
                         min_value=int(df['bathrooms'].min()), 
                         max_value=int(df['bathrooms'].max()),
                         value=int(df['bathrooms'].median()))
    area_sq_m = st.slider('Area (sq m)', 
                         min_value=float(df['area_sq_m'].min()), 
                         max_value=float(df['area_sq_m'].max()),
                         value=float(df['area_sq_m'].median()))
    location = st.selectbox('Location', df['location'].unique())

# Create input DataFrame
input_data = {
    'property_type': property_type,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'area_sq_m': area_sq_m,
    'location': location
}
input_df = pd.DataFrame(input_data, index=[0])

# Combine user input with raw features
combined_df = pd.concat([input_df, X_raw], axis=0)

# Data preparation
encode = ['property_type', 'location']
encoded_df = pd.get_dummies(combined_df, columns=encode)

# Split back into user input and features
input_encoded = encoded_df[:1]
X_encoded = encoded_df[1:]

with st.expander('Encoded Features'):
    st.write('**Encoded Input Features**')
    st.write(input_encoded)
    
    st.write('**Encoded Training Data**')
    st.write(X_encoded)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_encoded, y_raw)

# Prediction
prediction = model.predict(input_encoded)

# Display prediction
st.subheader('Price Prediction')
st.success(f'Predicted Property Price: **{prediction[0]:,.2f} SAR**')

st.write('---')
st.write('Model Features Importance:')
importance_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))
