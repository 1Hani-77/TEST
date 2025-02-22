import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1. App Setup
st.title('🏠 Real Estate Price Predictor')
st.info('This app predicts property prices using machine learning!')

# 2. Data Loading
with st.expander('Real Estate Data'):
    # Load sample dataset (replace with your data)
    data_url = "https://drive.google.com/file/d/1V2lYixxk2AeBFM5nzIcH82729RbQqBuh/view?usp=drive_link"
    df = pd.read_csv(data_url)
    
    st.write('**Raw Data**')
    st.dataframe(df)
    
    st.write('**Features (X)**')
    X_raw = df.drop('price', axis=1)
    st.dataframe(X_raw)
    
    st.write('**Target (y)**')
    y_raw = df.price
    st.dataframe(y_raw)

# 3. Data Visualization
with st.expander('Market Insights'):
    st.write('**Price Distribution**')
    st.hist_chart(df.price)
    
    st.write('**Area vs Price**')
    st.scatter_chart(df, x='area', y='price', color='neighborhood_name')

# 4. Sidebar Inputs
with st.sidebar:
    st.header('Property Details')
    
    neighborhood_name = st.selectbox('Neighborhood', 
                                   df.neighborhood_name.unique())
    
    classification_name = st.selectbox('Property Classification',
                                     df.classification_name.unique())
    
    property_type_name = st.selectbox('Property Type',
                                    df.property_type_name.unique())
    
    area = st.number_input('Area (sqm)', 
                         min_value=30, 
                         max_value=1000, 
                         value=100)

# 5. Data Preparation
input_data = {
    'neighborhood_name': neighborhood_name,
    'classification_name': classification_name,
    'property_type_name': property_type_name,
    'area': area
}

input_df = pd.DataFrame(input_data, index=[0])
combined_data = pd.concat([input_df, X_raw], axis=0)

# Encode categorical features
encoded_data = pd.get_dummies(combined_data, 
                            columns=['neighborhood_name', 
                                   'classification_name',
                                   'property_type_name'])

# Split back into input/training data
X = encoded_data[1:]
input_encoded = encoded_data[:1]

# 6. Model Training
model = RandomForestRegressor()
model.fit(X, y_raw)

# 7. Prediction
prediction = model.predict(input_encoded)

# 8. Display Results
st.subheader('Price Prediction')
st.metric(label="Estimated Property Value", 
        value=f"${prediction[0]:,.2f}",
        delta="Market Average ${:,.2f}".format(y_raw.mean()))

st.write('**Feature Importance**')
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

st.bar_chart(importances.set_index('Feature'))
