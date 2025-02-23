import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title('🏠 Real Estate Price Prediction App')
st.info('This app predicts real estate prices in Abha!')

# Load and prepare data
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/1Hani-77/TEST/refs/heads/main/abha%20real%20estate.csv')
    # Display the column names to verify structure
    st.write("Column names:", df.columns.tolist())
    # Drop any rows with missing values
    df = df.dropna()
    st.dataframe(df)
    
    st.write('**Features (X)**')
    # Identify the actual target column name from the dataset
    X_raw = df.drop(['Price'], axis=1)  # Changed from 'price' to 'Price'
    st.dataframe(X_raw)
    
    st.write('**Target (y)**')
    y_raw = df['Price']  # Changed from 'price' to 'Price'
    st.dataframe(y_raw)

# Data visualization
with st.expander('Data visualization'):
    st.scatter_chart(
        data=df,
        x='Size',  # Updated column name
        y='Price', # Updated column name
        color='Property Type'  # Updated column name
    )

# Input features
with st.sidebar:
    st.header('Property Features')
    
    property_type = st.selectbox('Property Type', 
                                df['Property Type'].unique().tolist())
    
    size = st.slider('Size (sq meters)', 
                     float(df['Size'].min()), 
                     float(df['Size'].max()),
                     float(df['Size'].mean()))
    
    rooms = st.slider('Number of Rooms',
                      int(df['Rooms'].min()),
                      int(df['Rooms'].max()),
                      int(df['Rooms'].median()))
    
    bathrooms = st.slider('Number of Bathrooms',
                         int(df['Bathrooms'].min()),
                         int(df['Bathrooms'].max()),
                         int(df['Bathrooms'].median()))
    
    location = st.selectbox('Location',
                           df['Location'].unique().tolist())
    
    # Create DataFrame for input features
    input_data = {
        'Property Type': property_type,
        'Size': size,
        'Rooms': rooms,
        'Bathrooms': bathrooms,
        'Location': location
    }
    input_df = pd.DataFrame(input_data, index=[0])

# Show input data
with st.expander('Input features'):
    st.write('**Selected Property Features**')
    st.dataframe(input_df)

# Data preparation
# Combine input with training data for consistent encoding
input_properties = pd.concat([input_df, X_raw], axis=0)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['Property Type', 'Location']

for col in categorical_cols:
    input_properties[col] = le.fit_transform(input_properties[col])

# Separate back into input and training data
X = input_properties[1:]
input_row = input_properties[:1]

# Model training
# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y_raw)

# Make prediction
prediction = model.predict(input_row)

# Display prediction
st.subheader('Predicted Price')
predicted_price = float(prediction[0])
formatted_price = "{:,.2f}".format(predicted_price)
st.success(f"Estimated Price: SAR {formatted_price}")

# Feature importance
with st.expander('Feature Importance'):
    feature_importance = pd.DataFrame({
        'Feature': input_properties.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    st.bar_chart(feature_importance.set_index('Feature'))
