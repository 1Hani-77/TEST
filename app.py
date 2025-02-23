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
    # Get target column name dynamically
    price_col = [col for col in df.columns if 'price' in col.lower()][0]
    X_raw = df.drop([price_col], axis=1)
    st.dataframe(X_raw)
    
    st.write('**Target (y)**')
    y_raw = df[price_col]
    st.dataframe(y_raw)

# Data visualization
with st.expander('Data visualization'):
    size_col = [col for col in df.columns if 'size' in col.lower()][0]
    property_type_col = [col for col in df.columns if 'type' in col.lower()][0]
    
    st.scatter_chart(
        data=df,
        x=size_col,
        y=price_col,
        color=property_type_col
    )

# Input features
with st.sidebar:
    st.header('Property Features')
    
    property_type_col = [col for col in df.columns if 'type' in col.lower()][0]
    size_col = [col for col in df.columns if 'size' in col.lower()][0]
    rooms_col = [col for col in df.columns if 'room' in col.lower()][0]
    bathrooms_col = [col for col in df.columns if 'bath' in col.lower()][0]
    location_col = [col for col in df.columns if 'location' in col.lower()][0]
    
    property_type = st.selectbox('Property Type', 
                                df[property_type_col].unique().tolist())
    
    size = st.slider('Size (sq meters)', 
                     float(df[size_col].min()), 
                     float(df[size_col].max()),
                     float(df[size_col].mean()))
    
    rooms = st.slider('Number of Rooms',
                      int(df[rooms_col].min()),
                      int(df[rooms_col].max()),
                      int(df[rooms_col].median()))
    
    bathrooms = st.slider('Number of Bathrooms',
                         int(df[bathrooms_col].min()),
                         int(df[bathrooms_col].max()),
                         int(df[bathrooms_col].median()))
    
    location = st.selectbox('Location',
                           df[location_col].unique().tolist())
    
    # Create DataFrame for input features
    input_data = {
        property_type_col: property_type,
        size_col: size,
        rooms_col: rooms,
        bathrooms_col: bathrooms,
        location_col: location
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
categorical_cols = [property_type_col, location_col]

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
