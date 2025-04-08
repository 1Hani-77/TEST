import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Configure page
st.set_page_config(page_title="ABHA Real Estate Predictor", page_icon="🏠")

# Load data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/SLW-20/ProjectMIS/master/abha%20real%20estate.csv'
    return pd.read_csv(url)

df = load_data()

# App title
st.title('🏠 ABHA Real Estate Price Prediction')
st.markdown("Predict property prices in Abha using machine learning")

# Main content
with st.expander("📊 View Raw Data"):
    st.dataframe(df, use_container_width=True)
    st.write(f"Dataset shape: {df.shape}")

# Data preprocessing
def preprocess_data(input_df):
    # Combine user input with dataset
    combined = pd.concat([input_df, df.drop('price_sar', axis=1)], axis=0)
    
    # One-hot encode categorical features
    combined_encoded = pd.get_dummies(combined, columns=['property_type', 'location'])
    
    # Separate back into input and features
    input_encoded = combined_encoded[:1]
    features_encoded = combined_encoded[1:]
    
    return input_encoded, features_encoded

# Sidebar inputs
with st.sidebar:
    st.header("🏡 Property Details")
    
    property_type = st.selectbox(
        "Property Type",
        options=df['property_type'].unique()
    )
    
    location = st.selectbox(
        "Location",
        options=df['location'].unique()
    )
    
    bedrooms = st.slider(
        "Bedrooms",
        min_value=int(df['bedrooms'].min()),
        max_value=int(df['bedrooms'].max()),
        value=int(df['bedrooms'].median())
    )
    
    bathrooms = st.slider(
        "Bathrooms",
        min_value=int(df['bathrooms'].min()),
        max_value=int(df['bathrooms'].max()),
        value=int(df['bathrooms'].median())
    )
    
    area_sq_m = st.slider(
        "Area (sq meters)",
        min_value=float(df['area_sq_m'].min()),
        max_value=float(df['area_sq_m'].max()),
        value=float(df['area_sq_m'].median())
    )

# Create input dataframe
input_data = {
    'property_type': property_type,
    'location': location,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'area_sq_m': area_sq_m
}
input_df = pd.DataFrame(input_data, index=[0])

# Preprocess data
input_encoded, X_encoded = preprocess_data(input_df)
y = df['price_sar']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Prediction
prediction = model.predict(input_encoded)

# Display results
st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error", f"{mae:,.2f} SAR")
col2.metric("R² Score", f"{r2:.2f}")

st.subheader("🔮 Price Prediction")
st.metric("Predicted Property Price", f"{prediction[0]:,.2f} SAR")

# Feature importance
st.subheader("📊 Feature Importance")
importances = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
st.bar_chart(importances.set_index('Feature'))

# Save model
joblib.dump(model, 'abha_real_estate_model.pkl')

# Download button for model
with open('abha_real_estate_model.pkl', 'rb') as f:
    st.download_button(
        label="⬇️ Download Model",
        data=f,
        file_name='abha_real_estate_model.pkl',
        mime='application/octet-stream'
    )
