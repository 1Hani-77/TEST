import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Abha Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# App title and description
st.title("Abha Real Estate Price Predictor")
st.markdown("""
This app predicts real estate prices in Abha based on property characteristics.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Dataset", "About"])

# Function to load data
@st.cache_data
def load_data():
    # Load data from CSV URL
    data_url = "https://raw.githubusercontent.com/1Hani-77/TEST/refs/heads/main/abha%20real%20estate.csv"
    df = pd.read_csv(data_url)
    
    # Clean up column names (remove any spaces, special characters)
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename columns to match expected names if needed
    column_mapping = {
        'price': 'price_in_SAR',
        'price(sar)': 'price_in_SAR',
        'price in sar': 'price_in_SAR',
        'neighborhood': 'neighborhood_name'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Print column names for debugging
    st.sidebar.write("Dataset columns:", df.columns.tolist())
    
    # Make sure required columns exist
    required_columns = ['neighborhood_name', 'area', 'price_in_SAR']
    for col in required_columns:
        if col not in df.columns:
            st.sidebar.error(f"Required column '{col}' not found. Available columns: {df.columns.tolist()}")
    
    return df

# Load the data
df = load_data()

# Function to preprocess data
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Encode categorical columns if present
    if 'neighborhood_name' in processed_df.columns:
        processed_df['neighborhood_encoded'] = pd.factorize(processed_df['neighborhood_name'])[0]
    
    # Create price per square meter feature if area and price columns exist
    if 'area' in processed_df.columns and 'price_in_SAR' in processed_df.columns:
        processed_df['price_per_sqm'] = processed_df['price_in_SAR'] / processed_df['area']
    
    return processed_df

# Function to train model
def train_model(df):
    # Preprocess the data
    processed_df = preprocess_data(df)
    
    # Define features and target
    # Dynamically select available features
    features = []
    if 'neighborhood_encoded' in processed_df.columns:
        features.append('neighborhood_encoded')
    if 'area' in processed_df.columns:
        features.append('area')
    
    # Make sure we have the target column
    if 'price_in_SAR' not in processed_df.columns:
        st.error("Required column 'price_in_SAR' not found in dataset.")
        return None, processed_df, 0, 0, 0
    
    X = processed_df[features]
    y = processed_df['price_in_SAR']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, processed_df, mae, rmse, r2

# ... keep existing code (Home page section)

# Prediction page
elif page == "Prediction":
    st.header("Predict Real Estate Prices")
    
    # Check if dataframe has the required columns
    required_columns = ['neighborhood_name', 'area', 'price_in_SAR']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.write("Available columns:", df.columns.tolist())
    else:
        # Train the model
        model_result = train_model(df)
        
        if model_result is None:
            st.error("Failed to train model. Please check the dataset.")
        else:
            # ... keep existing code (prediction form and model metrics)

# ... keep existing code (Dataset and About page sections)
