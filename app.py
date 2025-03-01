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
    # In a real app, you would load your data from a file
    # For demonstration, we'll create a sample dataframe similar to your example
    data = {
        'neighborhood_name': ['حي/ال غليط', 'حي/السودة', 'حي/السودة', 'حي/المحالة ابو نخله', 
                            'حي/المحالة بمدينة سلطان', 'حي/الواقعة في قرية القرية', 
                            'حي/سد/السرحة حجاب مدينة سلطان', 'حي/سد/السرحة حجاب مدينة سلطان',
                            'حي/سد/السرحة حجاب مدينة سلطان', 'حي/سد/السرحة حجاب مدينة سلطان',
                            'حي/سد/السرحة حجاب مدينة سلطان'],
        'classification_name': ['سكني', 'سكني', 'سكني', 'سكني', 'سكني', 'سكني', 
                              'سكني', 'سكني', 'سكني', 'سكني', 'سكني'],
        'property_type_name': ['قطعة أرض', 'قطعة أرض', 'قطعة أرض', 'قطعة أرض', 'قطعة أرض', 
                             'قطعة أرض', 'قطعة أرض', 'قطعة أرض', 'قطعة أرض', 'قطعة أرض', 
                             'قطعة أرض'],
        'price_in_SAR': [500000, 300000, 450000, 530000, 125000, 3000000, 
                        300000, 300000, 150000, 150000, 150000],
        'area': [540, 900, 900, 735, 750, 4590, 530.97, 536, 300, 337.23, 335]
    }
    df = pd.DataFrame(data)
    return df
# Load the data
df = load_data()
# Function to preprocess data
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Encode neighborhood names
    processed_df['neighborhood_encoded'] = pd.factorize(processed_df['neighborhood_name'])[0]
    
    # Create price per square meter feature
    processed_df['price_per_sqm'] = processed_df['price_in_SAR'] / processed_df['area']
    
    # Keep only numerical columns for modeling
    numerical_df = processed_df[['neighborhood_encoded', 'area', 'price_in_SAR']]
    
    return processed_df, numerical_df
# Function to train model
def train_model(df):
    # Preprocess the data
    processed_df, numerical_df = preprocess_data(df)
    
    # Define features and target
    X = numerical_df[['neighborhood_encoded', 'area']]
    y = numerical_df['price_in_SAR']
    
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
# Home page
if page == "Home":
    st.header("Welcome to the Abha Real Estate Price Predictor")
    st.write("""
    This application uses machine learning to predict real estate prices in Abha, Saudi Arabia.
    
    **Features:**
    - Predict property prices based on neighborhood, area, and other features
    - Explore the dataset with interactive visualizations
    - Learn about the model's performance metrics
    
    Navigate using the sidebar to explore different sections of the app.
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Abha_City.jpg/1200px-Abha_City.jpg", 
             caption="Abha City, Saudi Arabia")
# Prediction page
elif page == "Prediction":
    st.header("Predict Real Estate Prices")
    
    # Train the model
    model, processed_df, mae, rmse, r2 = train_model(df)
    
    # Get unique neighborhoods
    neighborhoods = df['neighborhood_name'].unique()
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Enter Property Details")
        
        # Neighborhood selection
        neighborhood = st.selectbox("Select Neighborhood", neighborhoods)
        
        # Area input
        area = st.number_input("Property Area (sqm)", min_value=100, max_value=5000, value=500)
        
        # Submit button
        submitted = st.form_submit_button("Predict Price")
        
        if submitted:
            # Preprocess input
            neighborhood_encoded = processed_df[processed_df['neighborhood_name'] == neighborhood]['neighborhood_encoded'].values[0]
            
            # Make prediction
            input_data = pd.DataFrame({
                'neighborhood_encoded': [neighborhood_encoded],
                'area': [area]
            })
            
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"Predicted Price: {prediction:,.2f} SAR")
            
            # Display price per square meter
            price_per_sqm = prediction / area
            st.info(f"Price per Square Meter: {price_per_sqm:,.2f} SAR")
    
    # Display model metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", f"{mae:,.2f} SAR")
    col2.metric("Root Mean Squared Error", f"{rmse:,.2f} SAR")
    col3.metric("R² Score", f"{r2:.2f}")
# Dataset page
elif page == "Dataset":
    st.header("Dataset Overview")
    
    # Display the data
    st.subheader("Raw Data")
    st.dataframe(df)
    
    # Basic statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Data visualizations
    st.subheader("Visualizations")
    
    # Histogram of prices
    st.bar_chart(df['price_in_SAR'])
    
    # Scatter plot of area vs price
    st.subheader("Area vs Price")
    chart_data = pd.DataFrame({
        'Area (sqm)': df['area'],
        'Price (SAR)': df['price_in_SAR']
    })
    st.scatter_chart(chart_data)
# About page
else:
    st.header("About")
    st.write("""
    This application was built using Streamlit and Scikit-learn to demonstrate how machine learning 
    can be applied to real estate price prediction in Abha, Saudi Arabia.
    
    **Model Details:**
    - Algorithm: Random Forest Regressor
    - Features: Neighborhood, Property Area
    - Target: Property Price in SAR
    
    **Future Improvements:**
    - Add more features such as property age, number of rooms, etc.
    - Implement more advanced algorithms like Gradient Boosting or Neural Networks
    - Increase the dataset size for better predictions
    """)
    
    st.subheader("Resources")
    st.markdown("""
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
    - [Random Forest Algorithm](https://scikit-learn.org/stable/modules/ensemble.html#forest)
    """)
