import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# App title
st.title('🏠 ABHA Real Estate Price Predictor')

# Sidebar for user inputs
st.sidebar.header('Upload Your Dataset')
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Show raw data
    st.subheader('Raw Data Preview')
    st.write(df.head())
    
    # Data preprocessing
    st.subheader('Feature Selection')
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    features = st.multiselect('Select features', numeric_cols)
    target = st.selectbox('Select target variable', numeric_cols)
    
    if features and target:
        X = df[features]
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader('Model Performance')
        st.write(f'Mean Absolute Error: {mae:.2f}')
        st.write(f'R² Score: {r2:.2f}')
        
        # Save model
        joblib.dump(model, 'abha_price_model.pkl')
        
        # Prediction interface
        st.subheader('Make a Prediction')
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(feature, value=df[feature].mean())
        
        if st.button('Predict Price'):
            loaded_model = joblib.load('abha_price_model.pkl')
            prediction = loaded_model.predict(pd.DataFrame([input_data]))
            st.success(f'Predicted Price: {prediction[0]:.2f} SAR')
else:
    st.info('Please upload a CSV file to begin analysis')
