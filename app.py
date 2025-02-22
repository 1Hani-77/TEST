import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title('🏠 Smart Real Estate Valuator')
st.markdown("""
**Predict property prices** based on location, features, and market trends.
Explore market dynamics through interactive visualizations.
""")

# Constants
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'random_state': 42,
    'n_jobs': -1
}

# Data loading and preprocessing
@st.cache_data
def load_and_clean_data():
    """Load and preprocess real estate data"""
    url = "https://raw.githubusercontent.com/1Hani-77/TEST/main/abha%20real%20estate.csv"
    
    try:
        df = pd.read_csv(url)
        
        # Validate dataset structure
        required_columns = ['neighborhood_name', 'classification_name', 
                           'property_type_name', 'area', 'price']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        # Clean numerical fields
        numeric_cols = ['price', 'area']
        for col in numeric_cols:
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                )
                .abs()
                .replace(0, np.nan)
            )
        
        # Remove outliers using IQR
        Q1 = df[numeric_cols].quantile(0.05)
        Q3 = df[numeric_cols].quantile(0.95)
        IQR = Q3 - Q1
        mask = (
            (df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
            (df[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)
        
        clean_df = df[~mask].dropna(subset=numeric_cols)
        
        if clean_df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        return clean_df

    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

# Model training
@st.cache_resource
def train_price_model(_df):
    """Train and evaluate pricing model"""
    try:
        # Prepare features
        X = pd.get_dummies(
            _df[['neighborhood_name', 'classification_name', 
                'property_type_name', 'area']],
            drop_first=True
        )
        y = _df['price']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return model, X.columns, metrics

    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.stop()

# Main application
try:
    # Load data
    df = load_and_clean_data()
    
    # Sidebar - Property Inputs
    with st.sidebar:
        st.header("🔍 Property Details")
        neighborhood = st.selectbox(
            "Neighborhood",
            options=sorted(df['neighborhood_name'].unique()),
            index=0
        )
        classification = st.selectbox(
            "Property Class",
            options=sorted(df['classification_name'].unique())
        )
        property_type = st.selectbox(
            "Property Type",
            options=sorted(df['property_type_name'].unique())
        )
        area = st.slider(
            "Living Area (m²)",
            min_value=float(df['area'].quantile(0.05)),
            max_value=float(df['area'].quantile(0.95)),
            value=float(df['area'].median()),
            step=1.0
        )

    # Train model
    model, feature_names, metrics = train_price_model(df)

    # Prediction Section
    st.header("💰 Price Prediction")
    
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'neighborhood_name': neighborhood,
        'classification_name': classification,
        'property_type_name': property_type,
        'area': area
    }])
    
    # Prepare features
    X_input = pd.get_dummies(input_data).reindex(
        columns=feature_names, 
        fill_value=0
    )
    
    # Generate prediction
    prediction = model.predict(X_input)[0]
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Estimated Value", 
            f"${prediction:,.0f}",
            help="Predicted market value based on current inputs"
        )
    
    with col2:
        avg_price = df.loc[
            df['neighborhood_name'] == neighborhood, 'price'
        ].mean()
        price_diff = prediction - avg_price
        st.metric(
            "Neighborhood Average", 
            f"${avg_price:,.0f}", 
            delta=f"{price_diff:+,.0f} vs Average",
            delta_color="normal"
        )

    # Market Insights Section
    with st.expander("📊 Market Insights", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Price Analysis", "Model Performance", "Comparables"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df, x='price',
                    title="Price Distribution",
                    labels={'price': 'Price (USD)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    df, x='area', y='price',
                    color='neighborhood_name',
                    title="Price vs Area by Neighborhood",
                    labels={'area': 'Area (m²)', 'price': 'Price (USD)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Accuracy (R²)", f"{metrics['r2']:.1%}")
            col2.metric("Average Error (MAE)", f"${metrics['mae']:,.0f}")
            col3.metric("Error Range (RMSE)", f"${metrics['rmse']:,.0f}")
            
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
                title="Actual vs Predicted Prices"
            )
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                         x1=y_test.max(), y1=y_test.max(),
                         line=dict(color="red", dash="dot"))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            similar = df[
                (df['neighborhood_name'] == neighborhood) &
                (df['area'].between(area*0.8, area*1.2))
            ].sort_values('price')
            
            if not similar.empty:
                st.dataframe(
                    similar.head(10),
                    column_config={
                        "price": st.column_config.NumberColumn(
                            "Price", format="$ %.0f")
                    }
                )
            else:
                st.info("No comparable properties found in this area range")

except Exception as e:
    st.error("Application error: Please check your inputs and try again")
    st.error(f"Technical details: {str(e)}")
