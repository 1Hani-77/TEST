import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from logging import Logger

# Constants -------------------------------------------------------------------
CONFIG = {
    "data_url": "https://raw.githubusercontent.com/1Hani-77/TEST/main/abha%20real%20estate.csv",
    "required_cols": [
        'neighborhood_name', 
        'classification_name',
        'property_type_name', 
        'area', 
        'price'
    ],
    "model_params": {
        'n_estimators': 200,
        'max_depth': 20,
        'random_state': 42,
        'n_jobs': -1
    },
    "quantile_range": (0.05, 0.95),
    "outlier_threshold": 1.5,
    "comparison_range": 0.2
}

# Helper Functions ------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame) -> None:
    """Perform comprehensive data validation"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    
    missing = [col for col in CONFIG['required_cols'] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    if df.empty:
        raise ValueError("Dataset contains no entries after cleaning")

def create_preprocessor() -> ColumnTransformer:
    """Create sklearn preprocessing pipeline"""
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), [
                'neighborhood_name',
                'classification_name',
                'property_type_name'
            ])
        ],
        remainder='passthrough'
    )

# Data Processing -------------------------------------------------------------
@st.cache_data
def load_and_clean_data(logger: Logger = None) -> pd.DataFrame:
    """Load and preprocess real estate data with robust validation"""
    try:
        df = pd.read_csv(CONFIG['data_url'])
        
        # Initial validation
        validate_dataframe(df)
        
        # Clean numerical columns
        for col in ['price', 'area']:
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                )
                .abs()
                .replace(0, np.nan)
            )
        
        # Outlier detection
        Q1, Q3 = df[['price', 'area']].quantile(CONFIG['quantile_range'])
        IQR = Q3 - Q1
        outlier_mask = (
            (df[['price', 'area']] < (Q1 - CONFIG['outlier_threshold'] * IQR)) | 
            (df[['price', 'area']] > (Q3 + CONFIG['outlier_threshold'] * IQR))
        ).any(axis=1)
        
        clean_df = df[~outlier_mask].dropna()
        validate_dataframe(clean_df)
        
        # Post-cleaning validation
        if logger:
            logger.info(f"Data cleaned: {len(clean_df)}/{len(df)} records retained")
            
        return clean_df

    except Exception as e:
        error_msg = f"Data processing failed: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)

# Model Training --------------------------------------------------------------
@st.cache_resource
def train_model(_df: pd.DataFrame) -> tuple:
    """Train and evaluate pricing model with sklearn pipeline"""
    try:
        preprocessor = create_preprocessor()
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(**CONFIG['model_params']))
        ])
        
        X = _df.drop('price', axis=1)
        y = _df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return model, metrics

    except Exception as e:
        error_msg = f"Model training failed: {str(e)}"
        raise RuntimeError(error_msg)

# Visualization Components ----------------------------------------------------
def create_price_distribution(df: pd.DataFrame) -> go.Figure:
    """Create interactive price distribution histogram"""
    fig = px.histogram(
        df, x='price',
        nbins=50,
        title="Market Price Distribution",
        labels={'price': 'Price (USD)'},
        color_discrete_sequence=['#2c3e50']
    )
    fig.update_layout(
        hovermode='x unified',
        xaxis_title="Price Range",
        yaxis_title="Number of Properties"
    )
    return fig

def create_price_vs_area(df: pd.DataFrame) -> go.Figure:
    """Create interactive price vs area scatter plot"""
    fig = px.scatter(
        df, x='area', y='price',
        color='neighborhood_name',
        title="Price vs Living Area",
        labels={'area': 'Area (m²)', 'price': 'Price (USD)'},
        hover_name='property_type_name',
        trendline="lowess"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig

# Main Application ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Real Estate Valuation Platform",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header Section
    st.title('🏢 AI-Powered Property Valuation')
    st.markdown("""
    *Market insights and valuation powered by machine learning*  
    """)
    
    try:
        # Data Loading
        with st.spinner('Analyzing market data...'):
            df = load_and_clean_data()
        
        # Sidebar Inputs
        with st.sidebar:
            st.header("📋 Property Specifications")
            neighborhood = st.selectbox(
                "Neighborhood District",
                options=sorted(df['neighborhood_name'].unique()),
                help="Select the property's geographical location"
            )
            property_type = st.selectbox(
                "Property Type",
                options=sorted(df['property_type_name'].unique()),
                help="Select the type of property"
            )
            area = st.slider(
                "Total Living Area (m²)",
                min_value=float(df['area'].quantile(0.05)),
                max_value=float(df['area'].quantile(0.95)),
                value=float(df['area'].median()),
                step=1.0,
                format="%d m²"
            )
        
        # Model Training
        with st.spinner('Training valuation model...'):
            model, metrics = train_model(df)
        
        # Prediction
        input_data = pd.DataFrame([{
            'neighborhood_name': neighborhood,
            'classification_name': st.session_state.get('classification', 'Residential'),
            'property_type_name': property_type,
            'area': area
        }])
        
        predicted_price = model.predict(input_data)[0]
        
        # Main Display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Valuation Result")
            st.metric(
                "Estimated Market Value",
                f"${predicted_price:,.0f}",
                delta_color="off"
            )
        
        with col2:
            avg_price = df.loc[
                df['neighborhood_name'] == neighborhood, 'price'
            ].mean()
            st.subheader("Market Context")
            st.metric(
                "Neighborhood Average",
                f"${avg_price:,.0f}",
                delta=f"{predicted_price - avg_price:+,.0f} vs Average"
            )
        
        # Market Insights
        st.header("📈 Market Intelligence")
        tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Model Performance", "Comparable Properties"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_price_distribution(df), use_container_width=True)
            with col2:
                st.plotly_chart(create_price_vs_area(df), use_container_width=True)
        
        with tab2:
            st.subheader("Model Diagnostics")
            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{metrics['r2']:.1%}")
            col2.metric("Mean Absolute Error", f"${metrics['mae']:,.0f}")
            col3.metric("Root Mean Squared Error", f"${metrics['rmse']:,.0f}")
            
            # Actual vs Predicted plot
            X_test = model[:-1].transform(df.drop('price', axis=1))
            y_test = df['price']
            y_pred = model.predict(df.drop('price', axis=1))
            
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
                trendline="lowess",
                title="Model Accuracy: Actual vs Predicted Values"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            similar_properties = df[
                (df['neighborhood_name'] == neighborhood) &
                (df['area'].between(
                    area * (1 - CONFIG['comparison_range']),
                    area * (1 + CONFIG['comparison_range'])
                ))
            ].sort_values('price')
            
            if not similar_properties.empty:
                st.dataframe(
                    similar_properties.head(10),
                    column_config={
                        "price": st.column_config.NumberColumn(
                            "Price", format="$ %.0f")
                    },
                    hide_index=True
                )
            else:
                st.info("No comparable properties found in this area range")

    except Exception as e:
        st.error("""
        ## Application Error
        We encountered an issue processing your request.  
        Please try again or contact support if the problem persists.
        """)
        st.error(f"Technical details: {str(e)}")

if __name__ == "__main__":
    main()
