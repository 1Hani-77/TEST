import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# Page config
st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# App title
st.title('🏠 Real Estate Price Prediction App')
st.info('This app predicts real estate prices based on property features!')

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/1Hani-77/TEST/refs/heads/main/abha%20real%20estate.csv"
    df = pd.read_csv(url)
    
    # Data validation
    required_columns = ['neighborhood_name', 'classification_name', 'property_type_name', 'area', 'price']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert to numeric and clean data
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
    df['area'] = pd.to_numeric(df['area'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
   
   # Remove outliers using IQR method
Q1 = df[['price', 'area']].quantile(0.05)
Q3 = df[['price', 'area']].quantile(0.95)
IQR = Q3 - Q1
df = df[~((df[['price', 'area']] < (Q1 - 1.5 * IQR)) | 
         (df[['price', 'area']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df.dropna(subset=['price', 'area'])

try:
    df = load_data()
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Property Details")
        neighborhood = st.selectbox("Neighborhood", sorted(df['neighborhood_name'].unique()))
        classification = st.selectbox("Classification", sorted(df['classification_name'].unique()))
        property_type = st.selectbox("Property Type", sorted(df['property_type_name'].unique()))
        area = st.slider("Area (m²)", 
                        min_value=float(df['area'].quantile(0.05)),
                        max_value=float(df['area'].quantile(0.95)),
                        value=float(df['area'].median()))
        
        st.header("Model Settings")
        n_estimators = st.slider("Number of Trees", 50, 300, 150)
        max_depth = st.slider("Max Tree Depth", 2, 30, 15)

    # Model training and evaluation
    @st.cache_resource
    def train_model(df, n_estimators, max_depth):
        X = pd.get_dummies(df[['neighborhood_name', 'classification_name', 
                              'property_type_name', 'area']], drop_first=True)
        y = df['price']
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                    random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Retrain on full dataset
        model_full = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          random_state=42, n_jobs=-1)
        model_full.fit(X, y)
        
        return model_full, X.columns, metrics

    # Train and get metrics
    model, feature_names, metrics = train_model(df, n_estimators, max_depth)

    # Prediction section
    st.write("## Price Prediction")
    input_data = pd.DataFrame([{
        'neighborhood_name': neighborhood,
        'classification_name': classification,
        'property_type_name': property_type,
        'area': area
    }])
    
    X_input = pd.get_dummies(input_data).reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(X_input)[0]
    
    # Display prediction with style
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Price", f"${prediction:,.2f}")
    with col2:
        avg_price = df[df['neighborhood_name'] == neighborhood]['price'].mean()
        diff = prediction - avg_price
        st.metric("Neighborhood Average", f"${avg_price:,.2f}", 
                 delta=f"{diff:+,.2f} vs Average")

    # Model insights
    with st.expander("Model Performance"):
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{metrics['r2']:.2%}")
        col2.metric("MAE", f"${metrics['mae']:,.2f}")
        col3.metric("RMSE", f"${metrics['rmse']:,.2f}")
        
        # Actual vs Predicted plot
        X_train, X_test, y_train, y_test = train_test_split(
            pd.get_dummies(df[['neighborhood_name', 'classification_name', 
                             'property_type_name', 'area']], drop_first=True),
            df['price'], test_size=0.2, random_state=42
        )
        y_pred = model.predict(X_test)
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
                        title='Actual vs Predicted Prices')
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                     x1=y_test.max(), y1=y_test.max())
        st.plotly_chart(fig)

    # Data exploration
    with st.expander("Data Exploration"):
        tab1, tab2, tab3 = st.tabs(["Distribution", "Relationships", "Geospatial"])
        
        with tab1:
            col = st.selectbox("Select Feature", ['price', 'area', 'neighborhood_name'])
            fig = px.histogram(df, x=col, title=f"{col.title()} Distribution")
            st.plotly_chart(fig)
        
        with tab2:
            color_by = st.selectbox("Color by", ['neighborhood_name', 'classification_name'])
            fig = px.scatter(df, x='area', y='price', color=color_by,
                           hover_name='property_type_name',
                           title='Area vs Price Relationship')
            st.plotly_chart(fig)
        
        with tab3:
            st.warning("Geospatial features coming soon!")

    # Similar properties
    with st.expander("Comparable Properties"):
        similar = df[
            (df['neighborhood_name'] == neighborhood) &
            (df['area'].between(area*0.8, area*1.2))
        ].sort_values('price')
        
        if not similar.empty:
            st.write(f"Found {len(similar)} similar properties:")
            st.dataframe(similar)
            
            fig = px.scatter(similar, x='area', y='price', color='classification_name',
                            size='price', hover_data=['property_type_name'],
                            title='Similar Properties Comparison')
            st.plotly_chart(fig)
        else:
            st.info("No similar properties found in this area range")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your input parameters and try again")
