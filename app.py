import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="Advanced Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {border-radius: 5px;}
    .stDownloadButton>button {background-color: #4CAF50;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# Session State Initialization
if 'df' not in st.session_state:
    st.session_state.df = None

# Helper Functions
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def generate_report(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.describe().to_excel(writer, sheet_name='Summary')
        df.to_excel(writer, sheet_name='Full Data')
    return buffer

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings")
    analysis_type = st.radio(
        "Analysis Mode",
        ["Exploratory", "Statistical", "Machine Learning"],
        index=0
    )
    
    date_range = st.date_input(
        "Select Date Range",
        value=[datetime.today().replace(day=1), datetime.today()]
    )
    
    theme = st.selectbox(
        "Chart Theme",
        ["plotly", "plotly_white", "plotly_dark"]
    )
    
    st.divider()
    st.markdown("Built with ❤️ using Streamlit")

# Main Content
st.title("📈 Advanced Data Analysis Platform")
st.markdown("---")

# File Upload Section
with st.expander("📤 Data Upload", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV or Excel)",
        type=["csv", "xlsx"],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = load_data(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

# Data Preview
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Data Management
    with st.expander("🔍 Data Overview"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        
        st.dataframe(df.head(10), use_container_width=True)
    
    # Interactive Analysis
    st.markdown("## 🔬 Interactive Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📈 Statistics", "🧩 Data Tools", "📄 Report"])
    
    with tab1:
        st.subheader("Interactive Visualization")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            plot_type = st.selectbox(
                "Choose Visualization",
                ["Scatter Plot", "Histogram", "Line Chart", "Box Plot"]
            )
            x_axis = st.selectbox("X-Axis", df.columns)
            y_axis = st.selectbox("Y-Axis", df.columns) if plot_type != "Histogram" else None
            color_by = st.selectbox("Color By", [None] + list(df.columns))
        
        with col2:
            try:
                if plot_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, template=theme)
                elif plot_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_by, template=theme)
                elif plot_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_by, template=theme)
                elif plot_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_by, template=theme)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
    
    with tab2:
        st.subheader("Statistical Analysis")
        st.write("Descriptive Statistics:")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.write("Correlation Matrix:")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, template=theme)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns for correlation analysis")
    
    with tab3:
        st.subheader("Data Tools")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Filter Data")
            columns_to_show = st.multiselect(
                "Select Columns",
                df.columns,
                default=list(df.columns[:3])
            )
            
            filter_query = st.text_input("Pandas Query (e.g., 'Age > 30')")
        
        with col2:
            st.write("Transformations")
            normalize_col = st.selectbox("Normalize Column", [None] + list(df.columns))
            if normalize_col:
                df[normalize_col] = (df[normalize_col] - df[normalize_col].min()) / \
                                   (df[normalize_col].max() - df[normalize_col].min())
            
            if st.button("Apply Changes"):
                st.success("Transformations applied!")
        
        filtered_df = df[columns_to_show]
        if filter_query:
            try:
                filtered_df = filtered_df.query(filter_query)
            except:
                st.error("Invalid query syntax")
        st.dataframe(filtered_df, use_container_width=True)
    
    with tab4:
        st.subheader("Generate Report")
        if st.button("📥 Create Full Report"):
            report = generate_report(df)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="data_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("👋 Upload a file to begin analysis")
    if st.button("Load Sample Data"):
        st.session_state.df = px.data.iris()
        st.rerun()

# Performance Monitoring
with st.container():
    st.markdown("---")
    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Memory Usage", "2.4 GB", delta="+0.1 GB")
    with col2:
        st.metric("CPU Utilization", "34%", delta="-2%")
    with col3:
        st.metric("Active Users", "1", delta="+0")

# Run with: streamlit run app.py
