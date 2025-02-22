import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":rocket:",
    layout="wide"
)

# Add title and header
st.title("My First Streamlit Application")
st.header("Data Analysis Dashboard")

# Create sidebar
with st.sidebar:
    st.header("Configuration")
    user_name = st.text_input("Enter your name")
    age = st.slider("Select your age", 0, 100, 25)
    analysis_type = st.selectbox(
        "Choose analysis type",
        ["Descriptive", "Predictive", "Prescriptive"]
    )
    st.button("Save Settings")

# Main content area
tab1, tab2, tab3 = st.tabs(["Data Input", "Visualization", "Documentation"])

with tab1:
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        st.subheader("Data Summary")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write("Column types:")
        st.json(df.dtypes.astype(str).to_dict())

with tab2:
    st.subheader("Data Visualization")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribution Plot**")
            selected_column = st.selectbox("Select column to plot", df.columns)
            fig, ax = plt.subplots()
            ax.hist(df[selected_column], bins=20)
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Scatter Plot**")
            x_axis = st.selectbox("X-axis", df.columns, index=0)
            y_axis = st.selectbox("Y-axis", df.columns, index=1)
            fig, ax = plt.subplots()
            ax.scatter(df[x_axis], df[y_axis])
            st.pyplot(fig)

with tab3:
    st.subheader("Documentation")
    with st.expander("User Guide"):
        st.markdown("""
        ## Welcome to the Data Analysis Dashboard!
        
        - **Step 1**: Upload your CSV file in the Data Input tab
        - **Step 2**: Configure settings in the sidebar
        - **Step 3**: Explore visualizations in the Visualization tab
        """)
    
    st.write("### About")
    st.write("This application demonstrates Streamlit's capabilities for data analysis.")

# Add some custom components
st.divider()
with st.container():
    st.subheader("Additional Information")
    st.success("Analysis completed successfully!")
    st.warning("This is a warning message")
    st.error("This is an error message")

# Add a progress bar
with st.spinner("Processing data..."):
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
    st.success("Processing complete!")

# Run with: streamlit run app.py
