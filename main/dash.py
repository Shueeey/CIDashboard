import os
import streamlit as st
import pandas as pd
from datetime import datetime

# Check if Plotly is installed
try:
    import plotly.express as px
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    plotly_available = False
    st.warning("Plotly is not installed. Falling back to Streamlit's native charts.")

# Load the datasets
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    print(script_dir)

    # Use pd.read_csv for CSV files
    data4 = pd.read_csv('/mount/src/cidashboard/main/data4.csv')
    data5 = pd.read_csv('/mount/src/cidashboard/main/data5.csv')
    data6 = pd.read_csv('/mount/src/cidashboard/main/data6.csv')
    data7 = pd.read_csv('/mount/src/cidashboard/main/data7.csv')

    # Convert 'Aggregation Date' to datetime for better handling
    data4['Aggregation Date'] = pd.to_datetime(data4['Aggregation Date'])
    data5['Aggregation Date'] = pd.to_datetime(data5['Aggregation Date'])
    data6['Aggregation Date'] = pd.to_datetime(data6['Aggregation Date'])

    return data4, data5, data6, data7

data4, data5, data6, data7 = load_data()

# Function to compute correlation for numeric columns only
def compute_correlation(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    return numeric_data.corr()

# Set the title of the app
st.title('📊 App Data Utilization Dashboard')

# Sidebar for navigation and filters
st.sidebar.title("🔍 Navigation & Filters")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Dataset 4", "Dataset 5", "Dataset 6",