import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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

# Set the title of the app
st.title('📊 App Data Utilization Dashboard')

# Sidebar for navigation and filters
st.sidebar.title("🔍 Navigation & Filters")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Dataset 4", "Dataset 5", "Dataset 6", "Dataset 7"])

# Add a date range filter for datasets with date columns
if dataset_choice in ["Dataset 4", "Dataset 5", "Dataset 6"]:
    min_date = data4['Aggregation Date'].min()
    max_date = data4['Aggregation Date'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Display the selected dataset
if dataset_choice == "Dataset 4":
    st.header("📈 Dataset 4: App Launch Count by Device Platform")

    # Filter data based on date range
    filtered_data4 = data4[(data4['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
                           (data4['Aggregation Date'] <= pd.to_datetime(date_range[1]))]

    # Display KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total App Launches", filtered_data4['App Launch Count'].sum())
    col2.metric("Unique Devices", filtered_data4['Device Platform'].nunique())
    col3.metric("Average Launches per Day", round(filtered_data4['App Launch Count'].mean(), 2))

    # Interactive Plotly line chart
    fig = px.line(filtered_data4, x='Aggregation Date', y='App Launch Count', color='Device Platform',
                  title='App Launch Count Over Time', labels={'App Launch Count': 'Launch Count', 'Aggregation Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data in an expander
    with st.expander("View Raw Data"):
        st.dataframe(filtered_data4)

elif dataset_choice == "Dataset 5":
    st.header("📊 Dataset 5: App Launch Count by Aggregation Date")

    # Filter data based on date range
    filtered_data5 = data5[(data5['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
                           (data5['Aggregation Date'] <= pd.to_datetime(date_range[1]))]

    # Display KPIs
    col1, col2 = st.columns(2)
    col1.metric("Total App Launches", filtered_data5['App Launch Count'].sum())
    col2.metric("Average Launches per Day", round(filtered_data5['App Launch Count'].mean(), 2))

    # Interactive Plotly bar chart
    fig = px.bar(filtered_data5, x='Aggregation Date', y='App Launch Count',
                 title='App Launch Count by Date', labels={'App Launch Count': 'Launch Count', 'Aggregation Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data in an expander
    with st.expander("View Raw Data"):
        st.dataframe(filtered_data5)

elif dataset_choice == "Dataset 6":
    st.header("📈 Dataset 6: Active Users by Aggregation Date")

    # Filter data based on date range
    filtered_data6 = data6[(data6['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
                           (data6['Aggregation Date'] <= pd.to_datetime(date_range[1]))]

    # Display KPIs
    col1, col2 = st.columns(2)
    col1.metric("Total Active Users", filtered_data6['Active Users'].sum())
    col2.metric("Average Active Users per Day", round(filtered_data6['Active Users'].mean(), 2))

    # Interactive Plotly area chart
    fig = px.area(filtered_data6, x='Aggregation Date', y='Active Users',
                  title='Active Users Over Time', labels={'Active Users': 'Users', 'Aggregation Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data in an expander
    with st.expander("View Raw Data"):
        st.dataframe(filtered_data6)

elif dataset_choice == "Dataset 7":
    st.header("📊 Dataset 7: Active Users by Player Version")

    # Display KPIs
    col1, col2 = st.columns(2)
    col1.metric("Total Active Users", data7['Active Users'].sum())
    col2.metric("Unique Player Versions", data7['Player Version'].nunique())

    # Interactive Plotly bar chart
    fig = px.bar(data7, x='Player Version', y='Active Users',
                 title='Active Users by Player Version', labels={'Active Users': 'Users', 'Player Version': 'Version'})
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data in an expander
    with st.expander("View Raw Data"):
        st.dataframe(data7)

# Additional Visualizations and Metrics
st.sidebar.title("📊 Additional Visualizations")
if st.sidebar.checkbox("Show Summary Statistics"):
    st.subheader("📝 Summary Statistics")
    if dataset_choice == "Dataset 4":
        st.write(filtered_data4.describe())
    elif dataset_choice == "Dataset 5":
        st.write(filtered_data5.describe())
    elif dataset_choice == "Dataset 6":
        st.write(filtered_data6.describe())
    elif dataset_choice == "Dataset 7":
        st.write(data7.describe())

if st.sidebar.checkbox("Show Correlation Matrix"):
    st.subheader("📊 Correlation Matrix")
    if dataset_choice == "Dataset 4":
        st.write(filtered_data4.corr())
    elif dataset_choice == "Dataset 5":
        st.write(filtered_data5.corr())
    elif dataset_choice == "Dataset 6":
        st.write(filtered_data6.corr())
    elif dataset_choice == "Dataset 7":
        st.write(data7.corr())

# Add a footer
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This Streamlit app is designed to visualize and analyze the data utilization of a particular app. "
    "It provides insights into app launch counts, active users, and other relevant metrics."
)