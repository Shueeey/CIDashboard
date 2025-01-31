import os
import streamlit as st
import pandas as pd


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
    return data4, data5, data6, data7


data4, data5, data6, data7 = load_data()

# Set the title of the app
st.title('App Data Utilization Dashboard')

# Display the datasets with a select box
st.sidebar.title("Navigation")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Dataset 4", "Dataset 5", "Dataset 6", "Dataset 7"])

if dataset_choice == "Dataset 4":
    st.header("Dataset 4: App Launch Count by Device Platform")
    st.dataframe(data4)
    st.line_chart(data4.set_index('Aggregation Date')['App Launch Count'])

elif dataset_choice == "Dataset 5":
    st.header("Dataset 5: App Launch Count by Aggregation Date")
    st.dataframe(data5)
    st.bar_chart(data5.set_index('Aggregation Date')['App Launch Count'])

elif dataset_choice == "Dataset 6":
    st.header("Dataset 6: Active Users by Aggregation Date")
    st.dataframe(data6)
    st.area_chart(data6.set_index('Aggregation Date')['Active Users'])

elif dataset_choice == "Dataset 7":
    st.header("Dataset 7: Active Users by Player Version")
    st.dataframe(data7)
    st.bar_chart(data7.set_index('Player Version')['Active Users'])

# Add some additional visualizations and metrics
st.sidebar.title("Additional Visualizations")
if st.sidebar.checkbox("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    if dataset_choice == "Dataset 4":
        st.write(data4.describe())
    elif dataset_choice == "Dataset 5":
        st.write(data5.describe())
    elif dataset_choice == "Dataset 6":
        st.write(data6.describe())
    elif dataset_choice == "Dataset 7":
        st.write(data7.describe())

if st.sidebar.checkbox("Show Correlation Matrix"):
    st.subheader("Correlation Matrix")
    if dataset_choice == "Dataset 4":
        st.write(data4.corr())
    elif dataset_choice == "Dataset 5":
        st.write(data5.corr())
    elif dataset_choice == "Dataset 6":
        st.write(data6.corr())
    elif dataset_choice == "Dataset 7":
        st.write(data7.corr())

# Add a footer
st.sidebar.title("About")
st.sidebar.info(
    "This Streamlit app is designed to visualize and analyze the data utilization of a particular app. "
    "It provides insights into app launch counts, active users, and other relevant metrics."
)