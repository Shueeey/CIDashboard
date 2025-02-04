import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Combined Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Load all datasets
@st.cache_data
def load_all_data():
    # Load App Utilization data
    data4 = pd.read_csv('/mount/src/cidashboard/main/data4.csv')
    data5 = pd.read_csv('/mount/src/cidashboard/main/data5.csv')
    data6 = pd.read_csv('/mount/src/cidashboard/main/data6.csv')
    data7 = pd.read_csv('/mount/src/cidashboard/main/data7.csv')

    # Convert dates for app data
    for df in [data4, data5, data6]:
        df['Aggregation Date'] = pd.to_datetime(df['Aggregation Date'])

    # Load SSC data
    ssc_data = pd.read_csv('SSC.csv')

    # Convert SSC date columns
    date_columns = ['Date', 'Closed Date']
    for col in date_columns:
        ssc_data[col] = pd.to_datetime(ssc_data[col], errors='coerce')

    # Create month and year columns for SSC data
    ssc_data['Month'] = ssc_data['Date'].dt.month
    ssc_data['Year'] = ssc_data['Date'].dt.year
    ssc_data['MonthYear'] = ssc_data['Date'].dt.strftime('%Y-%m')

    return data4, data5, data6, data7, ssc_data


# Load all data
data4, data5, data6, data7, ssc_data = load_all_data()


# Function to compute correlation
def compute_correlation(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    return numeric_data.corr()


# Sidebar navigation
st.sidebar.title("🔍 Dashboard Navigation")
main_page = st.sidebar.selectbox(
    "Select Dashboard",
    ["App Utilization Analytics", "Ideas Management Dashboard"]
)

if main_page == "App Utilization Analytics":
    st.title('📱 App Utilization Analytics')

    # Dataset selection for app data
    dataset_choice = st.sidebar.selectbox(
        "Choose Dataset",
        ["App Launches by Platform", "Daily App Launches", "Active Users", "Player Versions"]
    )

    # Date range filter for time-series data
    if dataset_choice in ["App Launches by Platform", "Daily App Launches", "Active Users"]:
        min_date = data4['Aggregation Date'].min()
        max_date = data4['Aggregation Date'].max()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

    # Display appropriate dataset
    if dataset_choice == "App Launches by Platform":
        st.header("📊 App Launch Analysis by Platform")

        filtered_data = data4[
            (data4['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data4['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        # KPI metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Launches", f"{filtered_data['App Launch Count'].sum():,}")
        col2.metric("Platforms", filtered_data['Device Platform'].nunique())
        col3.metric("Avg Daily Launches", f"{filtered_data['App Launch Count'].mean():,.0f}")

        # Platform comparison
        fig = px.line(filtered_data,
                      x='Aggregation Date',
                      y='App Launch Count',
                      color='Device Platform',
                      title='App Launches by Platform Over Time')
        st.plotly_chart(fig, use_container_width=True)

        # Platform distribution
        platform_dist = filtered_data.groupby('Device Platform')['App Launch Count'].sum()
        fig_pie = px.pie(values=platform_dist.values,
                         names=platform_dist.index,
                         title='Launch Distribution by Platform')
        st.plotly_chart(fig_pie, use_container_width=True)

    elif dataset_choice == "Daily App Launches":
        st.header("📈 Daily App Launch Trends")

        filtered_data = data5[
            (data5['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data5['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        # KPI metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Launches", f"{filtered_data['App Launch Count'].sum():,}")
        col2.metric("Daily Average", f"{filtered_data['App Launch Count'].mean():,.0f}")

        # Daily trend
        fig = px.line(filtered_data,
                      x='Aggregation Date',
                      y='App Launch Count',
                      title='Daily App Launch Trend')
        fig.add_trace(go.Scatter(
            x=filtered_data['Aggregation Date'],
            y=filtered_data['App Launch Count'].rolling(7).mean(),
            name='7-day Moving Average',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig, use_container_width=True)

    elif dataset_choice == "Active Users":
        st.header("👥 Active Users Analysis")

        filtered_data = data6[
            (data6['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data6['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        # KPI metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Active Users", f"{filtered_data['Active Users'].sum():,}")
        col2.metric("Daily Average", f"{filtered_data['Active Users'].mean():,.0f}")

        # Active users trend
        fig = px.area(filtered_data,
                      x='Aggregation Date',
                      y='Active Users',
                      title='Active Users Over Time')
        st.plotly_chart(fig, use_container_width=True)

    else:  # Player Versions
        st.header("🎮 Player Version Distribution")

        # KPI metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Users", f"{data7['Active Users'].sum():,}")
        col2.metric("Versions", data7['Player Version'].nunique())

        # Version distribution
        fig = px.bar(data7,
                     x='Player Version',
                     y='Active Users',
                     title='Users by Player Version')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

else:  # Ideas Management Dashboard
    st.title("💡 Ideas Management Dashboard")

    # Date filter
    years = sorted(ssc_data['Year'].dropna().unique())
    selected_year = st.sidebar.selectbox('Select Year', years, index=len(years) - 1)

    # Team filter
    all_teams = sorted(ssc_data['Team'].dropna().unique())
    selected_teams = st.sidebar.multiselect('Select Teams', all_teams, default=all_teams)

    # Filter data
    filtered_data = ssc_data[
        (ssc_data['Year'] == selected_year) &
        (ssc_data['Team'].isin(selected_teams))
        ]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "👥 Team Analysis", "📈 Trends"])

    with tab1:
        # KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ideas", len(filtered_data))
        with col2:
            completed = len(filtered_data[filtered_data['State'] == 'Completed'])
            completion_rate = round((completed / len(filtered_data)) * 100, 1)
            st.metric("Completion Rate", f"{completion_rate}%")
        with col3:
            high_priority = len(filtered_data[filtered_data['Priority  Level'] == 'High'])
            st.metric("High Priority", high_priority)
        with col4:
            st.metric("Active Teams", filtered_data['Team'].nunique())

        # Ideas status
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(filtered_data,
                         names='State',
                         title='Ideas by Status',
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            priority_counts = filtered_data['Priority  Level'].value_counts()
            fig = px.bar(x=priority_counts.index,
                         y=priority_counts.values,
                         title='Ideas by Priority')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Team performance
        col1, col2 = st.columns(2)
        with col1:
            team_counts = filtered_data['Team'].value_counts()
            fig = px.bar(x=team_counts.index,
                         y=team_counts.values,
                         title='Ideas by Team')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            team_completion = (filtered_data[filtered_data['State'] == 'Completed']
                               .groupby('Team').size() /
                               filtered_data.groupby('Team').size() * 100).round(1)
            fig = px.bar(x=team_completion.index,
                         y=team_completion.values,
                         title='Team Completion Rate (%)')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Monthly trends
        monthly_ideas = filtered_data.groupby('MonthYear').size().reset_index()
        monthly_ideas.columns = ['Month', 'Count']
        fig = px.line(monthly_ideas,
                      x='Month',
                      y='Count',
                      title='Monthly Ideas Submission Trend')
        st.plotly_chart(fig, use_container_width=True)

        # State distribution heatmap
        state_team_dist = pd.crosstab(filtered_data['Team'], filtered_data['State'])
        fig = px.imshow(state_team_dist,
                        title='Team vs State Distribution',
                        aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

# Add data table with filters
st.sidebar.title("📊 Additional Options")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("🔍 Raw Data View")
    if main_page == "App Utilization Analytics":
        if dataset_choice == "App Launches by Platform":
            st.dataframe(filtered_data)
        elif dataset_choice == "Daily App Launches":
            st.dataframe(filtered_data)
        elif dataset_choice == "Active Users":
            st.dataframe(filtered_data)
        else:
            st.dataframe(data7)
    else:
        columns_to_display = st.multiselect(
            "Select columns to display",
            ssc_data.columns.tolist(),
            default=['Title', 'Team', 'State', 'Priority  Level', 'Lead', 'Date']
        )
        st.dataframe(filtered_data[columns_to_display])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Analytics Dashboard** - v1.0")