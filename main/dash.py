import os
import pandas as pd
from datetime import datetime
import calendar
import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Combined Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Check if Plotly is installed
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_available = True
except ImportError:
    plotly_available = False
    st.warning("Plotly is not installed. Using Streamlit's native charts.")

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


def check_dataframe(df, name):
    """Check if a dataframe is valid and not empty"""
    if df is None or df.empty:
        st.error(f"Error: {name} is empty or failed to load.")
        return False
    return True


# Load all datasets
@st.cache_data
def load_all_data():
    try:
        # Dictionary to store all dataframes
        dfs = {}

        # Load App Utilization data
        file_paths = {
            'data4': '/mount/src/cidashboard/main/data4.csv',
            'data5': '/mount/src/cidashboard/main/data5.csv',
            'data6': '/mount/src/cidashboard/main/data6.csv',
            'data7': '/mount/src/cidashboard/main/data7.csv',
            'ssc_data': '/mount/src/cidashboard/main/SSC.csv'
        }

        # Load each file
        for name, path in file_paths.items():
            try:
                dfs[name] = pd.read_csv(path)
            except Exception as e:
                st.error(f"Error loading {name}: {str(e)}")
                return None

        # Convert dates for app data
        for df_name in ['data4', 'data5', 'data6']:
            dfs[df_name]['Aggregation Date'] = pd.to_datetime(dfs[df_name]['Aggregation Date'])

        # Convert SSC date columns
        date_columns = ['Date', 'Closed Date']
        for col in date_columns:
            dfs['ssc_data'][col] = pd.to_datetime(dfs['ssc_data'][col], errors='coerce')

        # Create month and year columns for SSC data
        dfs['ssc_data']['Month'] = dfs['ssc_data']['Date'].dt.month
        dfs['ssc_data']['Year'] = dfs['ssc_data']['Date'].dt.year
        dfs['ssc_data']['MonthYear'] = dfs['ssc_data']['Date'].dt.strftime('%Y-%m')

        return dfs
    except Exception as e:
        st.error(f"Error in data loading process: {str(e)}")
        return None


# Load data and check if it's valid
dfs = load_all_data()

if dfs is None:
    st.error("Failed to load data. Please check the data files and paths.")
    st.stop()

# Verify all required dataframes are present and valid
required_dfs = ['data4', 'data5', 'data6', 'data7', 'ssc_data']
for df_name in required_dfs:
    if df_name not in dfs or not check_dataframe(dfs[df_name], df_name):
        st.stop()

# Assign dataframes to variables for easier access
data4 = dfs['data4']
data5 = dfs['data5']
data6 = dfs['data6']
data7 = dfs['data7']
ssc_data = dfs['ssc_data']

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
        if plotly_available:
            fig = px.line(filtered_data, x='Aggregation Date', y='App Launch Count',
                          color='Device Platform', title='App Launches by Platform Over Time')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(filtered_data.pivot(index='Aggregation Date',
                                              columns='Device Platform',
                                              values='App Launch Count'))

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
        if plotly_available:
            fig = px.line(filtered_data, x='Aggregation Date', y='App Launch Count',
                          title='Daily App Launch Trend')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(filtered_data.set_index('Aggregation Date')['App Launch Count'])

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
        if plotly_available:
            fig = px.area(filtered_data, x='Aggregation Date', y='Active Users',
                          title='Active Users Over Time')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.area_chart(filtered_data.set_index('Aggregation Date')['Active Users'])

    else:  # Player Versions
        st.header("🎮 Player Version Distribution")

        # KPI metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Users", f"{data7['Active Users'].sum():,}")
        col2.metric("Versions", data7['Player Version'].nunique())

        # Version distribution
        if plotly_available:
            fig = px.bar(data7, x='Player Version', y='Active Users',
                         title='Users by Player Version')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(data7.set_index('Player Version')['Active Users'])

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
            state_data = filtered_data['State'].value_counts()
            st.write("Ideas by Status")
            if plotly_available:
                fig = px.pie(values=state_data.values, names=state_data.index,
                             title='Ideas by Status')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(state_data)

        with col2:
            priority_data = filtered_data['Priority  Level'].value_counts()
            st.write("Ideas by Priority")
            if plotly_available:
                fig = px.bar(x=priority_data.index, y=priority_data.values,
                             title='Ideas by Priority')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(priority_data)

    with tab2:
        # Team performance
        col1, col2 = st.columns(2)
        with col1:
            team_data = filtered_data['Team'].value_counts()
            st.write("Ideas by Team")
            if plotly_available:
                fig = px.bar(x=team_data.index, y=team_data.values,
                             title='Ideas by Team')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(team_data)

        with col2:
            team_completion = (filtered_data[filtered_data['State'] == 'Completed']
                               .groupby('Team').size() /
                               filtered_data.groupby('Team').size() * 100).round(1)
            st.write("Team Completion Rate (%)")
            if plotly_available:
                fig = px.bar(x=team_completion.index, y=team_completion.values,
                             title='Team Completion Rate (%)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(team_completion)

    with tab3:
        # Monthly trends
        monthly_ideas = filtered_data.groupby('MonthYear').size()
        st.write("Monthly Ideas Submission Trend")
        if plotly_available:
            fig = px.line(x=monthly_ideas.index, y=monthly_ideas.values,
                          title='Monthly Ideas Submission Trend')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(monthly_ideas)

        # State distribution by team
        state_team_dist = pd.crosstab(filtered_data['Team'], filtered_data['State'])
        st.write("Team vs State Distribution")
        if plotly_available:
            fig = px.imshow(state_team_dist,
                            title='Team vs State Distribution',
                            aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(state_team_dist)

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