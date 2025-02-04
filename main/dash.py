import os
import pandas as pd
from datetime import datetime, timedelta
import calendar
import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Ideas Hub Dashboard",
    page_icon="💡",
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


def clean_submitter_name(submitter):
    """Cleans up the submitter name"""
    if pd.isna(submitter):
        return 'Unknown'
    elif ';#' in str(submitter):
        return submitter.split(';#')[0]
    return submitter


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

        # Clean submitter names
        dfs['ssc_data']['SubmittedBy'] = dfs['ssc_data']['SubmittedBy'].apply(clean_submitter_name)

        return dfs
    except Exception as e:
        st.error(f"Error in data loading process: {str(e)}")
        return None


def prepare_line_chart_data(df, x_col, y_col, color_col):
    """Prepare data for line chart by properly aggregating and handling duplicates"""
    try:
        # Group by both date and platform to handle duplicates
        grouped_data = df.groupby([x_col, color_col])[y_col].sum().reset_index()
        # Pivot the data for plotting
        pivot_data = grouped_data.pivot_table(
            index=x_col,
            columns=color_col,
            values=y_col,
            aggfunc='sum'
        ).fillna(0)
        return pivot_data
    except Exception as e:
        st.error(f"Error preparing chart data: {str(e)}")
        return pd.DataFrame()


def safe_division(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if denominator is 0"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


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
    .search-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stExpander {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        margin-bottom: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
dfs = load_all_data()

if dfs is None:
    st.error("Failed to load data. Please check the data files and paths.")
    st.stop()

# Assign dataframes to variables
data4 = dfs['data4']
data5 = dfs['data5']
data6 = dfs['data6']
data7 = dfs['data7']
ssc_data = dfs['ssc_data']

# Sidebar navigation
st.sidebar.title("🔍 Dashboard Navigation")
main_page = st.sidebar.selectbox(
    "Select Dashboard",
    ["Ideas Search & Analytics", "App Utilization Analytics"]
)

if main_page == "Ideas Search & Analytics":
    st.title("💡 Ideas Search & Analytics")

    # Create search container
    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    # Search and filter options in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        search_name = st.text_input("🔍 Search by Submitter", "")

    with col2:
        all_teams = ['All Teams'] + sorted(ssc_data['Team'].dropna().unique().tolist())
        selected_team = st.selectbox('👥 Filter by Team', all_teams)

    with col3:
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", value=None)
        with date_col2:
            end_date = st.date_input("End Date", value=None)

    st.markdown('</div>', unsafe_allow_html=True)

    # Filter data based on search criteria
    filtered_data = ssc_data.copy()

    if search_name:
        filtered_data = filtered_data[
            filtered_data['SubmittedBy'].str.contains(search_name, case=False, na=False)
        ]

    if selected_team != 'All Teams':
        filtered_data = filtered_data[filtered_data['Team'] == selected_team]

    if start_date:
        filtered_data = filtered_data[filtered_data['Date'].dt.date >= start_date]
    if end_date:
        filtered_data = filtered_data[filtered_data['Date'].dt.date <= end_date]

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Ideas", len(filtered_data))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        completed = len(filtered_data[filtered_data['State'] == 'Completed'])
        if len(filtered_data) > 0:
            completion_rate = round((completed / len(filtered_data)) * 100, 1)
        else:
            completion_rate = 0
        st.metric("Completion Rate", f"{completion_rate}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Active Teams", filtered_data['Team'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        high_priority = len(filtered_data[filtered_data['Priority  Level'] == 'High'])
        st.metric("High Priority", high_priority)
        st.markdown('</div>', unsafe_allow_html=True)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Results", "📊 Analytics", "📈 Trends"])

    with tab1:
        if filtered_data.empty:
            st.warning("No ideas found matching your search criteria.")
        else:
            # Group by submitter if only team filter is applied
            if not search_name and selected_team != 'All Teams':
                st.subheader("Ideas by Submitter")
                for submitter in sorted(filtered_data['SubmittedBy'].unique()):
                    with st.expander(f"📤 {submitter}"):
                        submitter_ideas = filtered_data[filtered_data['SubmittedBy'] == submitter]
                        for _, row in submitter_ideas.iterrows():
                            st.write(f"• {row['Title']}")
                            st.write(
                                f"  Date: {row['Date'].strftime('%d/%m/%Y') if pd.notna(row['Date']) else 'No date'}")
                            st.write(f"  State: {row['State']}")
                            if pd.notna(row['Idea']):
                                st.write(f"  Description: {row['Idea']}")
            else:
                # Regular list view
                for _, row in filtered_data.iterrows():
                    with st.expander(f"📌 {row['Title']}"):
                        cols = st.columns(3)
                        with cols[0]:
                            st.write("**Submitter**")
                            st.write(row['SubmittedBy'])
                        with cols[1]:
                            st.write("**Team**")
                            st.write(row['Team'])
                        with cols[2]:
                            st.write("**Date**")
                            st.write(row['Date'].strftime('%d/%m/%Y') if pd.notna(row['Date']) else 'No date')

                        st.write("**Status**")
                        st.write(row['State'])
                        st.write("**Priority**")
                        st.write(row['Priority  Level'])
                        if pd.notna(row['Idea']):
                            st.write("**Description**")
                            st.write(row['Idea'])

    with tab2:
        if not filtered_data.empty:
            col1, col2 = st.columns(2)

            with col1:
                # Team distribution
                team_dist = filtered_data['Team'].value_counts()
                st.subheader("Ideas by Team")
                if plotly_available:
                    fig = px.pie(values=team_dist.values, names=team_dist.index,
                                 title='Distribution by Team')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(team_dist)

            with col2:
                # Status distribution
                status_dist = filtered_data['State'].value_counts()
                st.subheader("Ideas by Status")
                if plotly_available:
                    fig = px.bar(x=status_dist.index, y=status_dist.values,
                                 title='Distribution by Status')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(status_dist)

            # Priority and Team Cross Analysis
            st.subheader("Team Performance Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Priority distribution by team
                priority_team = pd.crosstab(filtered_data['Team'], filtered_data['Priority  Level'])
                st.write("Priority Distribution by Team")
                if plotly_available:
                    fig = px.bar(priority_team,
                                 title='Priority Levels by Team',
                                 barmode='stack')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(priority_team)

            with col2:
                # Completion rate by team
                team_completion = (filtered_data[filtered_data['State'] == 'Completed']
                                   .groupby('Team').size() /
                                   filtered_data.groupby('Team').size() * 100).round(1)
                st.write("Completion Rate by Team (%)")
                if plotly_available:
                    fig = px.bar(x=team_completion.index, y=team_completion.values,
                                 title='Team Completion Rates')
                    fig.update_traces(marker_color='lightgreen')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(team_completion)

    with tab3:
        if not filtered_data.empty:
            # Trends over time
            st.subheader("Submission Trends")
            col1, col2 = st.columns(2)

            with col1:
                # Monthly submission trends
                monthly_trends = filtered_data.groupby('MonthYear').size().reset_index()
                monthly_trends.columns = ['Month', 'Count']

                st.write("Monthly Submission Trends")
                if plotly_available:
                    fig = px.line(monthly_trends, x='Month', y='Count',
                                  title='Ideas Submitted Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(monthly_trends.set_index('Month'))

            with col2:
                # Team trends over time
                team_monthly = filtered_data.groupby(['MonthYear', 'Team']).size().reset_index()
                team_monthly.columns = ['Month', 'Team', 'Count']

                st.write("Team Submission Trends")
                if plotly_available:
                    fig = px.line(team_monthly, x='Month', y='Count', color='Team',
                                  title='Team Submissions Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    pivot_data = team_monthly.pivot(index='Month', columns='Team', values='Count').fillna(0)
                    st.line_chart(pivot_data)

else:
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

        if not filtered_data.empty:
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
                chart_data = prepare_line_chart_data(
                    filtered_data,
                    'Aggregation Date',
                    'App Launch Count',
                    'Device Platform'
                )
                if not chart_data.empty:
                    st.line_chart(chart_data)
        else:
            st.warning("No data available for the selected date range.")

    elif dataset_choice == "Daily App Launches":
        st.header("📈 Daily App Launch Trends")

        filtered_data = data5[
            (data5['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data5['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        if not filtered_data.empty:
            # KPI metrics
            col1, col2 = st.columns(2)
            col1.metric("Total Launches", f"{filtered_data['App Launch Count'].sum():,}")
            col2.metric("Daily Average", f"{filtered_data['App Launch Count'].mean():,.0f}")

            # Daily trend
            if plotly_available:
                fig = px.line(filtered_data, x='Aggregation Date', y='App Launch Count',
                              title='Daily App Launch Trend')
                fig.add_trace(go.Scatter(
                    x=filtered_data['Aggregation Date'],
                    y=filtered_data['App Launch Count'].rolling(7).mean(),
                    name='7-day Moving Average',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(filtered_data.set_index('Aggregation Date')['App Launch Count'])

            # Show weekly patterns
            st.subheader("Weekly Launch Patterns")
            filtered_data['Day'] = filtered_data['Aggregation Date'].dt.day_name()
            daily_patterns = filtered_data.groupby('Day')['App Launch Count'].agg(['mean', 'max', 'min']).round(0)
            daily_patterns.index = pd.Categorical(daily_patterns.index, categories=[
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            daily_patterns = daily_patterns.sort_index()

            if plotly_available:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=daily_patterns.index, y=daily_patterns['mean'],
                                     name='Average Launches'))
                fig.update_layout(title='Average Daily Launch Patterns')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(daily_patterns['mean'])

        else:
            st.warning("No data available for the selected date range.")

    elif dataset_choice == "Active Users":
        st.header("👥 Active Users Analysis")

        filtered_data = data6[
            (data6['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data6['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        if not filtered_data.empty:
            # KPI metrics
            col1, col2 = st.columns(2)
            col1.metric("Total Active Users", f"{filtered_data['Active Users'].sum():,}")
            col2.metric("Daily Average", f"{filtered_data['Active Users'].mean():,.0f}")

            # Active users trend
            if plotly_available:
                fig = px.area(filtered_data, x='Aggregation Date', y='Active Users',
                              title='Active Users Over Time')

                # Add trend line
                ma7 = filtered_data['Active Users'].rolling(window=7).mean()
                fig.add_scatter(x=filtered_data['Aggregation Date'], y=ma7,
                                name='7-day Moving Average',
                                line=dict(color='red', dash='dash'))

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.area_chart(filtered_data.set_index('Aggregation Date')['Active Users'])

            # Monthly analysis
            st.subheader("Monthly Active Users Analysis")
            filtered_data['Month'] = filtered_data['Aggregation Date'].dt.to_period('M')
            monthly_users = filtered_data.groupby('Month')['Active Users'].agg([
                ('Average Users', 'mean'),
                ('Peak Users', 'max'),
                ('Min Users', 'min')
            ]).round(0)

            if plotly_available:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=monthly_users.index.astype(str),
                                     y=monthly_users['Average Users'],
                                     name='Average Users'))
                fig.add_trace(go.Scatter(x=monthly_users.index.astype(str),
                                         y=monthly_users['Peak Users'],
                                         name='Peak Users',
                                         mode='lines+markers'))
                fig.update_layout(title='Monthly User Metrics')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(monthly_users)

        else:
            st.warning("No data available for the selected date range.")

    else:  # Player Versions
        st.header("🎮 Player Version Distribution")

        if not data7.empty:
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

                # Pie chart view
                fig2 = px.pie(data7, values='Active Users', names='Player Version',
                              title='Version Distribution')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.bar_chart(data7.set_index('Player Version')['Active Users'])
        else:
            st.warning("No player version data available.")

# Add data table with filters
st.sidebar.title("📊 Additional Options")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("🔍 Raw Data View")
    try:
        if main_page == "App Utilization Analytics":
            if dataset_choice == "App Launches by Platform":
                if not filtered_data.empty:
                    st.dataframe(filtered_data)
                else:
                    st.warning("No data available to display.")
            elif dataset_choice == "Daily App Launches":
                if not filtered_data.empty:
                    st.dataframe(filtered_data)
                else:
                    st.warning("No data available to display.")
            elif dataset_choice == "Active Users":
                if not filtered_data.empty:
                    st.dataframe(filtered_data)
                else:
                    st.warning("No data available to display.")
            else:
                if not data7.empty:
                    st.dataframe(data7)
                else:
                    st.warning("No player version data available.")
        else:  # Ideas Management Dashboard
            if not filtered_data.empty:
                available_columns = filtered_data.columns.tolist()
                default_columns = ['Title', 'Team', 'State', 'Priority  Level', 'Lead', 'Date']
                # Only include default columns that actually exist in the dataset
                default_columns = [col for col in default_columns if col in available_columns]

                columns_to_display = st.multiselect(
                    "Select columns to display",
                    available_columns,
                    default=default_columns
                )
                if columns_to_display:
                    st.dataframe(filtered_data[columns_to_display])
                else:
                    st.warning("Please select at least one column to display.")
            else:
                st.warning("No data available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Analytics Dashboard** - v1.0")