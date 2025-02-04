import os
import pandas as pd
from datetime import datetime, timedelta
import calendar
import streamlit as st
import numpy as np

def safe_division(numerator, denominator):
    """Safely perform division handling zero denominator case"""
    try:
        if denominator == 0:
            return 0
        return (numerator / denominator) * 100
    except:
        return 0
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

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    .search-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTextInput > div > div > input {
        background-color: white;
        padding-left: 2rem;
        font-size: 1rem;
    }
    .search-icon {
        position: absolute;
        left: 0.5rem;
        top: 50%;
        transform: translateY(-50%);
        color: #6c757d;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stTab {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .idea-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-completed {
        background-color: #d4edda;
        color: #155724;
    }
    .status-progress {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-new {
        background-color: #cce5ff;
        color: #004085;
    }
    </style>
""", unsafe_allow_html=True)


def clean_submitter_name(submitter):
    """Cleans up the submitter name"""
    if pd.isna(submitter):
        return 'Unknown'
    elif ';#' in str(submitter):
        return submitter.split(';#')[0]
    return submitter


def calculate_team_metrics(df):
    """Calculate comprehensive team performance metrics"""
    metrics = {}

    # Safely calculate completion rate
    total = len(df)
    completed = len(df[df['State'] == 'Completed'])
    metrics['completion_rate'] = round(safe_division(completed, total), 1)

    # Other metrics calculations...
    metrics['high_priority_ratio'] = round(
        safe_division(
            len(df[df['Priority  Level'] == 'High']),
            total
        ), 1
    )

    metrics['active_projects'] = len(df[df['State'].isin(['In Progress', 'New'])])

    # Calculate average completion time only if there are completed projects
    completed_projects = df[df['State'] == 'Completed'].copy()
    if not completed_projects.empty:
        completed_projects['time_to_complete'] = (
                pd.to_datetime(completed_projects['Closed Date']) -
                pd.to_datetime(completed_projects['Date'])
        ).dt.days
        metrics['avg_completion_time'] = completed_projects['time_to_complete'].mean()
    else:
        metrics['avg_completion_time'] = 0

    return metrics


def get_team_participation(df):
    """Analyze team participation and identify non-participants"""
    team_stats = {}

    for team in df['Team'].unique():
        team_df = df[df['Team'] == team]
        team_stats[team] = {
            'total_members': team_df['SubmittedBy'].nunique(),
            'active_members': team_df[team_df['State'] != 'Completed']['SubmittedBy'].nunique(),
            'total_submissions': team_df.shape[0],
            'submissions_per_member': team_df.shape[0] / team_df['SubmittedBy'].nunique()
            if team_df['SubmittedBy'].nunique() > 0 else 0
        }

    return team_stats


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

        # Convert SSC date columns with explicit format
        date_columns = ['Date', 'Closed Date']
        for col in date_columns:
            dfs['ssc_data'][col] = pd.to_datetime(
                dfs['ssc_data'][col],
                format='%d/%m/%Y',
                errors='coerce',
                dayfirst=True  # Explicitly specify day-first format
            )

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


def create_search_interface():
    """Create an enhanced search interface"""
    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    # Advanced search interface with autocomplete
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Get unique submitters for autocomplete
        all_submitters = sorted(ssc_data['SubmittedBy'].unique())
        search_name = st.selectbox(
            "🔍 Search by Name",
            [""] + all_submitters,
            help="Search by submitter name"
        )

    with col2:
        all_teams = ['All Teams'] + sorted(ssc_data['Team'].dropna().unique().tolist())
        selected_team = st.selectbox(
            '👥 Filter by Team',
            all_teams,
            help="Select a specific team or view all"
        )

    with col3:
        show_inactive = st.checkbox(
            "Show Non-participants",
            help="Show team members with no submissions"
        )

    # Date range filter with improved UX
    col1, col2 = st.columns(2)
    with col1:
        min_date = ssc_data['Date'].min().date()
        max_date = ssc_data['Date'].max().date()
        start_date = st.date_input(
            "Start Date",
            value=None,
            min_value=min_date,
            max_value=max_date,
            help="Filter from this date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=None,
            min_value=min_date,
            max_value=max_date,
            help="Filter until this date"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    return search_name, selected_team, show_inactive, start_date, end_date


def create_project_management_metrics(filtered_data):
    """Create comprehensive project management metrics display"""
    st.markdown("### 📊 Project Management Overview")

    if filtered_data.empty:
        # Handle empty dataset
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Ideas", "0")
        col2.metric("Completion Rate", "0%")
        col3.metric("High Priority", "0")
        col4.metric("Active Teams", "0")
        return

    # Calculate metrics
    total_ideas = len(filtered_data)
    completed = len(filtered_data[filtered_data['State'] == 'Completed'])
    completion_rate = round(safe_division(completed, total_ideas), 1)
    high_priority = len(filtered_data[filtered_data['Priority  Level'] == 'High'])
    active_teams = filtered_data['Team'].nunique()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Ideas", total_ideas)
    with col2:
        st.metric("Completion Rate", f"{completion_rate}%")
    with col3:
        st.metric("High Priority", high_priority)
    with col4:
        st.metric("Active Teams", active_teams)


def create_team_analysis(filtered_data):
    """Create enhanced team analysis visualizations"""
    st.markdown("### 👥 Team Performance Analysis")

    # Get team participation stats
    team_stats = get_team_participation(filtered_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if plotly_available:
            # Create submission performance chart
            performance_data = pd.DataFrame(team_stats).T
            fig = px.bar(
                performance_data,
                y=['total_submissions', 'active_members'],
                title='Team Performance Overview',
                barmode='group',
                labels={
                    'index': 'Team',
                    'value': 'Count',
                    'variable': 'Metric'
                },
                color_discrete_sequence=['#4CAF50', '#2196F3']
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            performance_data = pd.DataFrame(team_stats).T[['total_submissions']]
            st.bar_chart(performance_data)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if plotly_available:
            # Create efficiency chart
            efficiency_data = pd.DataFrame(team_stats).T[['submissions_per_member']]
            fig = px.bar(
                efficiency_data,
                title='Team Efficiency (Submissions per Member)',
                labels={
                    'index': 'Team',
                    'value': 'Submissions per Member'
                },
                color_discrete_sequence=['#2ecc71']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame(team_stats).T[['submissions_per_member']])
        st.markdown('</div>', unsafe_allow_html=True)


def create_timeline_analysis(filtered_data):
    """Create timeline analysis visualizations"""
    st.markdown("### 📅 Timeline Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Monthly submission trends
        monthly_trends = filtered_data.groupby('MonthYear').size().reset_index()
        monthly_trends.columns = ['Month', 'Count']

        if plotly_available:
            fig = px.line(
                monthly_trends,
                x='Month',
                y='Count',
                title='Monthly Submission Trends',
                markers=True
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Submissions"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(monthly_trends.set_index('Month'))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Status distribution over time
        status_timeline = pd.crosstab(
            filtered_data['MonthYear'],
            filtered_data['State']
        )

        if plotly_available:
            fig = px.area(
                status_timeline,
                title='Status Distribution Over Time',
                labels={
                    'MonthYear': 'Month',
                    'value': 'Number of Ideas',
                    'State': 'Status'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.area_chart(status_timeline)
        st.markdown('</div>', unsafe_allow_html=True)


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
    ["Ideas Search & Analytics", "Project Management", "App Utilization Analytics"]
)

if main_page in ["Ideas Search & Analytics", "Project Management"]:
    # Enhanced header with metrics
    st.title("💡 Ideas & Project Management Hub")

    # Create search interface
    search_name, selected_team, show_inactive, start_date, end_date = create_search_interface()

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

    if main_page == "Ideas Search & Analytics":
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Search Results",
            "📊 Team Analysis",
            "📈 Trends",
            "👥 Member Insights"
        ])

        with tab1:
            if filtered_data.empty:
                st.warning("No ideas found matching your search criteria.")
            else:
                st.write(f"Found {len(filtered_data)} ideas matching your criteria")

                # Sorting options
                sort_col, sort_order = st.columns([2, 1])
                with sort_col:
                    sort_by = st.selectbox(
                        "Sort by",
                        ["Date", "Priority  Level", "State", "Team"],
                        index=0
                    )
                with sort_order:
                    ascending = st.checkbox("Ascending", True)

                # Sort the data
                filtered_data = filtered_data.sort_values(by=sort_by, ascending=ascending)

                # Display results
                for _, row in filtered_data.iterrows():
                    with st.expander(f"📌 {row['Title']}"):
                        # Status badge
                        status_class = {
                            'Completed': 'status-completed',
                            'In Progress': 'status-progress',
                            'New': 'status-new'
                        }.get(row['State'], '')

                        st.markdown(
                            f'<span class="status-badge {status_class}">{row["State"]}</span>',
                            unsafe_allow_html=True
                        )

                        col1, col2, col3 = st.columns([2, 2, 1])

                        with col1:
                            st.markdown(f"**Submitter:** {row['SubmittedBy']}")
                            st.markdown(f"**Team:** {row['Team']}")

                        with col2:
                            st.markdown(f"**Priority:** {row['Priority  Level']}")
                            if pd.notna(row['Lead']):
                                st.markdown(f"**Lead:** {row['Lead']}")

                        with col3:
                            st.markdown(f"**Date:** {row['Date'].strftime('%d/%m/%Y')}")
                            if pd.notna(row['Closed Date']):
                                st.markdown(
                                    f"**Completed:** {row['Closed Date'].strftime('%d/%m/%Y')}"
                                )

                        if pd.notna(row['Idea']):
                            st.markdown("**Description:**")
                            st.markdown(f">{row['Idea']}")

        with tab2:
            if not filtered_data.empty:
                create_team_analysis(filtered_data)
            else:
                st.warning("No data available for team analysis.")

        with tab3:
            if not filtered_data.empty:
                create_timeline_analysis(filtered_data)
            else:
                st.warning("No data available for trend analysis.")

        with tab4:
            if selected_team != 'All Teams':
                st.markdown("### 👥 Team Member Analysis")

                # Get team member statistics
                member_data = ssc_data[ssc_data['Team'] == selected_team]
                member_stats = []

                for member in member_data['SubmittedBy'].unique():
                    ideas = member_data[member_data['SubmittedBy'] == member]
                    stats = {
                        'Name': member,
                        'Total Ideas': len(ideas),
                        'Completed': len(ideas[ideas['State'] == 'Completed']),
                        'In Progress': len(ideas[ideas['State'].isin(['In Progress', 'New'])]),
                        'High Priority': len(ideas[ideas['Priority  Level'] == 'High']),
                        'Last Submission': ideas['Date'].max().strftime('%d/%m/%Y') if len(ideas) > 0 else 'Never'
                    }
                    member_stats.append(stats)

                if member_stats:
                    member_df = pd.DataFrame(member_stats)

                    # Create member insights
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        if plotly_available:
                            fig = px.bar(
                                member_df,
                                x='Name',
                                y=['Completed', 'In Progress', 'High Priority'],
                                title='Member Performance Overview',
                                barmode='group'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.bar_chart(member_df.set_index('Name')[['Total Ideas', 'Completed']])
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("### Member Statistics")
                        styled_df = member_df.style.background_gradient(
                            subset=['Total Ideas', 'Completed'],
                            cmap='YlOrRd'
                        )
                        st.dataframe(styled_df, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    if show_inactive:
                        st.markdown("### Team Members with No Recent Activity")
                        inactive_threshold = pd.Timestamp.now() - pd.Timedelta(days=30)
                        inactive_members = member_df[
                            pd.to_datetime(member_df['Last Submission']) < inactive_threshold
                            ]
                        if not inactive_members.empty:
                            for _, member in inactive_members.iterrows():
                                st.warning(
                                    f"🚫 {member['Name']} - Last submission: {member['Last Submission']}"
                                )
                        else:
                            st.success("All team members have been active in the last 30 days!")
                else:
                    st.warning("No member data available for the selected team.")
            else:
                st.info("Please select a specific team to view member insights.")

    else:  # Project Management view
        # Create project management metrics
        create_project_management_metrics(filtered_data)

        # Create visualization tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Performance Metrics",
            "⏱️ Timeline Analysis",
            "🎯 Priority Analysis"
        ])

        with tab1:
            if not filtered_data.empty:
                # Team completion rates
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                team_completion = filtered_data.groupby('Team').apply(
                    lambda x: len(x[x['State'] == 'Completed']) / len(x) * 100
                ).round(1)

                if plotly_available:
                    fig = px.bar(
                        x=team_completion.index,
                        y=team_completion.values,
                        title='Team Completion Rates (%)',
                        labels={'x': 'Team', 'y': 'Completion Rate (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(team_completion)
                st.markdown('</div>', unsafe_allow_html=True)

                # Priority distribution
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                priority_dist = pd.crosstab(
                    filtered_data['Team'],
                    filtered_data['Priority  Level'],
                    normalize='index'
                ) * 100

                if plotly_available:
                    fig = px.imshow(
                        priority_dist,
                        title='Priority Distribution by Team (%)',
                        labels={'x': 'Priority Level', 'y': 'Team', 'color': 'Percentage'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(priority_dist)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            if not filtered_data.empty:
                create_timeline_analysis(filtered_data)
            else:
                st.warning("No data available for timeline analysis.")

        with tab3:
            if not filtered_data.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    if plotly_available:
                        fig = px.pie(
                            filtered_data,
                            names='Priority  Level',
                            title='Priority Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(filtered_data['Priority  Level'].value_counts())
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    state_priority = pd.crosstab(
                        filtered_data['State'],
                        filtered_data['Priority  Level']
                    )
                    if plotly_available:
                        fig = px.bar(
                            state_priority,
                            title='Status Distribution by Priority',
                            barmode='stack'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(state_priority)
                    st.markdown('</div>', unsafe_allow_html=True)

else:  # App Utilization Analytics
    st.title('📱 App Utilization Analytics')

    # Dataset selection
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

    # Display appropriate dataset visualization
    if dataset_choice == "App Launches by Platform":
        st.header("📊 App Launch Analysis by Platform")

        filtered_data = data4[
            (data4['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data4['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        if not filtered_data.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Launches", f"{filtered_data['App Launch Count'].sum():,}")
            col2.metric("Platforms", filtered_data['Device Platform'].nunique())
            col3.metric("Avg Daily Launches", f"{filtered_data['App Launch Count'].mean():,.0f}")

            if plotly_available:
                fig = px.line(
                    filtered_data,
                    x='Aggregation Date',
                    y='App Launch Count',
                    color='Device Platform',
                    title='App Launches by Platform Over Time'
                )
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

    elif dataset_choice == "Daily App Launches":
        st.header("📈 Daily App Launch Trends")

        filtered_data = data5[
            (data5['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data5['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        if not filtered_data.empty:
            col1, col2 = st.columns(2)
            col1.metric("Total Launches", f"{filtered_data['App Launch Count'].sum():,}")
            col2.metric("Daily Average", f"{filtered_data['App Launch Count'].mean():,.0f}")

            if plotly_available:
                fig = px.line(
                    filtered_data,
                    x='Aggregation Date',
                    y='App Launch Count',
                    title='Daily App Launch Trend'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(filtered_data.set_index('Aggregation Date')['App Launch Count'])

    elif dataset_choice == "Active Users":
        st.header("👥 Active Users Analysis")

        filtered_data = data6[
            (data6['Aggregation Date'] >= pd.to_datetime(date_range[0])) &
            (data6['Aggregation Date'] <= pd.to_datetime(date_range[1]))
            ]

        if not filtered_data.empty:
            col1, col2 = st.columns(2)
            col1.metric("Total Active Users", f"{filtered_data['Active Users'].sum():,}")
            col2.metric("Daily Average", f"{filtered_data['Active Users'].mean():,.0f}")

        if plotly_available:
            fig = px.area(
                filtered_data,
                x='Aggregation Date',
                y='Active Users',
                title='Active Users Over Time'
            )
            fig.add_scatter(
                x=filtered_data['Aggregation Date'],
                y=filtered_data['Active Users'].rolling(7).mean(),
                name='7-day Moving Average',
                line=dict(color='red', dash='dash')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.area_chart(filtered_data.set_index('Aggregation Date')['Active Users'])

            # Weekly patterns analysis
        st.subheader("Weekly Activity Patterns")
        filtered_data['Day'] = filtered_data['Aggregation Date'].dt.day_name()
        daily_patterns = filtered_data.groupby('Day')['Active Users'].agg([
            ('Average', 'mean'),
            ('Peak', 'max'),
            ('Minimum', 'min')
        ]).round(0)

        # Ensure correct day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_patterns = daily_patterns.reindex(day_order)

        if plotly_available:
            fig = px.bar(
                daily_patterns,
                y='Average',
                title='Average Daily Active Users',
                labels={'index': 'Day', 'Average': 'Active Users'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(daily_patterns['Average'])

    else:  # Player Versions
        st.header("🎮 Player Version Distribution")

        if not data7.empty:
            col1, col2 = st.columns(2)
            col1.metric("Total Users", f"{data7['Active Users'].sum():,}")
            col2.metric("Versions", data7['Player Version'].nunique())

            # Version distribution
            if plotly_available:
                fig = px.bar(
                    data7,
                    x='Player Version',
                    y='Active Users',
                    title='Users by Player Version'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                # Pie chart view
                fig2 = px.pie(
                    data7,
                    values='Active Users',
                    names='Player Version',
                    title='Version Distribution'
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.bar_chart(data7.set_index('Player Version')['Active Users'])

            # Version adoption metrics
            st.subheader("Version Adoption Metrics")
            total_users = data7['Active Users'].sum()
            version_metrics = data7.copy()
            version_metrics['Percentage'] = (version_metrics['Active Users'] / total_users * 100).round(1)
            version_metrics = version_metrics.sort_values('Active Users', ascending=False)

            st.dataframe(
                version_metrics[['Player Version', 'Active Users', 'Percentage']],
                use_container_width=True
            )
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
                    st.dataframe(
                        filtered_data[columns_to_display].style.background_gradient(
                            subset=['Priority  Level']
                        ),
                        use_container_width=True
                    )
                else:
                    st.warning("Please select at least one column to display.")
            else:
                st.warning("No data available for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Analytics Dashboard** - v2.0")
st.sidebar.markdown("Built with ❤️ by Sudaraka")

# Add help tooltip
st.sidebar.info(
    """
    💡 **Tips:**
    - Use the search bar to find specific submitters
    - Filter by team to see team-specific analytics
    - Toggle 'Show Raw Data' to view detailed information
    - Use date filters to analyze trends over time
    """
)