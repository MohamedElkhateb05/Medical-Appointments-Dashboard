import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Data Preprocessing and Feature Engineering (All previous steps combined) ---
# Load the dataset from your public URL
try:
    # This is the direct download URL for the file from your Google Drive link.
    # The 'view' link was converted to a 'download' link for direct access.
    data_url = 'KaggleV2-May-2016.csv'
    df = pd.read_csv(data_url)

    # Drop the ID columns as they are not needed for analysis
    df = df.drop(['PatientId', 'AppointmentID'], axis=1)

    # Standardize all column names to be lowercase with underscores for consistency
    df.columns = [
        'gender', 'scheduled_day', 'appointment_day', 'age', 'neighborhood',
        'scholarship', 'hipertension', 'diabetes', 'alcoholism',
        'handicap', 'sms_received', 'no_show'
    ]

    # Convert 'No-show' column to a binary numerical format: 'No' -> 1 (showed up), 'Yes' -> 0 (no-show)
    df['no_show'] = df['no_show'].map({'No': 1, 'Yes': 0})

    # Convert 'Gender' column to a binary numerical format: 'F' -> 0, 'M' -> 1
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})

    # Convert date columns to datetime objects and remove the timezone info
    df['scheduled_day'] = pd.to_datetime(
        df['scheduled_day']).dt.tz_localize(None)
    df['appointment_day'] = pd.to_datetime(
        df['appointment_day']).dt.tz_localize(None)

    # Create a new feature for 'Day of Week' from the appointment date
    df['day_of_week'] = df['appointment_day'].dt.day_name()

    # Filter out rows where the 'age' is negative
    df = df[df['age'] >= 0]

    # Calculate the wait time in days between scheduling and appointment
    df['wait_time_days'] = (df['appointment_day'] -
                            df['scheduled_day']).dt.days

    # Filter out rows where the 'wait_time_days' is negative (data entry errors)
    df = df[df['wait_time_days'] >= 0]

    # --- Added to create a map visualization ---
    # Create a simple mapping of neighborhoods to approximate, fixed lat/lon points
    # This is a proxy since the dataset doesn't contain geographical coordinates.
    # The coordinates are centered around Vitoria, Brazil, where the data is from.
    neighborhood_coords = {
        'JARDIM CAMBURI': (-20.2763, -40.2970),
        'MARIA ORTIZ': (-20.2605, -40.2941),
        'JARDIM DA PENHA': (-20.2949, -40.2878),
        'ITARARÉ': (-20.2906, -40.3204),
        'RESISTÊNCIA': (-20.2947, -40.3341),
        'CENTRO': (-20.3188, -40.3370),
        'outros': (-20.2988, -40.3209)
    }

    # Assign coordinates to each appointment based on neighborhood
    df['lat'] = df['neighborhood'].map(lambda x: neighborhood_coords.get(
        x, neighborhood_coords['outros'])[0] + np.random.uniform(-0.01, 0.01))
    df['lon'] = df['neighborhood'].map(lambda x: neighborhood_coords.get(
        x, neighborhood_coords['outros'])[1] + np.random.uniform(-0.01, 0.01))

except Exception as e:
    print(f"Error loading data: {e}")
    # Create an empty DataFrame to prevent the app from crashing
    df = pd.DataFrame(columns=[
        'gender', 'scheduled_day', 'appointment_day', 'age', 'neighborhood',
        'scholarship', 'hipertension', 'diabetes', 'alcoholism',
        'handicap', 'sms_received', 'no_show', 'day_of_week', 'wait_time_days', 'lat', 'lon'
    ])

# --- Dashboard Initialization and Layout ---
# Use the CYBORG theme for a dark, cyber look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    # Banner with a larger, more prominent title
    dbc.Row(
        [
            dbc.Col(
                html.H2(
                    "Medical Appointment Dashboard",
                    className="text-primary text-center my-4",
                    style={'font-size': '2.5rem'}
                ),
                width=12
            )
        ],
        className="mb-4"
    ),

    # Main container with a dark grey background
    dbc.Container([
        dbc.Row([
            # Left column for filters and key metrics
            dbc.Col([
                # Filters and key metrics in a single, compact card
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Analysis Tools", className="text-primary"),
                        html.P("Filter the data and view key metrics.",
                               className="text-secondary"),

                        # Key Metrics Section
                        dbc.Row([
                            dbc.Col(html.Div([
                                html.H6("Total", className="text-secondary"),
                                html.H4(id='total-appointments',
                                        className="text-white")
                            ], className="text-center p-2 rounded mb-2"), width=6),
                            dbc.Col(html.Div([
                                html.H6("Show-Up Rate",
                                        className="text-secondary"),
                                html.H4(id='show-up-rate',
                                        className="text-white")
                            ], className="text-center p-2 rounded mb-2"), width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.Div([
                                html.H6("Avg Wait", className="text-secondary"),
                                html.H4(id='avg-waiting-days',
                                        className="text-white")
                            ], className="text-center p-2 rounded mb-2"), width=6),
                            dbc.Col(html.Div([
                                html.H6("SMS Rate", className="text-secondary"),
                                html.H4(id='sms-rate', className="text-white")
                            ], className="text-center p-2 rounded mb-2"), width=6)
                        ]),
                        html.Hr(className="my-3"),

                        # Filters Section
                        html.Label("Age Range", className="text-secondary"),
                        dcc.RangeSlider(
                            id='age-slider',
                            min=0,
                            max=100,
                            step=5,
                            value=[0, 100],
                            marks={i: {'label': str(i), 'style': {
                                'color': '#9C9C9C'}} for i in range(0, 101, 20)},
                            className="mb-3"
                        ),
                        html.Label("Neighborhood",
                                   className="text-secondary mt-4"),
                        dcc.Dropdown(
                            id='neighborhood-dropdown',
                            options=[{'label': n, 'value': n}
                                     for n in sorted(df['neighborhood'].unique())],
                            value=None,
                            placeholder="Select a neighborhood",
                            multi=True,
                            className="dbc-dropdown"
                        ),
                        html.Label("Medical Conditions",
                                   className="text-secondary mt-4"),
                        dcc.Checklist(
                            id='condition-checklist',
                            options=[
                                {'label': 'Diabetes', 'value': 'diabetes'},
                                {'label': 'Hypertension', 'value': 'hipertension'},
                                {'label': 'Alcoholism', 'value': 'alcoholism'},
                                {'label': 'Handicap', 'value': 'handicap'}
                            ],
                            value=[],
                            className="text-white"
                        )
                    ]),
                    color="dark",
                    outline=True
                )
            ], width=3, className="p-4"),

            # Right column for visualizations
            dbc.Col([
                html.H4("Visualizations",
                        className="text-primary text-center mb-4"),
                dbc.Row([
                    # Combined graph for show-up rate by gender and day of week
                    dbc.Col(dcc.Graph(id='gender-day-bar',
                            config={'responsive': True}), width=6),
                    # Combined scatter plot for age vs wait time, colored by attendance
                    dbc.Col(dcc.Graph(id='age-wait-scatter',
                            config={'responsive': True}), width=6),
                ], className="mb-4"),
                dbc.Row([
                    # Pie chart for show-up distribution
                    dbc.Col(dcc.Graph(id='show-no-show-pie',
                            config={'responsive': True}), width=6),
                    # Map visualization
                    dbc.Col(dcc.Graph(id='neighborhood-map',
                            config={'responsive': True}), width=6),
                ], className="mb-4"),
                dbc.Row([
                    # Bar chart for show-up rate by medical condition
                    dbc.Col(dcc.Graph(id='condition-bar',
                            config={'responsive': True}), width=6),
                    # Box plot for waiting days by medical condition
                    dbc.Col(dcc.Graph(id='condition-wait-boxplot',
                            config={'responsive': True}), width=6),
                ])
            ], width=9, className="p-4")
        ])
    ], fluid=True, style={'backgroundColor': '#212529'})  # A dark grey color
], fluid=True, className="p-0")


# --- Callbacks for Interactivity ---
@app.callback(
    [
        Output('total-appointments', 'children'),
        Output('show-up-rate', 'children'),
        Output('avg-waiting-days', 'children'),
        Output('sms-rate', 'children'),
        Output('gender-day-bar', 'figure'),
        Output('age-wait-scatter', 'figure'),
        Output('show-no-show-pie', 'figure'),
        Output('neighborhood-map', 'figure'),
        Output('condition-bar', 'figure'),
        Output('condition-wait-boxplot', 'figure')
    ],
    [
        Input('age-slider', 'value'),
        Input('neighborhood-dropdown', 'value'),
        Input('condition-checklist', 'value')
    ]
)
def update_dashboard(age_range, neighborhoods, conditions):
    # Filter data based on age range
    filtered_df = df[
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1])
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Handle neighborhood filter
    if neighborhoods:
        filtered_df = filtered_df[filtered_df['neighborhood'].isin(
            neighborhoods)]

    # Handle medical conditions filter
    if conditions:
        for condition in conditions:
            filtered_df = filtered_df[filtered_df[condition] == 1]

    # Check if filtered data is empty
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(
            annotations=[
                dict(text="No data available for selected filters", showarrow=False)],
            template='plotly_dark'  # Use dark template for empty graph
        )
        return (
            0,
            "0%",
            "0",
            "0%",
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig
        )

    # Calculate metrics
    total_appointments = len(filtered_df)
    show_up_rate = f"{filtered_df['no_show'].mean()*100:.1f}%"
    avg_waiting_days = f"{filtered_df['wait_time_days'].mean():.1f}"
    sms_rate = f"{filtered_df['sms_received'].mean()*100:.1f}%"

    # Define a cyber-themed color palette
    cyber_palette = {
        'male': '#00bfff',  # Deep Sky Blue
        'female': '#ff69b4',  # Hot Pink
        'show': '#00ff7f',  # Spring Green
        'no_show': '#ff4500',  # Orange Red
        'default': '#ffc300'  # Golden Yellow
    }

    # 1. Combined bar chart for show-up rate by gender and day of week
    day_gender_data = filtered_df.groupby(['day_of_week', 'gender'])[
        'no_show'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_gender_data['day_of_week'] = pd.Categorical(
        day_gender_data['day_of_week'], categories=day_order, ordered=True)
    day_gender_data = day_gender_data.sort_values('day_of_week')

    gender_day_bar = px.bar(
        day_gender_data,
        x='day_of_week',
        y='no_show',
        color='gender',
        barmode='group',
        title='Show-Up Rate by Gender and Day of Week',
        labels={'no_show': 'Show-Up Rate',
                'day_of_week': 'Day of Week', 'gender': 'Gender'},
        color_discrete_map={
            0: cyber_palette['female'], 1: cyber_palette['male']},
        template='plotly_dark'
    )
    gender_day_bar.update_yaxes(tickformat='.0%')
    gender_day_bar.update_layout(
        showlegend=True, legend_title_text='Gender (F: 0, M: 1)')

    # 2. Combined scatter for age vs wait time, colored by attendance
    age_wait_scatter = px.scatter(
        filtered_df,
        x='age',
        y='wait_time_days',
        color='no_show',
        title='Wait Time and Age vs Attendance',
        labels={
            'age': 'Age', 'wait_time_days': 'Wait Time (Days)', 'no_show': 'Attendance (1: Show, 0: No-Show)'},
        color_discrete_map={
            0: cyber_palette['no_show'], 1: cyber_palette['show']},
        template='plotly_dark'
    )

    # 3. Pie chart for show/no-show distribution
    show_data = filtered_df['no_show'].value_counts().reset_index()
    show_data.columns = ['no_show', 'count']
    pie_fig = px.pie(
        show_data,
        names='no_show',
        values='count',
        title='Show-Up Distribution',
        labels={'no_show': 'Attendance'},
        color='no_show',
        color_discrete_map={
            0: cyber_palette['no_show'], 1: cyber_palette['show']},
        template='plotly_dark'
    )
    pie_fig.update_traces(hole=.4, textinfo='percent+label',
                          marker=dict(line=dict(color='#000000', width=2)))
    pie_fig.update_layout(showlegend=False)

    # 4. Map visualization
    neighborhood_map = px.scatter_mapbox(
        filtered_df,
        lat="lat",
        lon="lon",
        color="no_show",
        hover_name="neighborhood",
        hover_data={"wait_time_days": True, "age": True, "no_show": True},
        color_discrete_map={
            0: cyber_palette['no_show'], 1: cyber_palette['show']},
        zoom=12,
        center={"lat": -20.2988, "lon": -40.3209},
        mapbox_style="carto-darkmatter",
        title='Appointments by Neighborhood'
    )
    neighborhood_map.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        mapbox_style="carto-darkmatter",
        mapbox_accesstoken=None,
    )

    # 5. Bar chart for Show-Up Rate by Medical Condition
    condition_cols = ['hipertension', 'diabetes', 'alcoholism', 'handicap']
    df_conditions = filtered_df.copy()
    df_conditions['total'] = 1  # Helper column for calculating totals
    condition_data = pd.DataFrame()
    for col in condition_cols:
        condition_data.loc[col,
                           'total'] = df_conditions[df_conditions[col] == 1]['total'].sum()
        condition_data.loc[col, 'show_up_count'] = df_conditions[
            (df_conditions[col] == 1) & (df_conditions['no_show'] == 1)
        ]['total'].sum()
    condition_data['show_up_rate'] = condition_data['show_up_count'] / \
        condition_data['total']
    condition_data.reset_index(inplace=True)
    condition_data.rename(columns={'index': 'condition'}, inplace=True)

    condition_fig = px.bar(
        condition_data,
        x='condition',
        y='show_up_rate',
        title='Show-Up Rate by Medical Condition',
        labels={'show_up_rate': 'Show-Up Rate',
                'condition': 'Medical Condition'},
        color_discrete_sequence=[cyber_palette['default']],
        template='plotly_dark'
    )
    condition_fig.update_yaxes(tickformat='.0%')

    # 6. Box plot for Waiting Days by Medical Condition
    df_long = pd.melt(
        filtered_df,
        id_vars=['wait_time_days', 'no_show'],
        value_vars=['hipertension', 'diabetes', 'alcoholism', 'handicap'],
        var_name='condition',
        value_name='has_condition'
    )
    df_long_filtered = df_long[df_long['has_condition'] == 1]
    condition_wait_boxplot = px.box(
        df_long_filtered,
        x='condition',
        y='wait_time_days',
        color='no_show',
        title='Waiting Days by Medical Condition',
        labels={
            'wait_time_days': 'Waiting Days',
            'condition': 'Medical Condition',
            'no_show': 'Attendance'
        },
        color_discrete_map={
            0: cyber_palette['no_show'], 1: cyber_palette['show']},
        template='plotly_dark'
    )

    return (
        total_appointments,
        show_up_rate,
        avg_waiting_days,
        sms_rate,
        gender_day_bar,
        age_wait_scatter,
        pie_fig,
        neighborhood_map,
        condition_fig,
        condition_wait_boxplot
    )


if __name__ == '__main__':
    app.run(debug=True)
