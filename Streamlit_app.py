
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def generate_station_map(station_df, lat_col, lon_col, station_ids_to_highlight=None):
    if station_df.empty or not lat_col or not lon_col:
        return None

    df_map_source = station_df.copy()

    # Filter to only include stations that are in station_ids_to_highlight if provided
    if station_ids_to_highlight is not None and len(station_ids_to_highlight) > 0:
        df_map_source = df_map_source[df_map_source['MonitoringLocationIdentifier'].isin(station_ids_to_highlight)]
        if df_map_source.empty:
            st.warning("No stations found matching the selected characteristic and ranges to display on map.")
            return None

    # Drop rows where latitude or longitude are missing
    df_map = df_map_source.dropna(subset=[lat_col, lon_col]).copy()

    if df_map.empty:
        st.warning("No valid latitude and longitude data found after filtering for map display.")
        return None

    # Ensure lat/lon are numeric
    df_map[lat_col] = pd.to_numeric(df_map[lat_col], errors='coerce')
    df_map[lon_col] = pd.to_numeric(df_map[lon_col], errors='coerce')
    df_map.dropna(subset=[lat_col, lon_col], inplace=True)

    if df_map.empty:
        st.warning("All latitude/longitude values became invalid after conversion for map display.")
        return None

    center_lat = df_map[lat_col].mean()
    center_lon = df_map[lon_col].mean()

    station_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    for index, row in df_map.iterrows():
        popup_text = f"Station: {row.get('MonitoringLocationName', 'N/A')}<br>ID: {row.get('MonitoringLocationIdentifier', 'N/A')}<br>Lat: {row[lat_col]:.2f}<br>Lon: {row[lon_col]:.2f}"
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup_text,
            tooltip=row.get('MonitoringLocationName', 'Station')
        ).add_to(station_map)
    return station_map

def plot_characteristic_trend(df_filtered_data, characteristic_name, value_col, date_col, site_id_col):
    if df_filtered_data.empty:
        st.warning(f"No data available to plot trend for '{characteristic_name}'.")
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(
        x=date_col,
        y=value_col,
        hue=site_id_col,
        data=df_filtered_data,
        marker='o',
        linestyle='-',
        dashes=False,
        alpha=0.8,
        markersize=6,
        ax=ax
    )
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{characteristic_name} Value')
    ax.set_title(f'Trend of {characteristic_name} over Time (Filtered Stations)')
    ax.tick_params(axis='y')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Station ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

# --- Streamlit App Layout ---
st.title('Water Quality Data Explorer')
st.markdown("Upload your water quality station data (`station.csv`) and measurement results (`narrowresult.csv`) to explore characteristics, filter by value and date ranges, and visualize station locations and trends over time.")

st.sidebar.header('Upload Data')
station_file = st.sidebar.file_uploader("Upload station.csv", type=['csv'])
narrowresult_file = st.sidebar.file_uploader("Upload narrowresult.csv", type=['csv'])

station_df = load_data(station_file)
narrowresult_df = load_data(narrowresult_file)

if station_df.empty or narrowresult_df.empty:
    st.info("Please upload both `station.csv` and `narrowresult.csv` to proceed with the analysis.")
else:
    st.success("Data loaded successfully!")

    # Define column names (adjust these if your CSVs have different headers)
    station_id_col = 'MonitoringLocationIdentifier'
    lat_col = 'LatitudeMeasure'
    lon_col = 'LongitudeMeasure'
    station_name_col = 'MonitoringLocationName'

    char_col = 'CharacteristicName'
    value_col = 'ResultMeasureValue'
    date_col = 'ActivityStartDate'
    result_station_id_col = 'MonitoringLocationIdentifier'

    # Preprocess narrowresult_df
    narrowresult_df[date_col] = pd.to_datetime(narrowresult_df[date_col], errors='coerce')
    narrowresult_df[value_col] = pd.to_numeric(narrowresult_df[value_col], errors='coerce')
    narrowresult_df.dropna(subset=[date_col, value_col, char_col, result_station_id_col], inplace=True)

    st.sidebar.header('Filter Characteristics')

    unique_characteristics = narrowresult_df[char_col].unique().tolist()
    if not unique_characteristics:
        st.warning("No unique characteristics found in `narrowresult.csv` after initial processing. Please check your data.")
        st.stop()

    selected_characteristic = st.sidebar.selectbox(
        'Select Water Quality Characteristic', unique_characteristics
    )

    if not selected_characteristic:
        st.warning("No characteristics available for selection.")
        st.stop()

    # Filter for the selected characteristic
    char_filtered_df = narrowresult_df[narrowresult_df[char_col] == selected_characteristic].copy()

    if char_filtered_df.empty:
        st.warning(f"No data for selected characteristic: '{selected_characteristic}'. Please adjust filters.")
        st.stop()

    st.sidebar.header('Define Ranges')

    # Value Range
    min_val, max_val = float(char_filtered_df[value_col].min()), float(char_filtered_df[value_col].max())
    value_range = st.sidebar.slider(
        f'{selected_characteristic} Value Range', 
        min_value=min_val, 
        max_value=max_val, 
        value=(min_val, max_val),
        step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.1, # Prevent zero division
        format='%.2f'
    )

    # Date Range
    min_date_data = char_filtered_df[date_col].min().to_pydatetime()
    max_date_data = char_filtered_df[date_col].max().to_pydatetime()

    date_start, date_end = st.sidebar.date_input(
        'Date Range',
        value=(min_date_data, max_date_data),
        min_value=min_date_data,
        max_value=max_date_data
    )

    # Convert selected dates back to pandas datetime for filtering
    date_start_pd = pd.to_datetime(date_start)
    date_end_pd = pd.to_datetime(date_end)

    # Apply all filters
    final_filtered_data = char_filtered_df[
        (char_filtered_df[value_col] >= value_range[0]) &
        (char_filtered_df[value_col] <= value_range[1]) &
        (char_filtered_df[date_col] >= date_start_pd) &
        (char_filtered_df[date_col] <= date_end_pd)
    ].copy()

    # Identify stations to highlight on the map
    stations_to_highlight_ids = final_filtered_data[result_station_id_col].unique()

    # --- Display Results ---
    st.subheader('Filtered Data Summary')
    st.write(f"Characteristic: **{selected_characteristic}**")
    st.write(f"Value range selected: **{value_range[0]:.2f} - {value_range[1]:.2f}**")
    st.write(f"Date range selected: **{date_start.strftime('%Y-%m-%d')} - {date_end.strftime('%Y-%m-%d')}**")
    st.write(f"Total data points matching filters: **{len(final_filtered_data)}**")
    st.write(f"Unique stations with matching data: **{len(stations_to_highlight_ids)}**")


    # Section 1: Map
    st.subheader('Station Locations for Filtered Data')
    if not stations_to_highlight_ids.size == 0:
        station_map = generate_station_map(station_df, lat_col, lon_col, stations_to_highlight_ids)
        if station_map:
            st_folium(station_map, height=500, width='100%')
        else:
            st.info("Could not generate map. Check station data or filters.")
    else:
        st.info("No stations to display on the map with the current filters.")

    # Section 2: Trend Plot
    st.subheader(f'Trend of {selected_characteristic} over Time for Filtered Stations')
    trend_plot_fig = plot_characteristic_trend(final_filtered_data, selected_characteristic, value_col, date_col, result_station_id_col)
    if trend_plot_fig:
        st.pyplot(trend_plot_fig)
    else:
        st.info("No trend data to display with the current filters.")
