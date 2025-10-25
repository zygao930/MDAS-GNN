import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from scipy import sparse
import seaborn as sns
from matplotlib.colors import LogNorm
import geopandas as gpd
from shapely.geometry import Polygon
import contextily as ctx
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from scipy.stats import entropy
import sys
import os
from sklearn.metrics.pairwise import haversine_distances

# Central London boundary coordinates
central_london_coords = [
    (-0.2200, 51.4800), (-0.2000, 51.4750), (-0.1800, 51.4700),
    (-0.1500, 51.4650), (-0.1200, 51.4600), (-0.0900, 51.4650),
    (-0.0600, 51.4700), (-0.0400, 51.4800), (-0.0300, 51.5000),
    (-0.0400, 51.5200), (-0.0600, 51.5400), (-0.0800, 51.5500),
    (-0.1000, 51.5600), (-0.1200, 51.5650), (-0.1400, 51.5700),
    (-0.1600, 51.5650), (-0.1800, 51.5600), (-0.2000, 51.5500),
    (-0.2100, 51.5300), (-0.2150, 51.5100), (-0.2200, 51.4900),
    (-0.2200, 51.4800)
]

lons = [coord[0] for coord in central_london_coords]
lats = [coord[1] for coord in central_london_coords]
lon_min, lon_max = min(lons), max(lons)
lat_min, lat_max = min(lats), max(lats)

# Load data
df = pd.read_csv("./data/accident/accidents_merged.csv")
road_mapping = pd.read_csv("./data/accident/road_id_mapping.csv")

# Select essential columns
essential_cols = ["nearest_osm_id", "Date", "Time", "Accident_Severity", 
                 "Road_Surface_Conditions", "Road_Type", "Speed_limit", 
                 "Number_of_Casualties"]

optional_cols = ["Weather_Conditions", "Light_Conditions", "Junction_Control",
                 "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities"]
for col in optional_cols:
    if col in df.columns:
        essential_cols.append(col)

df = df[essential_cols].copy()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour

# Load geographic data
gdf = gpd.read_file('./data/GIS/TQ_RoadLink.shp')
gdf = gdf.to_crs('EPSG:27700')
accidents = pd.read_csv('./data/accident/accidents_2009_to_2014.csv', low_memory=False)

bbox = Polygon(central_london_coords)
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326").to_crs(gdf.crs)

# Configuration
CONFIG = {
    'bbox': (lon_min, lon_max, lat_min, lat_max), 
    'analysis_radius': 500,  
    'road_buffer': 300,
    'crs_proj': 'EPSG:27700',
    'crs_geo': 'EPSG:4326',
    'central_london_coords': central_london_coords  
}

# Risk weight configurations
PEDESTRIAN_CROSSING_HUMAN_CONTROL_WEIGHTS = {
    'Control by school crossing patrol': 0.2,      
    'Control by other authorised person': 0.3,    
    'None within 50 metres': 0.4,                  
}

PEDESTRIAN_CROSSING_PHYSICAL_FACILITIES_WEIGHTS = {
    'Footbridge or subway': 0.1,                           
    'Pedestrian phase at traffic signal junction': 0.2,
    'non-junction pedestrian crossing': 0.3,
    'Zebra crossing': 0.35,
    'Central refuge': 0.4,
    'No physical crossing within 50 meters': 0.6,
}

LIGHT_CONDITIONS_WEIGHTS = {
    'Daylight: Street light present': 0.2,
    'Darkness: Street lights present and lit': 0.4,
    'Darkness: Street lighting unknown': 0.6,
    'Darkness: Street lights present but unlit': 0.7,
    'Darkeness: No street lighting': 0.8,
}

JUNCTION_CONTROL_WEIGHTS = {
    'Authorised person': 0.2,
    'Automatic traffic signal': 0.3,
    'Stop Sign': 0.5,
    'Giveway or uncontrolled': 0.7,
}

ROAD_SURFACE_CONDITIONS_WEIGHTS = {
    'Dry': 0.2,
    'Wet/Damp': 0.5,
    'Snow': 0.7,
    'Frost/Ice': 0.8,
    'Flood (Over 3cm of water)': 0.7,
}

WEATHER_CONDITIONS_WEIGHTS = {
    'Fine without high winds': 0.2,
    'Fine with high winds': 0.3,
    'Raining without high winds': 0.5,
    'Raining with high winds': 0.7,
    'Fog or mist': 0.6,
    'Snowing without high winds': 0.7,
    'Snowing with high winds': 0.8,
    'Other': 0.4,
    'Unknown': 0.4
}

# Time range setup
start_date = df["Date"].min()
end_date = df["Date"].max()
start_date = start_date - pd.Timedelta(days=start_date.dayofweek)  # Align to week start
end_date = end_date + pd.Timedelta(days=(6 - end_date.dayofweek))  # Align to week end
num_weeks = (end_date - start_date).days // 7 + 1

# Road network processing
all_osm_ids = road_mapping['original_id'].unique()
num_nodes = len(all_osm_ids)
osm_id_to_index = {osm_id: idx for idx, osm_id in enumerate(all_osm_ids)}

# Road attributes preprocessing
road_attributes = {
    'Road_Type': {
        'Roundabout': 1.5, 'Slip road': 1.3, 
        'Dual carriageway': 1.2, 'One way street': 1.1,
        'Single carriageway': 1.0, np.nan: 1.0
    },
    'Speed_limit': lambda x: 0.5 + x/120  # Speed limit normalization
}

# Feature matrix initialization (3 core features)
# 0: accident_severity_intensity
# 1: infrastructure_risk  
# 2: environmental_risk
num_features = 3
aggregated_data = np.zeros((num_weeks, num_nodes, num_features))

# Add baseline road risk values
for _, road in road_mapping.iterrows():
    n = osm_id_to_index[road['original_id']]
    road_type = road.get('Road_Type', np.nan)
    speed = road.get('Speed_limit', 30)
    
    # Baseline risk = road type coefficient * speed coefficient
    base_risk = (
        road_attributes['Road_Type'].get(road_type, 1.0) *
        road_attributes['Speed_limit'](speed)
    )
    
    # Apply baseline infrastructure risk to all time periods
    aggregated_data[:, n, 1] = base_risk * 0.2

# Feature weight configurations
severity_weights = {1: 0.5, 2: 1.0, 3: 1.5}  # Fatal=1.5, Serious=1.0, Slight=0.5
hour_weights = {
    0: 0.6, 1: 0.4, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.6,
    6: 0.8, 7: 1.2, 8: 1.4, 9: 1.0, 10: 0.9, 11: 0.9,
    12: 1.0, 13: 1.1, 14: 1.1, 15: 1.3, 16: 1.4, 17: 1.5,
    18: 1.3, 19: 1.1, 20: 1.0, 21: 0.9, 22: 0.8, 23: 0.7
}

# Process accident data by week
infrastructure_processed_weeks = 0
environmental_processed_weeks = 0

for week_num in range(num_weeks):
    week_start = start_date + pd.Timedelta(weeks=week_num)
    week_end = week_start + pd.Timedelta(days=6)
    
    week_data = df[(df["Date"] >= week_start) & (df["Date"] <= week_end)]
    
    if len(week_data) == 0:
        continue
        
    week_idx = week_num
    week_road_count = 0
    week_infrastructure_count = 0
    week_environmental_count = 0
    
    for _, row in week_data.iterrows():
        osm_id = row["nearest_osm_id"]
        
        if osm_id not in osm_id_to_index:
            continue
            
        node_idx = osm_id_to_index[osm_id]
        week_road_count += 1
        
        # Feature 0: Accident Severity Intensity
        severity = row.get("Accident_Severity", 2)
        casualties = row.get("Number_of_Casualties", 1)
        hour = row.get("Time", 12)
        
        severity_weight = severity_weights.get(severity, 1.0)
        temporal_weight = hour_weights.get(hour, 1.0)
        
        severity_intensity = casualties * severity_weight * temporal_weight
        aggregated_data[week_idx, node_idx, 0] += severity_intensity
        
        # Feature 1: Infrastructure Risk
        infrastructure_risk = 0
        infrastructure_components = 0
        
        # Pedestrian crossing human control
        crossing_human = row.get("Pedestrian_Crossing-Human_Control", "")
        if pd.notna(crossing_human) and crossing_human in PEDESTRIAN_CROSSING_HUMAN_CONTROL_WEIGHTS:
            infrastructure_risk += PEDESTRIAN_CROSSING_HUMAN_CONTROL_WEIGHTS[crossing_human]
            infrastructure_components += 1
        
        # Pedestrian crossing physical facilities
        crossing_physical = row.get("Pedestrian_Crossing-Physical_Facilities", "")
        if pd.notna(crossing_physical) and crossing_physical in PEDESTRIAN_CROSSING_PHYSICAL_FACILITIES_WEIGHTS:
            infrastructure_risk += PEDESTRIAN_CROSSING_PHYSICAL_FACILITIES_WEIGHTS[crossing_physical]
            infrastructure_components += 1
        
        # Light conditions
        light = row.get("Light_Conditions", "")
        if pd.notna(light) and light in LIGHT_CONDITIONS_WEIGHTS:
            infrastructure_risk += LIGHT_CONDITIONS_WEIGHTS[light]
            infrastructure_components += 1
        
        # Junction control
        junction = row.get("Junction_Control", "")
        if pd.notna(junction) and junction in JUNCTION_CONTROL_WEIGHTS:
            infrastructure_risk += JUNCTION_CONTROL_WEIGHTS[junction]
            infrastructure_components += 1
        
        if infrastructure_components > 0:
            infrastructure_risk = infrastructure_risk / infrastructure_components
            existing_value = aggregated_data[week_idx, node_idx, 1]
            
            if existing_value > 0:
                aggregated_data[week_idx, node_idx, 1] = (existing_value + infrastructure_risk) / 2
            else:
                aggregated_data[week_idx, node_idx, 1] = infrastructure_risk
            week_infrastructure_count += 1
        
        # Feature 2: Environmental Risk
        environmental_risk = 0
        environmental_components = 0
        
        # Road surface conditions
        surface = row.get("Road_Surface_Conditions", "")
        if pd.notna(surface) and surface in ROAD_SURFACE_CONDITIONS_WEIGHTS:
            environmental_risk += ROAD_SURFACE_CONDITIONS_WEIGHTS[surface]
            environmental_components += 1
        
        # Weather conditions
        weather = row.get("Weather_Conditions", "")
        if pd.notna(weather) and weather in WEATHER_CONDITIONS_WEIGHTS:
            environmental_risk += WEATHER_CONDITIONS_WEIGHTS[weather]
            environmental_components += 1
        
        if environmental_components > 0:
            environmental_risk = environmental_risk / environmental_components
            existing_value = aggregated_data[week_idx, node_idx, 2]
            
            if existing_value > 0:
                aggregated_data[week_idx, node_idx, 2] = (existing_value + environmental_risk) / 2
            else:
                aggregated_data[week_idx, node_idx, 2] = environmental_risk
            week_environmental_count += 1
    
    if week_infrastructure_count > 0:
        infrastructure_processed_weeks += 1
    if week_environmental_count > 0:
        environmental_processed_weeks += 1

# Feature names
feature_names = [
    'Accident Severity Intensity', 
    'Infrastructure Risk', 
    'Environmental Risk'
]

# Spatial diffusion functions
def build_road_graph(road_mapping, k=3, sigma=1000):
    positions = np.column_stack([
        road_mapping['longitude'].values,
        road_mapping['latitude'].values
    ])
    
    # Calculate real road distance matrix using Haversine formula
    rad_pos = np.radians(positions)
    dist_matrix = haversine_distances(rad_pos) * 6371000  # Earth radius in meters
    
    # Build weighted adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    
    # Method A: Distance threshold connection
    threshold = 500  # Connect roads within 500 meters
    adj[dist_matrix <= threshold] = 1
    
    # Method B: kNN + distance weights
    for i in range(num_nodes):
        neighbors = np.argpartition(dist_matrix[i], k+1)[1:k+1]
        for j in neighbors:
            d = dist_matrix[i,j]
            adj[i,j] = np.exp(-d**2/(2*sigma**2))
    
    # Ensure symmetry
    adj = np.maximum(adj, adj.T)
    
    # Normalization
    rowsum = adj.sum(axis=1)
    adj = adj / (rowsum[:, np.newaxis] + 1e-6)
    
    return adj

def graph_diffusion(data, adj, alpha, iterations):
    """
    Improved graph diffusion algorithm
    Args:
        alpha: diffusion coefficient (recommended 0.2-0.5)
        iterations: diffusion iterations (recommended 1-3)
    """
    diffused = data.copy()
    degree = adj.sum(axis=1)
    
    for _ in range(iterations):
        # Use symmetric normalized diffusion
        diffused = (1-alpha)*diffused + alpha*(adj @ diffused)
        # Maintain original non-zero value weights
        mask = (data > 0)
        diffused[mask] = 0.7*diffused[mask] + 0.3*data[mask]
    
    return diffused

# Apply spatial diffusion
adj_matrix = build_road_graph(road_mapping)

for t in range(num_weeks):
    # Feature 0: Strong diffusion for accident severity (spatial clustering)
    aggregated_data[t, :, 0] = graph_diffusion(
        aggregated_data[t, :, 0], 
        adj_matrix,
        alpha=0.25,
        iterations=1
    )
    
    # Feature 1: Moderate diffusion for infrastructure risk
    aggregated_data[t, :, 1] = graph_diffusion(
        aggregated_data[t, :, 1], 
        adj_matrix,
        alpha=0.15,
        iterations=1
    )
    
    # Feature 2: Strong diffusion for environmental risk (weather correlation)
    aggregated_data[t, :, 2] = graph_diffusion(
        aggregated_data[t, :, 2], 
        adj_matrix,
        alpha=0.3,
        iterations=2
    )

# Post-processing
# Remove negative values
aggregated_data = np.maximum(aggregated_data, 0)

# Handle extreme outliers only
for i in range(num_features):
    feature_data = aggregated_data[:, :, i]
    if np.any(feature_data > 0):
        p999 = np.percentile(feature_data[feature_data > 0], 99.9)
        aggregated_data[:, :, i] = np.minimum(feature_data, p999)

# Cross-temporal consistency check
for i in range(num_features):
    feature_data = aggregated_data[:, :, i]
    
    nodes_time_coverage = []
    for node in range(num_nodes):
        time_steps_with_data = np.count_nonzero(feature_data[:, node])
        if time_steps_with_data > 0:
            nodes_time_coverage.append(time_steps_with_data)

# Save results
os.makedirs("./data/processed", exist_ok=True)

# Save main data
np.savez_compressed(
    "./data/accident/accident.npz",
    data=aggregated_data,
    node_ids=all_osm_ids,
    date_range=[start_date, end_date]
)

# Save visualization data
np.savez_compressed(
    "./data/processed/visualization_data.npz",
    aggregated_data=aggregated_data,
    road_mapping_ids=road_mapping['original_id'].values,
    road_mapping_longitude=road_mapping['longitude'].values,
    road_mapping_latitude=road_mapping['latitude'].values,
    feature_names=feature_names,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    num_weeks=num_weeks,
    adj_matrix=adj_matrix,
    central_london_coords=central_london_coords,
    bbox_bounds=(lon_min, lon_max, lat_min, lat_max)
)