import requests
import pandas as pd

def fetch_regional_weather(lat, lon, region_prefix, start_date="2025-01-01", end_date="2025-12-31"):
    """Fetches hourly weather data from Open-Meteo for a specific coordinate."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,shortwave_radiation",
        "timezone": "America/Los_Angeles"
    }
    
    print(f"Fetching data for {region_prefix}...")
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract hourly data
    hourly = data['hourly']
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date_Time": pd.to_datetime(hourly['time']),
        f"{region_prefix}_Temp": hourly['temperature_2m'],
        f"{region_prefix}_Solar": hourly['shortwave_radiation']
    })
    
    return df

if __name__ == "__main__":
    # Define representative coordinates for each CAISO Load Zone
    regions = {
        "PGE":  {"lat": 38.5816, "lon": -121.4944}, # Sacramento
        "SCE":  {"lat": 34.0522, "lon": -118.2437}, # Los Angeles
        "SDGE": {"lat": 32.7157, "lon": -117.1611}, # San Diego
        "VEA":  {"lat": 36.2083, "lon": -115.9839}  # Pahrump, NV
    }
    
    # Fetch data for all regions
    dfs = []
    for region, coords in regions.items():
        df = fetch_regional_weather(coords['lat'], coords['lon'], region)
        dfs.append(df)
        
    # Merge all regional dataframes on the timestamp
    final_df = dfs[0]
    for i in range(1, len(dfs)):
        final_df = pd.merge(final_df, dfs[i], on="Date_Time")
        
    # Split Date_Time into 'Date' and 'HR' to match your CAISO excel file
    final_df['Date'] = final_df['Date_Time'].dt.date
    final_df['HR'] = final_df['Date_Time'].dt.hour
    
    # Drop the combined timestamp and reorder columns
    final_df = final_df.drop(columns=['Date_Time'])
    
    # Save to CSV
    output_file = "CAISO_Regional_Weather_2025.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Success! Regional weather saved to {output_file}")