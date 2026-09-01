import requests
import pandas as pd

def fetch_historical_weather():
    print("Fetching historical weather for Punchbowl, Sydney...")
    
    # Coordinates for Punchbowl, NSW
    lat, lon = -33.9333, 151.0450
    
    # We pull a wide date range to ensure it covers FY25
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date=2024-01-01&end_date=2025-12-31&"
        f"hourly=temperature_2m,relative_humidity_2m,shortwave_radiation&"
        f"timezone=Australia%2FSydney"
    )
    
    response = requests.get(url)
    data = response.json()
    
    # Parse into DataFrame
    hourly_df = pd.DataFrame({
        'Datetime': pd.to_datetime(data['hourly']['time']),
        'Temperature_2m': data['hourly']['temperature_2m'],
        'Humidity': data['hourly']['relative_humidity_2m'],
        'Solar_Rad': data['hourly']['shortwave_radiation']
    })
    
    # Set index and remove timezone info to match substation data
    hourly_df.set_index('Datetime', inplace=True)
    hourly_df.index = hourly_df.index.tz_localize(None)
    
    # Resample to 15-minute intervals and interpolate the weather gaps
    print("Resampling hourly weather to 15-minute intervals...")
    weather_15min = hourly_df.resample('15min').interpolate(method='linear')
    
    weather_15min.to_csv("Substation/punchbowl_weather_15min.csv")
    print("Saved to Substation/punchbowl_weather_15min.csv")

if __name__ == "__main__":
    fetch_historical_weather()