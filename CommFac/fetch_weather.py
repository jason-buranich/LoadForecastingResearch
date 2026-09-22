import pandas as pd
import requests

def fetch_latrobe_weather(start_date="2021-05-01", end_date="2022-04-30", output_csv="latrobe_weather_15min.csv"):
    """
    Fetches hourly historical weather data for La Trobe University (Bundoora Campus) 
    and interpolates it to 15-minute intervals.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": -37.718,   # Melbourne (Bundoora) coordinates
        "longitude": 145.048,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation",
        "timezone": "Australia/Melbourne"
    }
    
    print(f"Fetching hourly weather data for La Trobe ({start_date} to {end_date})...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API Request Failed: {response.status_code}\n{response.text}")
        
    data = response.json()
    hourly_data = data["hourly"]
    
    # Build DataFrame matching the expected covariate column names
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(hourly_data["time"]),
        "Temperature_2m": hourly_data["temperature_2m"],
        "Humidity": hourly_data["relative_humidity_2m"],
        "Solar_Rad": hourly_data["shortwave_radiation"]
    })
    
    df.set_index("Datetime", inplace=True)
    
    # Upsample to 15-minute intervals and apply linear interpolation
    print("Interpolating hourly data to 15-minute intervals...")
    df_15min = df.resample('15min').interpolate(method='linear')
    
    # Save to CSV
    df_15min.to_csv(output_csv)
    print(f"Successfully saved weather data to {output_csv}")
    print(f"Dataset Shape: {df_15min.shape}")

if __name__ == "__main__":
    
    fetch_latrobe_weather(start_date="2021-05-01", end_date="2022-04-30", output_csv="CommFac/latrobe_weather_15min.csv")