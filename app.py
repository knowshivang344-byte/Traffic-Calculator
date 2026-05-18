import os
import joblib
import pandas as pd
import numpy as np
import datetime
import requests
import holidays
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Pro Traffic AI Predictor")

# Load model and assets
try:
    model = joblib.load('traffic_model.pkl')
    weather_encoder = joblib.load('weather_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
except Exception as e:
    print(f"Error loading model assets: {e}")
    model = None

class PredictionRequest(BaseModel):
    city: str

def map_wmo_to_weather_main(code: int):
    """Maps Open-Meteo WMO codes to Metro Dataset weather_main categories."""
    if code == 0: return 'Clear'
    if code in [1, 2, 3]: return 'Clouds'
    if code in [45, 48]: return 'Fog'
    if code in [51, 53, 55, 56, 57]: return 'Drizzle'
    if code in [61, 63, 65, 66, 67, 80, 81, 82]: return 'Rain'
    if code in [71, 73, 75, 77, 85, 86]: return 'Snow'
    if code in [95, 96, 99]: return 'Thunderstorm'
    return 'Clouds' # Fallback

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/predict")
def predict_traffic(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model assets not found.")
    
    # 1. Geocoding & Population
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={req.city}&count=1&language=en&format=json"
    geo_resp = requests.get(geo_url).json()
    
    if "results" not in geo_resp or not geo_resp["results"]:
        raise HTTPException(status_code=404, detail="City not found.")
        
    city_data = geo_resp["results"][0]
    lat, lon = city_data["latitude"], city_data["longitude"]
    country_code = city_data.get("country_code", "US")
    
    # Population-based scaling (Minneapolis Metro ~3.6M)
    # We use a log scale to prevent extreme outliers for mega-cities
    city_pop = city_data.get("population", 500000) # Default to 500k if unknown
    baseline_pop = 3600000 
    pop_scale = np.log10(city_pop) / np.log10(baseline_pop)
    pop_scale = max(0.5, min(pop_scale, 1.5)) # Bound scaling between 0.5x and 1.5x
    
    # 2. Weather & Local Time
    # Using timezone=auto to get the city's actual local time
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,rain,cloud_cover,weather_code,wind_speed_10m,relative_humidity_2m&timezone=auto"
    weather_resp = requests.get(weather_url).json()
    
    if "current" not in weather_resp:
        raise HTTPException(status_code=500, detail="Weather data unavailable.")
        
    curr = weather_resp["current"]
    local_time_str = curr["time"]
    local_dt = datetime.datetime.fromisoformat(local_time_str)
    
    # Feature Extraction
    hour = local_dt.hour
    day_of_week = local_dt.weekday()
    month = local_dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Cyclical Features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Weather Encoding
    weather_main = map_wmo_to_weather_main(curr["weather_code"])
    try:
        weather_enc = weather_encoder.transform([weather_main])[0]
    except:
        weather_enc = weather_encoder.transform(['Clouds'])[0]
        
    # Holiday Checking
    is_holiday = 0
    try:
        if local_dt in holidays.country_holidays(country_code):
            is_holiday = 1
    except:
        pass

    # 3. Model Prediction
    # ['hour_sin', 'hour_cos', 'day_of_week', 'month', 'is_weekend', 'holiday', 'temp', 'rain_1h', 'clouds_all', 'weather_enc']
    input_data = pd.DataFrame([[
        hour_sin, hour_cos, day_of_week, month, is_weekend, is_holiday,
        curr["temperature_2m"] + 273.15, # Convert to Kelvin
        curr.get("rain", 0.0),
        curr.get("cloud_cover", 0),
        weather_enc
    ]], columns=feature_names)
    
    base_prediction = model.predict(input_data)[0]
    final_prediction = int(base_prediction * pop_scale)
    
    # Traffic Status
    status = "Low"
    if final_prediction > 4000: status = "Heavy"
    elif final_prediction > 2000: status = "Moderate"
    
    # Generate 24-hour trend
    trend_data = []
    labels = []
    for h in range(24):
        h_sin = np.sin(2 * np.pi * h / 24)
        h_cos = np.cos(2 * np.pi * h / 24)
        row = pd.DataFrame([[
            h_sin, h_cos, day_of_week, month, is_weekend, is_holiday,
            curr["temperature_2m"] + 273.15,
            curr.get("rain", 0.0),
            curr.get("cloud_cover", 0),
            weather_enc
        ]], columns=feature_names)
        pred = int(model.predict(row)[0] * pop_scale)
        trend_data.append(pred)
        ampm = "AM" if h < 12 else "PM"
        display_h = h if h <= 12 else h - 12
        if display_h == 0: display_h = 12
        labels.append(f"{display_h} {ampm}")
        
    return {
        "city": city_data["name"],
        "country": city_data.get("country", "Unknown"),
        "local_time": local_dt.strftime("%I:%M %p"),
        "city_details": {
            "population": city_pop,
            "latitude": lat,
            "longitude": lon,
            "timezone": weather_resp.get("timezone", "Unknown")
        },
        "weather": {
            "main": weather_main,
            "temp_c": curr["temperature_2m"],
            "rain_mm": curr.get("rain", 0.0),
            "clouds": curr.get("cloud_cover", 0),
            "wind_kph": curr.get("wind_speed_10m", 0.0),
            "humidity": curr.get("relative_humidity_2m", 0)
        },
        "prediction": {
            "volume": final_prediction,
            "status": status,
            "pop_scale": round(pop_scale, 2)
        },
        "trend": {
            "labels": labels,
            "data": trend_data
        }
    }

# Ensure static dir exists and mount
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Global Traffic AI Server starting at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
