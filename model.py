import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

def train_pro_model():
    print("Loading and cleaning dataset...")
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    
    # Drop duplicates by date_time to avoid bias
    df = df.drop_duplicates(subset=['date_time'], keep='first')
    
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Feature Engineering
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclical Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Encode Holiday
    df['holiday'] = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)
    
    # Encode Weather
    le = LabelEncoder()
    df['weather_enc'] = le.fit_transform(df['weather_main'])
    
    # Features
    features = ['hour_sin', 'hour_cos', 'day_of_week', 'month', 'is_weekend', 'holiday', 'temp', 'rain_1h', 'clouds_all', 'weather_enc']
    X = df[features]
    y = df['traffic_volume']
    
    print("Training High-Accuracy Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = RandomForestRegressor(n_estimators=150, max_depth=25, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\nModel Training Complete!")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f} vehicles")
    
    # Save assets
    joblib.dump(model, 'traffic_model.pkl')
    joblib.dump(le, 'weather_encoder.pkl')
    joblib.dump(features, 'feature_names.pkl')
    print("Model and encoders saved.")

if __name__ == "__main__":
    train_pro_model()