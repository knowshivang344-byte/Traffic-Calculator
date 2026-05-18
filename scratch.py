import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df = df.dropna()

# Convert numeric columns
df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df['rain_1h'] = pd.to_numeric(df['rain_1h'], errors='coerce')
df['clouds_all'] = pd.to_numeric(df['clouds_all'], errors='coerce')
df['traffic_volume'] = pd.to_numeric(df['traffic_volume'], errors='coerce')
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

# Time features
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month
df['day_of_week'] = df['date_time'].dt.dayofweek
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

# Holiday
df['holiday'] = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)

# Categorical weather
le = LabelEncoder()
df['weather_main_enc'] = le.fit_transform(df['weather_main'])

df = df.dropna()

# Features
features = ['sin_hour', 'cos_hour', 'month', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'holiday', 'weather_main_enc']
X = df[features]
y = df['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))

# Save label encoder and model
import joblib
joblib.dump(rf, 'traffic_model.pkl')
joblib.dump(le, 'weather_encoder.pkl')
joblib.dump(features, 'feature_names.pkl')
