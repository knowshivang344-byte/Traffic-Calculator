import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df = df.drop_duplicates(subset=['date_time'], keep='first')
df['date_time'] = pd.to_datetime(df['date_time'])

df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['holiday'] = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)

le = joblib.load('weather_encoder.pkl')
features = joblib.load('feature_names.pkl')
model = joblib.load('traffic_model.pkl')

df['weather_enc'] = le.transform(df['weather_main'])

X = df[features]
y = df['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("Making predictions...")
preds = model.predict(X_test)

# Set style
plt_sns.set_theme(style="whitegrid")

# 1. Actual vs Predicted (Accuracy Graph)
print("Generating Accuracy Graph...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.3, color='#00f2fe')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Model Accuracy: Actual vs Predicted Traffic Volume', fontsize=16, pad=15)
plt.xlabel('Actual Traffic Volume', fontsize=12)
plt.ylabel('Predicted Traffic Volume', fontsize=12)
plt.tight_layout()
plt.savefig('accuracy_graph.png', dpi=300)
plt.close()

# 2. Feature Importance Graph
print("Generating Feature Importance Graph...")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt_sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
plt.title('Model Performance: Feature Importance', fontsize=16, pad=15)
plt.xlabel('Relative Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# 3. Residuals Graph
print("Generating Residuals Graph...")
residuals = y_test - preds
plt.figure(figsize=(10, 6))
plt_sns.histplot(residuals, bins=50, kde=True, color='#f093fb')
plt.title('Error Distribution (Residuals)', fontsize=16, pad=15)
plt.xlabel('Error (Actual - Predicted Vehicles)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('residuals_graph.png', dpi=300)
plt.close()

print("Graphs generated successfully!")
