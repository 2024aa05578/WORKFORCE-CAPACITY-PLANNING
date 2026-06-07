
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load Data
df = pd.read_csv('data/workforce_data.csv')

# Date features
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month

# Convert location
df['Location'] = df['Location'].astype('category').cat.codes

# Features
X = df[['Location', 'Engineers', 'Skill_Level', 'Day', 'Month']]
y = df['Workload']

# Train Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save Model
joblib.dump(model, 'models/workforce_model.pkl')

print("Model Saved")
