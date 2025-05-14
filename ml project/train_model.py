import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV with correct delimiter (comma)
df = pd.read_csv(r'C:\Users\varsh\OneDrive\Desktop\ml project\2019.csv', sep='\t')


# Clean column names
df.columns = df.columns.str.strip()

# Display column names to verify
print("Columns:", df.columns.tolist())

# Drop unnecessary columns
df = df.drop(columns=['Country or region', 'Overall rank'])

# Set target and features
target_column = 'Score'
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Mean Absolute Error: {mae:.2f}")

# Save model and feature order
with open('happiness_model.pkl', 'wb') as f:
    pickle.dump((model, X.columns.tolist()), f)

print("✅ Model trained and saved as 'happiness_model.pkl'")
