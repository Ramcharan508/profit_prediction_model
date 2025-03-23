import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
csv_file = "1000_Companies.csv"
try:
    df = pd.read_csv(csv_file)  # Ensure this file is available
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
    exit()

# Selecting relevant features
features = ['R&D Spend', 'Administration', 'Marketing Spend']
target = 'Profit'

X = df[features]
y = df[target]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, "scaler.pkl")

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train and evaluate models
results = {}
best_model_name = None
best_r2 = -float("inf")  # Initialize with very low value

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
    
    # Save each model
    model_filename = f"{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_filename)

    # Update best model if this one has higher R2 score
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = model_filename

# Save the best model as 'best_model.pkl' (this is what app.py expects)
if best_model_name:
    joblib.dump(joblib.load(best_model_name), "best_model.pkl")
    print(f"Best model saved as: best_model.pkl")

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).T
print(results_df)
