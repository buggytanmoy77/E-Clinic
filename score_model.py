import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

Input_data = pd.read_csv("Final_E-clinic_dataset.csv") # E-clinic Dataset which has Vital Signs data and Health Score

# Ideal Values For all the vital Signs
ideal_sbp = 120
ideal_dbp = 80
ideal_bmi = 21.7
ideal_sp02 = 95
ideal_heart_rate = 75
ideal_respiratory_rate = 16
ideal_body_temp = 36.75
total_ideal_weights = 1.0

# Function to calculate health score
def calculate_health_score(row):
    systolic_bp = row['Systolic Blood Pressure']
    diastolic_bp = row['Diastolic Blood Pressure']
    bmi = row['Derived_BMI']
    sp02 = row['Oxygen Saturation']
    heart_rate = row['Heart Rate']
    respiratory_rate = row['Respiratory Rate']
    body_temp = row['Body Temperature']

    # Calculate deviations from the ideal values
    norm_deviation_sbp = abs(systolic_bp - ideal_sbp) / ideal_sbp if not (110 <= systolic_bp <= 140) else 0
    norm_deviation_dbp = abs(diastolic_bp - ideal_dbp) / ideal_dbp if not (70 <= systolic_bp <= 90) else 0
    norm_deviation_bmi = abs(bmi - ideal_bmi) / ideal_bmi if not (18.5 <= bmi <= 24.9) else 0
    norm_deviation_body_temp = abs(body_temp - ideal_body_temp) / ideal_body_temp if not (36 <= body_temp <= 37.5) else 0
    norm_deviation_heart_rate = abs(heart_rate - ideal_heart_rate) / ideal_heart_rate if not (60 <= heart_rate <= 90) else 0
    norm_deviation_respiratory_rate = abs(respiratory_rate - ideal_respiratory_rate) / ideal_respiratory_rate if not (12 <= respiratory_rate <= 20) else 0
    norm_deviation_sp02 = abs(sp02 - ideal_sp02) / ideal_sp02

    # Feature scores
    feature_score_sbp = max(0, 1 - norm_deviation_sbp)
    feature_score_dbp = max(0, 1 - norm_deviation_dbp)
    feature_score_bmi = max(0, 1 - norm_deviation_bmi)
    feature_score_sp02 = max(0, 1 - norm_deviation_sp02)
    feature_score_heart_rate = max(0, 1 - norm_deviation_heart_rate)
    feature_score_respiratory_rate = max(0, 1 - norm_deviation_respiratory_rate)
    feature_score_body_temp = max(0, 1 - norm_deviation_body_temp)

    # Weighted score calculation for making the model dynamic
    total_weighted_score = (
        feature_score_sbp * 0.20 +
        feature_score_dbp * 0.20 +
        feature_score_bmi * 0.15 +
        feature_score_sp02 * 0.15 +
        feature_score_heart_rate * 0.15 +
        feature_score_respiratory_rate * 0.10 +
        feature_score_body_temp * 0.05
    )
    health_score = (total_weighted_score / total_ideal_weights) * 100
    return health_score

# Calculate Health Scores for all data
Input_data['Health Score'] = Input_data.apply(calculate_health_score, axis=1)

# Select relevant features for model training
features = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
            'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Weight (kg)', 'Height (m)']

# Spliting data into train and test
train_data, test_data = train_test_split(Input_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Standardization 
scaler = StandardScaler()
train_data[features] = scaler.fit_transform(train_data[features])
val_data[features] = scaler.transform(val_data[features])
test_data[features] = scaler.transform(test_data[features])

# Training Random Forest model
RF_model = RandomForestRegressor(n_estimators=100, random_state=42)
X_train = train_data[features]
y_train = train_data['Health Score']
RF_model.fit(X_train, y_train)

# Save the trained model using pickle
with open('health_score_model.pkl', 'wb') as model_file:
    pickle.dump(RF_model, model_file)

print("Model saved successfully!")
