from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# API credentials
rapid_api_host = "rto-vehicle-information-verification-india.p.rapidapi.com"
rapid_api_key = "3fbf2cd874msh96cd513cd2a4575p1becb4jsnc0da7aeeab7d"

# Driving Behavior Functions
def preprocess_data(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.ffill()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method='ffill')
    df['Speed (km/h)'] = df['Speed (km/h)'].apply(lambda x: max(x, 0))
    df['Vehicle Distance (km)'] = df['Vehicle Distance (km)'].apply(lambda x: max(x, 0))
    return df

def enhanced_feature_engineering(df):
    df = df.sort_values(by=['Trip_ID', 'Timestamp'])
    df['Acceleration (m/s^2)'] = df.groupby('Trip_ID')['Speed (km/h)'].diff().fillna(0) / df.groupby('Trip_ID')['Timestamp'].diff().dt.total_seconds().fillna(1)
    df['Deceleration (m/s^2)'] = df['Acceleration (m/s^2)'].apply(lambda x: x if x < 0 else 0)
    df['Fuel Efficiency (km/L)'] = df['Vehicle Distance (km)'] / (df['Fuel Consumption (L/100km)'] / 100)
    df['RPM Efficiency'] = df['RPM'] / df['Gear'].replace(0, 1)
    df['Overspeed'] = (df['Speed (km/h)'] > 120).astype(int)
    df['High RPM'] = (df['RPM'] > 3000).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method='ffill')
    return df

def aggregate_trip_data(df):
    behavior_data = df.groupby('Trip_ID').agg({
        'Timestamp': ['min', 'max'],
        'Acceleration (m/s^2)': 'mean',
        'Deceleration (m/s^2)': 'mean',
        'Fuel Efficiency (km/L)': 'mean',
        'RPM Efficiency': 'mean',
        'Overspeed': 'sum',
        'High RPM': 'sum'
    })
    behavior_data.columns = ['start_time', 'end_time', 'Avg acceleration', 'Avg deceleration',
                             'Avg fuel_efficiency', 'Avg rpm_efficiency', 'Overspeeding Frequency', 'High_rpm_frequency']
    behavior_data = behavior_data.reset_index()
    behavior_data['Trip Duration'] = (behavior_data['end_time'] - behavior_data['start_time']).dt.total_seconds() / 60
    return behavior_data

def label_driving_behavior(behavior_data):
    behavior_data['Aggressive_Driving'] = (
        (behavior_data['Avg acceleration'] > 3) | 
        (behavior_data['Avg deceleration'] < -3) | 
        (behavior_data['Overspeeding Frequency'] > 1) | 
        (behavior_data['High_rpm_frequency'] > 1)
    ).astype(int)
    behavior_data['Driving_Behavior'] = behavior_data['Aggressive_Driving'].apply(lambda x: 'Aggressive' if x == 1 else 'Efficient')
    return behavior_data

def apply_model(behavior_data):
    feature_columns = ['Avg acceleration', 'Avg deceleration', 'Avg fuel_efficiency', 'Avg rpm_efficiency',
                       'Overspeeding Frequency', 'High_rpm_frequency', 'Trip Duration']
    X = behavior_data[feature_columns]
    y = behavior_data['Aggressive_Driving']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return model

def driving_behavior_pipeline(df):
    df = preprocess_data(df)
    df = enhanced_feature_engineering(df)
    driving_data = aggregate_trip_data(df)
    driving_data = label_driving_behavior(driving_data)
    model = apply_model(driving_data)
    return model, driving_data

def train_and_predict(car_name, brand, model, fuel_type, km_driven, df):
    X = df[['car_name', 'brand', 'model', 'fuel_type', 'km_driven']]
    y = df['selling_price']
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['car_name', 'brand', 'model', 'fuel_type']),
        remainder='passthrough'
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=506)
    xgb = XGBRegressor()
    pipe = make_pipeline(column_trans, xgb)
    pipe.fit(X_train, y_train)
    input_data = pd.DataFrame({
        'car_name': [car_name],
        'brand': [brand],
        'model': [model],
        'fuel_type': [fuel_type],
        'km_driven': [km_driven]
    })
    prediction = pipe.predict(input_data)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/car-login')
def car_login():
    return render_template('car-login.html')

@app.route('/submit-car-number', methods=['POST'])
def submit_car_number():
    car_number = request.form['car_number']
    
    # Call RapidAPI to fetch car details
    url = "https://rto-vehicle-information-verification-india.p.rapidapi.com/api/v1/rc/vehicleinfo"
    payload = {
        "reg_no": car_number,
        "consent": "Y",
        "consent_text": "I hear by declare my consent agreement for fetching my information via AITAN Labs API"
    }
    headers = {
        "x-rapidapi-key": rapid_api_key,
        "x-rapidapi-host": rapid_api_host,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        car_data1 = response.json()
        car_data = car_data1["result"]
        df = pd.read_csv('cardekho_dataset.csv')  # Replace with actual dataset

        # Predict the car price
        predicted_price = train_and_predict(
            car_name=car_data['vehicle_manufacturer_name'] + " " + car_data['model'], 
            brand=car_data['vehicle_manufacturer_name'], 
            model=car_data['model'], 
            fuel_type=car_data['fuel_descr'], 
            km_driven=25000, 
            df=df
        )
        
        # Load driving behavior data (ensure the CSV path is correct)
        try:
            driving_behavior_df = pd.read_csv('Cleaned OBD-2 Data 2.csv')  # Ensure this filename matches your dataset
        except FileNotFoundError:
            return "Driving behavior data file not found. Please check the path."

        model, driving_data = driving_behavior_pipeline(driving_behavior_df)

        # Prepare the trip options
        trip_options = driving_data['Trip_ID'].unique().tolist()

        # Render the car details dashboard with fetched data, predicted price, and trip options
        return render_template('car-details.html', car_data=car_data, predicted_price=predicted_price,
                               driving_data=driving_data, trip_options=trip_options)
    else:
        return f"Failed to fetch details for car number {car_number}. Please try again.<br>Details: {response.text}"

@app.route('/car-details/<selected_trip>', methods=['GET'])
def car_details(selected_trip):
    # Load the driving behavior data and filter by selected trip
    try:
        driving_behavior_df = pd.read_csv('Cleaned OBD-2 Data 2.csv')  # Ensure this filename matches your dataset
    except FileNotFoundError:
        return "Driving behavior data file not found. Please check the path."

    driving_data = aggregate_trip_data(driving_behavior_df)
    selected_trip_data = driving_data[driving_data['Trip_ID'] == selected_trip]

    return render_template('trip-details.html', trip_data=selected_trip_data)

if __name__ == '__main__':
    app.run(debug=True)
