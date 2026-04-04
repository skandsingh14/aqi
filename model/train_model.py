import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
import zipfile

def create_real_data(zip_filename="archive (2).zip", max_samples=250000):
    print(f"Streaming data directly from {zip_filename} to preserve memory...")
    
    # We will collect dataframes efficiently
    dfs = []
    total_loaded = 0
    
    with zipfile.ZipFile(zip_filename, 'r') as z:
        # Get all CSV files except stations.csv
        csv_files = [f for f in z.namelist() if f.endswith('.csv') and 'stations' not in f.lower()]
        
        # Shuffle to get a good mix of states
        np.random.seed(42)
        np.random.shuffle(csv_files)
        
        for file in csv_files:
            if total_loaded >= max_samples:
                break
                
            try:
                # Read a chunk from the CSV
                with z.open(file) as f:
                    chunk = pd.read_csv(f, usecols=['pm2.5', 'pm10', 'no2', 'so2', 'co', 'ozone'])
                    
                    # Drop nulls
                    chunk = chunk.dropna()
                    
                    if chunk.empty:
                        continue
                        
                    # Standardize columns to match our API expectations
                    chunk = chunk.rename(columns={
                        'pm2.5': 'PM2_5',
                        'pm10': 'PM10',
                        'no2': 'NO2',
                        'so2': 'SO2',
                        'ozone': 'O3'
                    })
                    
                    # API returns CO in ug/m3, but CPCB dataset has it in mg/m3
                    # Multiply by 1000 to normalize training data to match live API inputs
                    chunk['CO'] = chunk['co'] * 1000.0
                    chunk = chunk.drop(columns=['co'])
                    
                    # Shuffle and take a safe sample (e.g., 5000 rows max per file so we get all states)
                    if len(chunk) > 5000:
                        chunk = chunk.sample(n=5000, random_state=42)
                        
                    dfs.append(chunk)
                    total_loaded += len(chunk)
                    print(f"Loaded {len(chunk)} clean records from {file}... (Total: {total_loaded})")
                    
            except Exception as e:
                continue
                
    # Combine all chunks
    df = pd.concat(dfs, ignore_index=True)
    
    # Cap if we went slightly over
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        
    print(f"Final combined dataset size: {len(df)} rows.")

    def calc_pm25(x):
        if x <= 30: return x * 50 / 30
        elif x <= 60: return 50 + (x - 30) * 50 / 30
        elif x <= 90: return 100 + (x - 60) * 100 / 30
        elif x <= 120: return 200 + (x - 90) * 100 / 30
        elif x <= 250: return 300 + (x - 120) * 100 / 130
        else: return 400 + (x - 250) * 100 / 130

    def calc_pm10(x):
        if x <= 50: return x
        elif x <= 100: return x
        elif x <= 250: return 100 + (x - 100) * 100 / 150
        elif x <= 350: return 200 + (x - 250) * 100 / 100
        elif x <= 430: return 300 + (x - 350) * 100 / 80
        else: return 400 + (x - 430) * 100 / 80

    def calc_no2(x):
        if x <= 40: return x * 50 / 40
        elif x <= 80: return 50 + (x - 40) * 50 / 40
        elif x <= 180: return 100 + (x - 80) * 100 / 100
        elif x <= 280: return 200 + (x - 180) * 100 / 100
        elif x <= 400: return 300 + (x - 280) * 100 / 120
        else: return 400 + (x - 400) * 100 / 120

    def calc_so2(x):
        if x <= 40: return x * 50 / 40
        elif x <= 80: return 50 + (x - 40) * 50 / 40
        elif x <= 380: return 100 + (x - 80) * 100 / 300
        elif x <= 800: return 200 + (x - 380) * 100 / 420
        elif x <= 1600: return 300 + (x - 800) * 100 / 800
        else: return 400 + (x - 1600) * 100 / 800

    def calc_co(x):
        if x <= 1.0: return x * 50 / 1.0
        elif x <= 2.0: return 50 + (x - 1.0) * 50 / 1.0
        elif x <= 10.0: return 100 + (x - 2.0) * 100 / 8.0
        elif x <= 17.0: return 200 + (x - 10.0) * 100 / 7.0
        elif x <= 34.0: return 300 + (x - 17.0) * 100 / 17.0
        else: return 400 + (x - 34.0) * 100 / 17.0

    def calc_o3(x):
        if x <= 50: return x * 50 / 50
        elif x <= 100: return 50 + (x - 50) * 50 / 50
        elif x <= 168: return 100 + (x - 100) * 100 / 68
        elif x <= 208: return 200 + (x - 168) * 100 / 40
        elif x <= 748: return 300 + (x - 208) * 100 / 540
        else: return 400 + (x - 748) * 100 / 540

    df['AQI'] = 0
    print("Computing verified Indian Standard AQI classifications for all real-world sensor records...")
    
    # Vectorized computation for speed on large datasets
    pm25_aqi = df['PM2_5'].apply(calc_pm25)
    pm10_aqi = df['PM10'].apply(calc_pm10)
    no2_aqi = df['NO2'].apply(calc_no2)
    so2_aqi = df['SO2'].apply(calc_so2)
    co_aqi = (df['CO'] / 1000.0).apply(calc_co)  # Pass as mg/m3 into calc_co
    o3_aqi = df['O3'].apply(calc_o3)
    
    df['AQI'] = np.maximum.reduce([pm25_aqi, pm10_aqi, no2_aqi, so2_aqi, co_aqi, o3_aqi])
    df['AQI'] = np.clip(df['AQI'], 10, 500).astype(int)

    # Ensure column order matches API precisely
    features = ['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    X = df[features]
    y = df['AQI']
    
    return X, y

if __name__ == '__main__':
    # Create directory if missing
    os.makedirs('model', exist_ok=True)
    
    X, y = create_real_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model trained with R^2 score: {score:.3f}")
    
    # Save the model
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
