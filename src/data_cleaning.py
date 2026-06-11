import pandas as pd
import numpy as np
import os

def load_raw_datasets(base_path="../data/raw"):
    weather_path = os.path.join(base_path, "weather_daily.csv")
    soil_path = os.path.join(base_path, "soil_sensor_data.csv")

    weather_df = pd.read_csv(weather_path, na_values=['NA', ''])
    soil_df = pd.read_csv(soil_path, na_values=['NA', ''])

    weather_df['date'] = pd.to_datetime(weather_df['date'])
    soil_df['timestamp'] = pd.to_datetime(soil_df['timestamp'])

    return weather_df, soil_df


def clean_weather_data(weather_df):
    df_clean = weather_df.copy()
    
    df_clean.loc[df_clean['temperature_c'] > 40.0, 'temperature_c'] = np.nan
    df_clean['temperature_c'] = df_clean['temperature_c'].ffill()
    
    df_clean['rainfall_mm'] = df_clean['rainfall_mm'].fillna(0.0)
    
    numeric_cols = ['humidity_pct', 'wind_speed_mps', 'solar_index']
    for col in numeric_cols:
        df_clean[col] = df_clean[col].interpolate(method='linear')
        
    return df_clean


def clean_soil_data(soil_df):
    df_clean = soil_df.copy()
    
    # CRITICAL FIX: Sort chronologically to ensure accurate time-series interpolation
    df_clean = df_clean.sort_values(by=['zone_id', 'timestamp']).reset_index(drop=True)
    
    # Isolate known physical outliers and transmission errors
    df_clean.loc[df_clean['tank_level_liters'] > 5000, 'tank_level_liters'] = np.nan
    df_clean.loc[df_clean['sensor_status'] == 'CHECK', 'soil_moisture_pct'] = np.nan
    
    # Apply interpolation on correctly ordered data
    df_clean['soil_moisture_pct'] = df_clean.groupby('zone_id')['soil_moisture_pct'].transform(
        lambda group: group.interpolate(method='linear')
    )
    
    df_clean['tank_level_liters'] = df_clean.groupby('zone_id')['tank_level_liters'].transform(
        lambda group: group.interpolate(method='linear')
    )
    
    return df_clean

def build_unified_dataset(soil_clean, weather_clean):
    soil_clean['join_date'] = soil_clean['timestamp'].dt.date
    weather_clean['join_date'] = weather_clean['date'].dt.date
    
    merged_df = pd.merge(soil_clean, weather_clean, on='join_date', how='left')
    merged_df.drop(columns=['join_date'], inplace=True)
    
    return merged_df


def process_pipeline():
    weather_raw, soil_raw = load_raw_datasets()
    weather_clean = clean_weather_data(weather_raw)
    soil_clean = clean_soil_data(soil_raw)
    final_df = build_unified_dataset(soil_clean, weather_clean)
    return final_df


def export_cleaned_data(df, output_dir="../data/processed", filename="cleaned_irrigation_dataset.csv"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    return output_path

def generate_structural_dictionary(df, dataset_name="Dataset"):
    """
    Generates structural metadata for a dataframe,
    acting as an automated component of the project's Data Dictionary.
    """
    print(f"\n===Structural Dictionary: {dataset_name} ===")
    print(f"Total Rows: {df.shape[0]} | Total Features: {df.shape[1]}\n")
    
    # Build metadata columns programmatically
    summary_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Null Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    return summary_df