import sys
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('vacances-scolaires-france')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from vacances_scolaires_france import SchoolHolidayDates

# Load data
bike_df_train = pd.read_parquet("/kaggle/input/mdsb-2023/train.parquet")
bike_df_test = pd.read_parquet("/kaggle/input/mdsb-2023/final_test.parquet")

# Load the weather data
weather_data = pd.read_csv("/kaggle/input/mdsb-2023/external_data.csv")

# Initialize the SchoolHolidayDates object
school_holidays = SchoolHolidayDates()

def is_school_holiday(datetime_obj):
    date_obj = datetime_obj.date()
    return school_holidays.is_holiday_for_zone(date_obj, 'C')

def simplified_weather_categorization(row):
    temp_cold_threshold = 278.15
    temp_warm_threshold = 298.15
    rain_threshold = 1.0
    temp = row['t']
    rain = row.get('rr1', 0)
    if rain >= rain_threshold:
        return "Rainy"
    elif temp <= temp_cold_threshold:
        return "Cold"
    elif temp >= temp_warm_threshold:
        return "Warm"
    else:
        return "Moderate"

weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data['simplified_weather_category'] = weather_data.apply(simplified_weather_categorization, axis=1)

bike_df_train = bike_df_train.merge(weather_data[['date', 'simplified_weather_category']], on='date', how='left')
bike_df_test = bike_df_test.merge(weather_data[['date', 'simplified_weather_category']], on='date', how='left')

for df in [bike_df_train, bike_df_test]:
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_school_holiday'] = df['date'].apply(is_school_holiday)

def categorize_hour(hour):
    if 5 <= hour < 10:
        return 'morning'
    elif 10 <= hour < 15:
        return 'midday'
    elif 15 <= hour < 20:
        return 'afternoon'
    else:
        return 'night'

for df in [bike_df_train, bike_df_test]:
    df['time_of_day'] = df['hour'].apply(categorize_hour)

def create_weather_time_interaction(row):
    return f"{row['simplified_weather_category']}_{row['time_of_day']}"

for df in [bike_df_train, bike_df_test]:
    df['weather_time_interaction'] = df.apply(create_weather_time_interaction, axis=1)

for df in [bike_df_train, bike_df_test]:
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

columns_to_use = ['year', 'month', 'hour', 'weekday', 'is_school_holiday', 'latitude', 
                  'time_of_day', 'is_weekend', 'counter_name', 'simplified_weather_category', 'weather_time_interaction']

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), ['year', 'month', 'hour', 'weekday', 'latitude']),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ['time_of_day', 'is_school_holiday', 'counter_name', 'simplified_weather_category', 'weather_time_interaction']),
    ],
    remainder='passthrough'
)

X_train = bike_df_train[columns_to_use]
y_train = bike_df_train["log_bike_count"]
X_test = bike_df_test[columns_to_use]

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
X_test = preprocessor.transform(X_test)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

model = lgb.train(params,
                  train_data,
                  valid_sets=[train_data, valid_data],
                  num_boost_round=2000,
                  early_stopping_rounds=50,
                  verbose_eval=50)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

results = pd.DataFrame({'Id': np.arange(len(y_pred)), 'log_bike_count': y_pred})
results.to_csv("submission.csv", index=False)