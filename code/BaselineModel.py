import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures


def feature_engineering(df):
    df = df[df['passenger_count'] > 0].copy()
    df['passenger_count'] = df['passenger_count'].clip(upper=6)

    lat1, long1, lat2, long2 = df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude']

    df['Euclidean_dist'] = np.sqrt((long2 - long1) ** 2 + (lat2 - lat1) ** 2)
    df['Man_dist'] = np.abs(lat2 - lat1) + np.abs(long2 - long1) 

    def Haversine_dist(lat1, long1, lat2, long2):
        # D = 2R.arcsin√sin²(▽lat/2 ) + cos(lat1) * cos(lat2) * sin²(▽long/2)
        R = 6371
        lat1, long1, lat2, long2 = map(np.radians, [lat1, long1, lat2, long2])
        dlat = lat2 - lat1
        dlong = long2 - long1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    df['Haversine_dist'] = Haversine_dist(lat1, long1, lat2, long2)

    def bearing_array(lat1, long1, lat2, long2):
        R = 6371  # in km
        lat1, long1, lat2, long2 = map(np.radians, (lat1, long1, lat2, long2))
        dlong = long2 - long1
        y = np.sin(dlong) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlong)
        return np.degrees(np.arctan2(y, x))

    df['direction'] = bearing_array(lat1, long1, lat2, long2)
    df['sin_dir'] = np.sin(np.radians(df['direction']))
    df['cos_dir'] = np.cos(np.radians(df['direction']))

    lower = df['trip_duration'].quantile(0.05)
    upper = df['trip_duration'].quantile(0.99)
    df['trip_duration'] = df['trip_duration'].clip(lower, upper)

    Boundries = {
        "min_lat": 35.00,  # بدل 40.55
        "max_lat": 45.00,  # بدل 40.95
        "min_long": -80.00,  # بدل -74.15 (تشمل نيوآرك والمطارات)
        "max_long": -65.00  # بدل -73.70
    }
 
    df = df[
        (df['pickup_latitude'].between(Boundries["min_lat"], Boundries["max_lat"])) &
        (df['pickup_longitude'].between(Boundries["min_long"], Boundries["max_long"])) &
        (df['dropoff_latitude'].between(Boundries["min_lat"], Boundries["max_lat"])) &
        (df['dropoff_longitude'].between(Boundries["min_long"], Boundries["max_long"]))
        ].copy()


    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['dayofyear'] = df['pickup_datetime'].dt.dayofyear
    df['month'] = df['pickup_datetime'].dt.month

    def time_bucket(h): 
        if 4 <= h < 10:
            return 'morning'
        elif 10 <= h < 16:
            return 'midday'
        elif 16 <= h < 22:
            return 'evening'
        elif h >= 22 or h < 4:
            return 'late night'

    df['time_bucket'] = df['hour'].apply(time_bucket)

    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int) 
    df['is_friday_evening'] = ((df['dayofweek'] == 4) & (df['hour'] >= 19) & (df['hour'] <= 23)).astype(int)
    return df

def main(x_train, x_val, y_train, y_val, x_test, y_test):
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'time_bucket', 'store_and_fwd_flag', 'passenger_count']

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
    ], remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(fit_intercept=True, alpha=1))
    ])

    model = pipeline.fit(x_train, y_train)

    pred_y_train = model.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_y_train))
    print(f'Train RMSE: {train_rmse}')
    train_score = r2_score(y_train, pred_y_train)
    print(f'x score: {train_score:.10f}')

    pred_y_val = model.predict(x_val)
    val_score = r2_score(y_val, pred_y_val)
    print(f'r2score: {val_score:.10f}')

    pred_y_test = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_y_test))
    print(f'test rmse: {test_rmse:.10f}')
    test_score = r2_score(y_test, pred_y_test)
    print(f'r2 test score: {test_score:.10f}')
    return model

if __name__ == '__main__':
    train_df = pd.read_csv(r'C:\Users\Maher\NYCTTDProjects\NYCTTripDuration\data\train.csv')
    val_df = pd.read_csv(r'C:\Users\Maher\NYCTTDProjects\NYCTTripDuration\data\val.csv')
    test_df = pd.read_csv(r'C:\Users\Maher\NYCTTDProjects\NYCTTripDuration\data\test.csv')

    train_df = feature_engineering(df=train_df)
    val_df = feature_engineering(df=val_df)
    test_df = feature_engineering(df=test_df)

    drop_cols = ['id', 'pickup_datetime', 'trip_duration']

    x_train = train_df.drop(columns=drop_cols)
    y_train = train_df['trip_duration']

    x_val = val_df.drop(columns=drop_cols)
    y_val = val_df['trip_duration']

    x_test = test_df.drop(columns=drop_cols)
    y_test = test_df['trip_duration']
    model = main(x_train, x_val, y_train, y_val, x_test, y_test)

    with open(os.path.join(r'C:\Users\Maher\NYCTTDProjects\NYCTTripDuration', 'model.pkl'), 'wb') as file:
        pickle.dump(model, file)
