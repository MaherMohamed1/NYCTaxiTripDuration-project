#  NYC Taxi Trip Duration Prediction

A machine learning project that builds a **predictive regression model** to estimate NYC taxi trip durations (in seconds) using trip metadata such as
 pickup/dropoff coordinates, passenger count, and datetime features.

---

##  Project Overview

The project performs:

-  **Feature engineering** to derive spatial and temporal features  
-  **Data cleaning** and filtering  
-  **Model training** using Ridge Regression with polynomial expansion  
-  **Evaluation** using RMSE and R² metrics  
-  **Model serialization** for deployment via `pickle`

---

##  1. Code Structure

### `feature_engineering(df)`
Performs cleaning and feature extraction for the taxi dataset.

#### Steps:
1. **Remove invalid passengers**
   - Filter out rows with `passenger_count <= 0`
   - Clip upper passenger count at 6 (outlier mitigation)

2. **Compute geographic distances**
   - `Euclidean_dist`: Straight-line distance  
   - `Man_dist`: Manhattan distance  
   - `Haversine_dist`: Great-circle distance (most realistic)  
     ```
     2 * R * arcsin( sqrt( sin²(Δlat/2) + cos(lat1)cos(lat2)sin²(Δlong/2) ) )
     ```
     where `R = 6371 km`

3. **Compute direction (bearing)**
   - Bearing angle between pickup and dropoff  
   - Add `sin_dir` and `cos_dir` for circular encoding (avoid wraparound at 360°)

4. **Outlier removal on target**
   - Clip `trip_duration` between its 5th and 99th percentiles to reduce heavy-tailed noise

5. **Geographic boundaries**
   - Filter trips within expanded NYC region:  
     - Latitude: `35.00 → 45.00`  
     - Longitude: `-80.00 → -65.00`  
   - Keeps nearby airport trips while excluding extreme outliers

6. **Temporal feature extraction**
   - Parse `pickup_datetime` to datetime  
   - Create time-of-day bucket (`morning`, `midday`, `evening`, `late night`)  
   - Extract:
     - `hour`, `dayofweek`, `dayofyear`, `month`

7. **Weekend and Friday evening flags**
   - `is_weekend`: 1 if trip on Saturday/Sunday  
   - `is_friday_evening`: 1 if Friday between 7–11 PM (peak traffic)

**Output:**  
Returns a cleaned and feature-enriched DataFrame ready for modeling.

---

### `main(x_train, x_val, y_train, y_val, x_test, y_test)`
Trains and evaluates the regression model.

#### Steps:
1. **Feature groups**
   - Numeric: pickup/dropoff latitudes and longitudes  
   - Categorical: day/time indicators and trip metadata  

2. **Preprocessing pipeline**
   - `OneHotEncoder` for categorical variables  
   - `StandardScaler` for numerical variables  
   - Combined via `ColumnTransformer`

3. **Model pipeline**
   - `ColumnTransformer → PolynomialFeatures → Ridge Regression`  
   - Adds quadratic feature interactions for nonlinearity  
   - Regularized via `Ridge(alpha=1)` to prevent overfitting  

4. **Evaluation metrics**
   - **RMSE:** Root Mean Squared Error  
   - **R²:** Coefficient of Determination  

5. **Results printed**
   - Train RMSE / R²  
   - Validation R²  
   - Test RMSE / R²  

**Output:**  
Returns the trained `Pipeline` model ready for serialization.

---

### `if __name__ == '__main__':`
This block runs the entire training pipeline.

#### Workflow:
1. Load datasets (`train.csv`, `val.csv`, `test.csv`)  
2. Apply feature engineering to each  
3. Split features/targets  
4. Train and evaluate the model via `main()`  
5. Optionally, serialize model:
   ```python
   with open("model.pkl", "wb") as f:
       pickle.dump(model, f)

---

## 2. Model Rationale
**Stage**	           **Method**	                                          **Reason**
-Distance features	   -Haversine, Manhattan, Euclidean	                      -Capture spatial influence on duration
-Direction encoding	   -sin_dir, cos_dir	                                  -Handle cyclic angle data
-Time buckets          -Categorize hour into logical travel patterns          -Reflect traffic variations
-Polynomial Features   -Introducing feature interactions (e.g., lat × long)	  -Model nonlinear relationships
-Ridge Regression	   -Regularized linear model 							  -Robust to multicollinearity and overfitting

---

## 3. Evaluation Metrics
**Metric**	**Meaning**							   **Interpretation**
-RMSE	    -√MSE	                               -Measures average prediction error (in seconds). Lower = better.
-R²	        -Coefficient of Determination	       -Fraction of variance explained by the model (0–1). Higher = better.
**Example Output:**
-Train RMSE: `339.469`
-Train R²:   `0.705`
-Val RMSE:   `348.426`
-Val R²:     `0.686`
-Test RMSE:  `341.263`
-Test R²:    `0.706`

---

## 4. Project File Organization
NYCTTripDuration/
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   └── train_model.py      # ← this script
│
├── notebooks/
│   └── EDA.ipynb
│
└── README.md

---

## 5. Conceptual Summary
`Data Preprocessing` → `Feature Engineering` → `Model Pipeline (Polynomial Ridge Regression)` → `Evaluation` → `Model Deployment`
A complete supervised learning pipeline transforming raw trip data into an interpretable and deployable predictive model.

