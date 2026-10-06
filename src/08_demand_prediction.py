import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

orders = pd.read_csv(
    "data/processed/orders_engineered.csv"
)

orders["timestamp"] = pd.to_datetime(
    orders["timestamp"]
)


# ============================================================
# 2. AGGREGATE TO HOURLY DEMAND
# ============================================================

hourly_demand = (
    orders
    .set_index("timestamp")
    .resample("h")
    .agg(
        orders=("order_id", "count"),
        units_sold=("quantity", "sum"),
        revenue=("line_revenue", "sum")
    )
    .reset_index()
)


# ============================================================
# 3. CREATE TIME FEATURES
# ============================================================

hourly_demand["hour"] = (
    hourly_demand["timestamp"].dt.hour
)

hourly_demand["day_of_week"] = (
    hourly_demand["timestamp"].dt.dayofweek
)

hourly_demand["day_of_month"] = (
    hourly_demand["timestamp"].dt.day
)

hourly_demand["month"] = (
    hourly_demand["timestamp"].dt.month
)

hourly_demand["is_weekend"] = (
    hourly_demand["day_of_week"] >= 5
).astype(int)


# ============================================================
# 4. CREATE TRUE TIME-BASED LAG FEATURES
# ============================================================

hourly_demand["orders_lag_24"] = (
    hourly_demand["orders"].shift(24)
)

hourly_demand["orders_lag_168"] = (
    hourly_demand["orders"].shift(168)
)

hourly_demand["orders_lag_1"] = (
    hourly_demand["orders"].shift(1)
)


# ============================================================
# 5. CREATE ROLLING FEATURE
# ============================================================

hourly_demand["orders_rolling_24"] = (
    hourly_demand["orders"]
    .shift(1)
    .rolling(24)
    .mean()
)


# ============================================================
# 6. KEEP RESTAURANT OPERATING HOURS
# ============================================================

hourly_demand = hourly_demand[
    hourly_demand["timestamp"].dt.hour.between(11, 21)
].copy()

# Remove rows where lag/rolling features are unavailable
hourly_demand = hourly_demand.dropna().copy()

print("========== HOURLY DEMAND ==========\n")
print(hourly_demand.head())

print(
    "\nNumber of operating-hour observations:",
    len(hourly_demand)
)


# ============================================================
# 7. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "orders_lag_1",
    "orders_lag_24",
    "orders_lag_168",
    "orders_rolling_24"
]

X = hourly_demand[features]
y = hourly_demand["orders"]


# ============================================================
# 8. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(hourly_demand) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== TRAIN / TEST SPLIT ==========\n")

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print(
    "Training period:",
    hourly_demand["timestamp"].iloc[0],
    "to",
    hourly_demand["timestamp"].iloc[split_index - 1]
)

print(
    "Testing period:",
    hourly_demand["timestamp"].iloc[split_index],
    "to",
    hourly_demand["timestamp"].iloc[-1]
)


# ============================================================
# 9. BASELINE
# ============================================================

baseline_prediction = X_test["orders_lag_24"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)


# ============================================================
# 11. EVALUATE
# ============================================================

model_mae = mean_absolute_error(
    y_test,
    predictions
)

model_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mae_improvement = (
    (baseline_mae - model_mae)
    / baseline_mae
    * 100
)

rmse_improvement = (
    (baseline_rmse - model_rmse)
    / baseline_rmse
    * 100
)


print("\n========== MODEL RESULTS ==========\n")

print(
    "Baseline MAE:",
    round(baseline_mae, 2)
)

print(
    "Random Forest MAE:",
    round(model_mae, 2)
)

print(
    "MAE improvement:",
    round(mae_improvement, 2),
    "%"
)

print(
    "\nBaseline RMSE:",
    round(baseline_rmse, 2)
)

print(
    "Random Forest RMSE:",
    round(model_rmse, 2)
)

print(
    "RMSE improvement:",
    round(rmse_improvement, 2),
    "%"
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_importance = (
    pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })
    .sort_values(
        "importance",
        ascending=False
    )
)

print("\n========== FEATURE IMPORTANCE ==========\n")
print(feature_importance)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

results = hourly_demand.loc[
    X_test.index,
    ["timestamp", "orders"]
].copy()

results["predicted_orders"] = predictions
results["baseline_orders"] = baseline_prediction.values

results.to_csv(
    "data/processed/demand_predictions.csv",
    index=False
)

feature_importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print("\n========== PREDICTIONS SAVED ==========\n")
print("Saved demand_predictions.csv")
print("Saved feature_importance.csv")