import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    "data/processed/demand_predictions.csv"
)

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"]
)


# ============================================================
# 2. ACTUAL VS PREDICTED DEMAND
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    predictions["timestamp"],
    predictions["orders"],
    label="Actual Orders",
    alpha=0.7
)

plt.plot(
    predictions["timestamp"],
    predictions["predicted_orders"],
    label="Predicted Orders",
    alpha=0.7
)

plt.title("Actual vs Predicted Hourly Demand")
plt.xlabel("Date")
plt.ylabel("Number of Orders")

plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/figures/actual_vs_predicted_demand.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.read_csv(
    "data/processed/feature_importance.csv"
)

feature_importance = (
    feature_importance
    .sort_values("importance")
)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.title("Demand Prediction Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "outputs/figures/demand_feature_importance.png",
    dpi=300
)

plt.show()


print("\n========== PREDICTION VISUALIZATIONS COMPLETE ==========\n")
print("Saved figures to outputs/figures/")