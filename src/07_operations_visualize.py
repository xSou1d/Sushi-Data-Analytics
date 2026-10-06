import pandas as pd
import matplotlib.pyplot as plt

# Load analysis results
ingredient_analysis = pd.read_csv(
    "data/processed/ingredient_analysis.csv",
    index_col="ingredient"
)

daily_operations = pd.read_csv(
    "data/processed/daily_operations.csv"
)

daily_operations["date"] = pd.to_datetime(
    daily_operations["date"]
)

# ============================================================
# 1. WASTE COST BY INGREDIENT
# ============================================================

waste_cost = (
    ingredient_analysis["waste_cost"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

waste_cost.plot(kind="bar")

plt.title("Estimated Waste Cost by Ingredient")
plt.xlabel("Ingredient")
plt.ylabel("Waste Cost ($)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    "outputs/figures/waste_cost_by_ingredient.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. REVENUE PER LABOR HOUR
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    daily_operations["date"],
    daily_operations["revenue_per_labor_hour"]
)

plt.title("Revenue per Labor Hour Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue per Labor Hour ($)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/figures/revenue_per_labor_hour.png",
    dpi=300
)

plt.show()


print("\n========== OPERATIONS VISUALIZATIONS COMPLETE ==========\n")
print("Saved figures to outputs/figures/")