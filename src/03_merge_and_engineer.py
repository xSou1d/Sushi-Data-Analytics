import pandas as pd


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

orders = pd.read_csv("data/processed/orders_clean.csv")
menu = pd.read_csv("data/processed/menu_clean.csv")


# ============================================================
# 2. NORMALIZE MENU ITEM NAMES
# ============================================================

orders["menu_item"] = orders["menu_item"].str.strip().str.title()
menu["menu_item"] = menu["menu_item"].str.strip().str.title()


# ============================================================
# 3. MERGE ORDERS WITH MENU
# ============================================================

orders = orders.merge(
    menu,
    on="menu_item",
    how="left"
)


# ============================================================
# 4. CHECK THE MERGE
# ============================================================

print("========== MERGE RESULTS ==========\n")

print("Orders after merge:", orders.shape)

print("\nMissing menu information:")
print(orders[[
    "category",
    "price",
    "ingredient_cost",
    "prep_time_min"
]].isna().sum())


# ============================================================
# 5. CREATE REVENUE AND COST FEATURES
# ============================================================

orders["estimated_ingredient_cost"] = (
    orders["quantity"] * orders["ingredient_cost"]
)

orders["estimated_gross_profit"] = (
    orders["line_revenue"] - orders["estimated_ingredient_cost"]
)

orders["profit_margin"] = (
    orders["estimated_gross_profit"] / orders["line_revenue"]
)


# ============================================================
# 6. CREATE TIME FEATURES
# ============================================================

orders["timestamp"] = pd.to_datetime(orders["timestamp"])

orders["date"] = orders["timestamp"].dt.date
orders["hour"] = orders["timestamp"].dt.hour
orders["day_of_week"] = orders["timestamp"].dt.day_name()
orders["day_number"] = orders["timestamp"].dt.dayofweek
orders["is_weekend"] = orders["day_number"] >= 5


# ============================================================
# 7. CREATE MEAL-PERIOD FEATURE
# ============================================================

orders["meal_period"] = "Other"

orders.loc[
    orders["hour"].between(11, 15),
    "meal_period"
] = "Lunch"

orders.loc[
    orders["hour"].between(17, 21),
    "meal_period"
] = "Dinner"


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========== ENGINEERED DATA ==========\n")

print(orders.head())

print("\nNew columns:")
print([
    "estimated_ingredient_cost",
    "estimated_gross_profit",
    "profit_margin",
    "date",
    "hour",
    "day_of_week",
    "day_number",
    "is_weekend",
    "meal_period"
])


# ============================================================
# 9. BASIC SUMMARY
# ============================================================

print("\n========== BASIC SUMMARY ==========\n")

print("Total revenue:",
      round(orders["line_revenue"].sum(), 2))

print("Estimated ingredient cost:",
      round(orders["estimated_ingredient_cost"].sum(), 2))

print("Estimated gross profit:",
      round(orders["estimated_gross_profit"].sum(), 2))

print("Average profit margin:",
      round(orders["profit_margin"].mean() * 100, 2), "%")


# ============================================================
# 10. SAVE ENGINEERED DATA
# ============================================================

orders.to_csv(
    "data/processed/orders_engineered.csv",
    index=False
)

print("\n========== DATA SAVED ==========\n")
print("Saved to data/processed/orders_engineered.csv")