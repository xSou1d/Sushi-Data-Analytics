import pandas as pd


# ============================================================
# 1. LOAD ENGINEERED DATA
# ============================================================

orders = pd.read_csv("data/processed/orders_engineered.csv")


# ============================================================
# 2. MENU ITEM ANALYSIS
# ============================================================

menu_analysis = (
    orders
    .groupby("menu_item")
    .agg(
        units_sold=("quantity", "sum"),
        orders=("order_id", "count"),
        revenue=("line_revenue", "sum"),
        estimated_profit=("estimated_gross_profit", "sum")
    )
    .sort_values("units_sold", ascending=False)
)

print("========== MENU ITEM DEMAND ==========\n")
print(menu_analysis)


# ============================================================
# 3. DAY-OF-WEEK ANALYSIS
# ============================================================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_analysis = (
    orders
    .groupby("day_of_week")
    .agg(
        orders=("order_id", "count"),
        units_sold=("quantity", "sum"),
        revenue=("line_revenue", "sum"),
        estimated_profit=("estimated_gross_profit", "sum")
    )
    .reindex(day_order)
)

print("\n========== DAY-OF-WEEK ANALYSIS ==========\n")
print(day_analysis)


# ============================================================
# 4. HOURLY DEMAND ANALYSIS
# ============================================================

hour_analysis = (
    orders
    .groupby("hour")
    .agg(
        orders=("order_id", "count"),
        units_sold=("quantity", "sum"),
        revenue=("line_revenue", "sum")
    )
    .sort_index()
)

print("\n========== HOURLY DEMAND ==========\n")
print(hour_analysis)


# ============================================================
# 5. MEAL-PERIOD ANALYSIS
# ============================================================

meal_analysis = (
    orders
    .groupby("meal_period")
    .agg(
        orders=("order_id", "count"),
        units_sold=("quantity", "sum"),
        revenue=("line_revenue", "sum"),
        estimated_profit=("estimated_gross_profit", "sum")
    )
    .sort_values("orders", ascending=False)
)

print("\n========== MEAL-PERIOD ANALYSIS ==========\n")
print(meal_analysis)


# ============================================================
# 6. ORDER-TYPE ANALYSIS
# ============================================================

order_type_analysis = (
    orders
    .groupby("order_type")
    .agg(
        orders=("order_id", "count"),
        units_sold=("quantity", "sum"),
        revenue=("line_revenue", "sum"),
        estimated_profit=("estimated_gross_profit", "sum")
    )
    .sort_values("orders", ascending=False)
)

print("\n========== ORDER-TYPE ANALYSIS ==========\n")
print(order_type_analysis)


# ============================================================
# 7. TOP ITEMS
# ============================================================

print("\n========== TOP 5 ITEMS BY UNITS SOLD ==========\n")
print(menu_analysis.head(5))

print("\n========== TOP 5 ITEMS BY REVENUE ==========\n")
print(
    menu_analysis
    .sort_values("revenue", ascending=False)
    .head(5)
)

print("\n========== TOP 5 ITEMS BY ESTIMATED PROFIT ==========\n")
print(
    menu_analysis
    .sort_values("estimated_profit", ascending=False)
    .head(5)
)