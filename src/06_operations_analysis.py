import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

orders = pd.read_csv("data/processed/orders_engineered.csv")
inventory = pd.read_csv("data/processed/inventory_clean.csv")
staffing = pd.read_csv("data/processed/staffing_clean.csv")


# Convert dates
orders["date"] = pd.to_datetime(orders["date"])
inventory["date"] = pd.to_datetime(inventory["date"])
staffing["date"] = pd.to_datetime(staffing["date"])


# 2. REVENUE CONSISTENCY CHECK
orders["expected_revenue"] = (
    orders["price"] * orders["quantity"]
)

orders["revenue_difference"] = (
    orders["line_revenue"] - orders["expected_revenue"]
)

revenue_issues = orders[
    orders["revenue_difference"].abs() > 0.01
]

print("========== REVENUE CONSISTENCY CHECK ==========\n")
print("Records checked:", len(orders))
print("Revenue inconsistencies:", len(revenue_issues))

if len(revenue_issues) > 0:
    print("\nLargest revenue discrepancies:")
    print(
        revenue_issues[
            [
                "order_id",
                "menu_item",
                "quantity",
                "price",
                "line_revenue",
                "expected_revenue",
                "revenue_difference"
            ]
        ]
        .sort_values(
            "revenue_difference",
            key=lambda x: x.abs(),
            ascending=False
        )
        .head(10)
    )


# ============================================================
# 2. INVENTORY WASTE ANALYSIS
# ============================================================

inventory["waste_cost"] = (
    inventory["units_wasted"] * inventory["unit_cost"]
)

inventory["used_cost"] = (
    inventory["units_used"] * inventory["unit_cost"]
)

inventory["received_cost"] = (
    inventory["units_received"] * inventory["unit_cost"]
)


ingredient_analysis = (
    inventory
    .groupby("ingredient")
    .agg(
        units_received=("units_received", "sum"),
        units_used=("units_used", "sum"),
        units_wasted=("units_wasted", "sum"),
        waste_cost=("waste_cost", "sum"),
        used_cost=("used_cost", "sum")
    )
)

ingredient_analysis["waste_rate"] = (
    ingredient_analysis["units_wasted"]
    / (
        ingredient_analysis["units_used"]
        + ingredient_analysis["units_wasted"]
    )
)

ingredient_analysis = ingredient_analysis.sort_values(
    "waste_cost",
    ascending=False
)


print("========== INVENTORY / WASTE ANALYSIS ==========\n")
print(ingredient_analysis)


# ============================================================
# 3. TOTAL WASTE
# ============================================================

total_waste_cost = inventory["waste_cost"].sum()

print("\n========== TOTAL WASTE ==========\n")

print(
    "Total estimated waste cost:",
    round(total_waste_cost, 2)
)

print(
    "Total units wasted:",
    round(inventory["units_wasted"].sum(), 2)
)


# ============================================================
# 4. STAFFING ANALYSIS
# ============================================================

staff_analysis = (
    staffing
    .groupby("role")
    .agg(
        employees=("employee", "nunique"),
        total_hours=("hours_worked", "sum"),
        average_hours=("hours_worked", "mean")
    )
    .sort_values("total_hours", ascending=False)
)

print("\n========== STAFFING BY ROLE ==========\n")
print(staff_analysis)


# ============================================================
# 5. DAILY LABOR
# ============================================================

daily_staffing = (
    staffing
    .groupby("date")["hours_worked"]
    .sum()
    .reset_index(name="labor_hours")
)

daily_revenue = (
    orders
    .groupby("date")["line_revenue"]
    .sum()
    .reset_index()
)

daily_operations = daily_revenue.merge(
    daily_staffing,
    on="date",
    how="left"
)

daily_operations["revenue_per_labor_hour"] = (
    daily_operations["line_revenue"]
    / daily_operations["labor_hours"]
)


print("\n========== DAILY LABOR EFFICIENCY ==========\n")

print(daily_operations.head())

print(
    "\nAverage revenue per labor hour:",
    round(
        daily_operations["revenue_per_labor_hour"].mean(),
        2
    )
)


# ============================================================
# 6. BEST / WORST LABOR DAYS
# ============================================================

print("\n========== TOP 5 DAYS BY REVENUE PER LABOR HOUR ==========\n")

print(
    daily_operations
    .sort_values(
        "revenue_per_labor_hour",
        ascending=False
    )
    .head(5)
)


print("\n========== BOTTOM 5 DAYS BY REVENUE PER LABOR HOUR ==========\n")

print(
    daily_operations
    .sort_values(
        "revenue_per_labor_hour"
    )
    .head(5)
)


# ============================================================
# 7. SAVE OPERATIONS TABLES
# ============================================================

ingredient_analysis.to_csv(
    "data/processed/ingredient_analysis.csv"
)

daily_operations.to_csv(
    "data/processed/daily_operations.csv",
    index=False
)

print("\n========== OPERATIONS ANALYSIS SAVED ==========\n")

print("Saved ingredient_analysis.csv")
print("Saved daily_operations.csv")