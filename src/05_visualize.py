import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD ENGINEERED DATA
# ============================================================

orders = pd.read_csv("data/processed/orders_engineered.csv")


# ============================================================
# 2. TOP MENU ITEMS BY UNITS SOLD
# ============================================================

menu_demand = (
    orders
    .groupby("menu_item")["quantity"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

menu_demand.plot(kind="bar")

plt.title("Menu Items by Units Sold")
plt.xlabel("Menu Item")
plt.ylabel("Units Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/figures/menu_demand.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. REVENUE BY DAY OF WEEK
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

daily_revenue = (
    orders
    .groupby("day_of_week")["line_revenue"]
    .sum()
    .reindex(day_order)
)

plt.figure(figsize=(10, 6))

daily_revenue.plot(kind="bar")

plt.title("Revenue by Day of Week")
plt.xlabel("Day")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/figures/revenue_by_day.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. ORDERS BY HOUR
# ============================================================

hourly_orders = (
    orders
    .groupby("hour")["order_id"]
    .count()
)

plt.figure(figsize=(10, 6))

hourly_orders.plot(kind="line", marker="o")

plt.title("Order Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.xticks(hourly_orders.index)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/orders_by_hour.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. REVENUE BY ORDER TYPE
# ============================================================

order_type_revenue = (
    orders
    .groupby("order_type")["line_revenue"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 6))

order_type_revenue.plot(kind="bar")

plt.title("Revenue by Order Type")
plt.xlabel("Order Type")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "outputs/figures/revenue_by_order_type.png",
    dpi=300
)

plt.show()


print("\n========== VISUALIZATIONS COMPLETE ==========\n")
print("Saved figures to outputs/figures/")