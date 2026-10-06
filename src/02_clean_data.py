import pandas as pd

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

orders = pd.read_csv("data/raw/orders.csv")
inventory = pd.read_csv("data/raw/inventory.csv")
menu = pd.read_csv("data/raw/menu.csv")
staffing = pd.read_csv("data/raw/staffing.csv")


# ============================================================
# 2. REMOVE EXACT DUPLICATES
# ============================================================

orders = orders.drop_duplicates().copy()


# ============================================================
# 3. CONVERT DATE/TIME COLUMNS
# ============================================================

orders["timestamp"] = pd.to_datetime(orders["timestamp"])
inventory["date"] = pd.to_datetime(inventory["date"])
staffing["date"] = pd.to_datetime(staffing["date"])


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

# Missing server values are labeled as "Unknown"
orders["server"] = orders["server"].fillna("Unknown")

# Missing inventory waste is treated as zero recorded waste
inventory["units_wasted"] = inventory["units_wasted"].fillna(0)

# Missing staffing hours are filled with the median for that dataset
staffing["hours_worked"] = staffing["hours_worked"].fillna(
    staffing["hours_worked"].median()
)


# ============================================================
# 5. CHECK FOR INVALID VALUES
# ============================================================

print("========== VALIDATION ==========")

print("Orders with non-positive quantities:",
      (orders["quantity"] <= 0).sum())

print("Orders with non-positive revenue:",
      (orders["line_revenue"] <= 0).sum())

print("Inventory with negative received:",
      (inventory["units_received"] < 0).sum())

print("Inventory with negative used:",
      (inventory["units_used"] < 0).sum())

print("Inventory with negative wasted:",
      (inventory["units_wasted"] < 0).sum())

print("Staffing with non-positive hours:",
      (staffing["hours_worked"] <= 0).sum())


# ============================================================
# 6. BASIC DATASET VALIDATION
# ============================================================

print("\n========== FINAL DATASET SIZES ==========")

print("Orders:", orders.shape)
print("Inventory:", inventory.shape)
print("Menu:", menu.shape)
print("Staffing:", staffing.shape)


print("\nMissing values after cleaning:")

print("\nOrders:")
print(orders.isna().sum())

print("\nInventory:")
print(inventory.isna().sum())

print("\nMenu:")
print(menu.isna().sum())

print("\nStaffing:")
print(staffing.isna().sum())


# ============================================================
# 7. INVESTIGATE INVALID RECORDS
# ============================================================

print("\n========== INVALID ORDER QUANTITIES ==========\n")

invalid_orders = orders[orders["quantity"] <= 0]

print(invalid_orders)


print("\n========== INVALID INVENTORY USAGE ==========\n")

invalid_inventory = inventory[inventory["units_used"] < 0]

print(invalid_inventory)


# ============================================================
# 8. REMOVE INVALID RECORDS
# ============================================================

orders = orders[orders["quantity"] > 0].copy()

inventory = inventory[inventory["units_used"] >= 0].copy()

print("\n========== AFTER INVALID RECORD REMOVAL ==========\n")

print("Orders:", orders.shape)
print("Inventory:", inventory.shape)

print("\nRemaining invalid order quantities:",
      (orders["quantity"] <= 0).sum())

print("Remaining negative inventory usage:",
      (inventory["units_used"] < 0).sum())


# ============================================================
# 9. SAVE CLEANED DATA
# ============================================================

orders.to_csv("data/processed/orders_clean.csv", index=False)
inventory.to_csv("data/processed/inventory_clean.csv", index=False)
menu.to_csv("data/processed/menu_clean.csv", index=False)
staffing.to_csv("data/processed/staffing_clean.csv", index=False)

print("\n========== CLEANED DATA SAVED ==========\n")
print("Saved cleaned datasets to data/processed/")