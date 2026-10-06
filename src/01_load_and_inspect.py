import pandas as pd

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

orders = pd.read_csv("data/raw/orders.csv")
inventory = pd.read_csv("data/raw/inventory.csv")
menu = pd.read_csv("data/raw/menu.csv")
staffing = pd.read_csv("data/raw/staffing.csv")


# ============================================================
# 2. BASIC DATASET STRUCTURE
# ============================================================

print("========== DATASET STRUCTURE ==========\n")

print(f"Orders has {orders.shape[0]} rows and {orders.shape[1]} columns")
print(f"Columns: {list(orders.columns)}")
print(orders.dtypes)
print()

print(f"Inventory has {inventory.shape[0]} rows and {inventory.shape[1]} columns")
print(f"Columns: {list(inventory.columns)}")
print(inventory.dtypes)
print()

print(f"Menu has {menu.shape[0]} rows and {menu.shape[1]} columns")
print(f"Columns: {list(menu.columns)}")
print(menu.dtypes)
print()

print(f"Staffing has {staffing.shape[0]} rows and {staffing.shape[1]} columns")
print(f"Columns: {list(staffing.columns)}")
print(staffing.dtypes)
print()


# ============================================================
# 3. SAMPLE RECORDS
# ============================================================

print("========== SAMPLE RECORDS ==========\n")

print("First 3 orders:")
print(orders.head(3))
print("\nLast 3 orders:")
print(orders.tail(3))
print()

print("First 3 inventory records:")
print(inventory.head(3))
print("\nLast 3 inventory records:")
print(inventory.tail(3))
print()

print("First 3 menu records:")
print(menu.head(3))
print("\nLast 3 menu records:")
print(menu.tail(3))
print()

print("First 3 staffing records:")
print(staffing.head(3))
print("\nLast 3 staffing records:")
print(staffing.tail(3))
print()


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("========== MISSING VALUES ==========\n")

print("Orders:")
print(orders.isna().sum())
print()

print("Inventory:")
print(inventory.isna().sum())
print()

print("Menu:")
print(menu.isna().sum())
print()

print("Staffing:")
print(staffing.isna().sum())
print()


# ============================================================
# 5. DUPLICATE ORDERS
# ============================================================

print("========== DUPLICATE ORDERS ==========\n")

print("Duplicate rows in orders:", orders.duplicated().sum())

duplicates = orders[orders.duplicated(keep=False)]

print("\nExample duplicate records:")
print(duplicates.sort_values("order_id").head(20))

print("\nUnique order IDs:", orders["order_id"].nunique())
print("Total order rows:", len(orders))
print()


# ============================================================
# 6. CREATE CLEAN ORDERS DATASET
# ============================================================

orders_clean = orders.drop_duplicates().copy()

print("========== CLEANED ORDERS ==========\n")

print("Original rows:", len(orders))
print("Cleaned rows:", len(orders_clean))
print("Rows removed:", len(orders) - len(orders_clean))
print()


# ============================================================
# 7. CHECK FOR INVALID ORDER VALUES
# ============================================================

print("========== ORDER VALUE CHECKS ==========\n")

print("Quantity statistics:")
print(orders_clean["quantity"].describe())
print()

print("Revenue statistics:")
print(orders_clean["line_revenue"].describe())
print()

print("Non-positive quantities:")
print(orders_clean[orders_clean["quantity"] <= 0])
print()

print("Non-positive revenue:")
print(orders_clean[orders_clean["line_revenue"] <= 0])