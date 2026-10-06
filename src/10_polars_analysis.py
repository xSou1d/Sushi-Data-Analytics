import polars as pl


# ============================================================
# 1. LOAD DATA WITH POLARS
# ============================================================

orders = pl.read_csv(
    "data/processed/orders_clean.csv"
)

menu = pl.read_csv(
    "data/processed/menu_clean.csv"
)


# ============================================================
# 2. NORMALIZE MENU ITEM NAMES
# ============================================================

orders = orders.with_columns(
    pl.col("menu_item")
    .str.strip_chars()
    .str.to_titlecase()
)

menu = menu.with_columns(
    pl.col("menu_item")
    .str.strip_chars()
    .str.to_titlecase()
)


# ============================================================
# 3. JOIN ORDERS WITH MENU
# ============================================================

orders = orders.join(
    menu,
    on="menu_item",
    how="left"
)


# ============================================================
# 4. CALCULATE PROFIT FEATURES
# ============================================================

orders = orders.with_columns(
    (
        pl.col("quantity")
        * pl.col("ingredient_cost")
    ).alias("estimated_ingredient_cost")
)

orders = orders.with_columns(
    (
        pl.col("line_revenue")
        - pl.col("estimated_ingredient_cost")
    ).alias("estimated_gross_profit")
)


# ============================================================
# 5. GROUP AND SUM
# ============================================================

menu_analysis = (
    orders
    .group_by("menu_item")
    .agg(
        pl.col("quantity").sum().alias("units_sold"),
        pl.col("line_revenue").sum().alias("revenue"),
        pl.col("estimated_gross_profit")
        .sum()
        .alias("estimated_profit")
    )
    .sort(
        "units_sold",
        descending=True
    )
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("========== POLARS MENU ANALYSIS ==========\n")

print(menu_analysis)

print("\n========== TOP 5 ITEMS ==========\n")

print(
    menu_analysis.head(5)
)


# ============================================================
# 7. SAVE RESULT
# ============================================================

menu_analysis.write_csv(
    "data/processed/polars_menu_analysis.csv"
)

print("\n========== POLARS ANALYSIS SAVED ==========\n")
print(
    "Saved polars_menu_analysis.csv"
)