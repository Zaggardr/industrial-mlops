import polars as pl

DATA_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/raw/housing.csv"

df = pl.read_csv(DATA_PATH)

print("=== DATA QUALITY REPORT ===")

# 1. Basic information
print("\n--- Shape ---")
print(df.shape)

# 2. Summary statistics
print("\n--- Statistics ---")
print(df.describe())

# 3. Unique values for categorical columns
print("\n--- Ocean Proximity ---")
print(
    df
    .group_by("ocean_proximity")
    .len()
    .sort("len", descending=True)
)

# 4. Check duplicated rows
print("\n--- Duplicates ---")
print("Duplicate rows:", df.is_duplicated().sum())

# 5. Check target distribution
print("\n--- Target statistics ---")
print(
    df.select(
        [
            pl.col("median_house_value").min().alias("min"),
            pl.col("median_house_value").max().alias("max"),
            pl.col("median_house_value").mean().alias("mean"),
            pl.col("median_house_value").median().alias("median"),
        ]
    )
)

# 6. Check impossible / suspicious values
print("\n--- Suspicious values ---")

print(
    "Median income <= 0:",
    df.filter(pl.col("median_income") <= 0).height
)

print(
    "Total rooms <= 0:",
    df.filter(pl.col("total_rooms") <= 0).height
)

print(
    "Population <= 0:",
    df.filter(pl.col("population") <= 0).height
)