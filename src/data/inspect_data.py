import polars as pl

DATA_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/raw/housing.csv"

df = pl.read_csv(DATA_PATH)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.null_count())