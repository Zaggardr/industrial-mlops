import polars as pl

DATA_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/raw/housing.csv"
OUTPUT_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/processed/housing_features.parquet"


def create_features(df: pl.DataFrame) -> pl.DataFrame:

    df = df.with_columns(
        [
            (
                pl.col("total_rooms") /
                pl.col("households")
            ).alias("rooms_per_household"),

            (
                pl.col("total_bedrooms") /
                pl.col("total_rooms")
            ).alias("bedrooms_per_room"),

            (
                pl.col("population") /
                pl.col("households")
            ).alias("population_per_household"),
        ]
    )

    return df


def main():

    # Load
    df = pl.read_csv(DATA_PATH)

    print("Original shape:", df.shape)

    # Feature engineering
    df = create_features(df)

    print("New shape:", df.shape)

    print("\nNew columns:")
    print(df.columns)

    print("\nFeature preview:")
    print(
        df.select(
            [
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "rooms_per_household",
                "bedrooms_per_room",
                "population_per_household",
            ]
        ).head()
    )

    # Create output directory
    import os
    os.makedirs("data/processed", exist_ok=True)

    # Save as Parquet
    df.write_parquet(OUTPUT_PATH)

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()