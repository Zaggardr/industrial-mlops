import polars as pl

DATA_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/processed/housing_features.parquet"

TRAIN_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/processed/train.parquet"
VAL_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/processed/validation.parquet"
TEST_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/processed/test.parquet"


def main():

    # Load engineered dataset
    df = pl.read_parquet(DATA_PATH)

    print("Total dataset:", df.shape)

    # Shuffle deterministically
    df = df.sample(
        fraction=1.0,
        shuffle=True,
        seed=42
    )

    # Calculate split sizes
    n = df.height

    train_size = int(n * 0.70)
    val_size = int(n * 0.15)

    # Split
    train_df = df[:train_size]

    val_df = df[
        train_size:
        train_size + val_size
    ]

    test_df = df[
        train_size + val_size:
    ]

    # Save
    train_df.write_parquet(TRAIN_PATH)
    val_df.write_parquet(VAL_PATH)
    test_df.write_parquet(TEST_PATH)

    # Report
    print("\n=== SPLIT RESULTS ===")

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    print("\nSaved:")
    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()