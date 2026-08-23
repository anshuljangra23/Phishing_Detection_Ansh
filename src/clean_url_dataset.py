import csv
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/raw/urls/urlset.csv")
OUTPUT_FILE = Path("data/processed/clean_urls.csv")

EXPECTED_COLUMNS = [
    "domain",
    "ranking",
    "mld_res",
    "mld.ps_res",
    "card_rem",
    "ratio_Rrem",
    "ratio_Arem",
    "jaccard_RR",
    "jaccard_RA",
    "jaccard_AR",
    "jaccard_AA",
    "jaccard_ARrd",
    "jaccard_ARrem",
    "label",
]


def find_malformed_rows():
    bad_rows = []

    with open(INPUT_FILE, "r", encoding="latin1", newline="") as file:
        reader = csv.reader(file)

        header = next(reader)

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                bad_rows.append(line_number)

    return header, bad_rows


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Checking dataset...")

    header, bad_rows = find_malformed_rows()

    print(f"Expected columns: {len(header)}")
    print(f"Malformed rows: {len(bad_rows)}")

    # Read only valid rows.
    valid_rows = []

    with open(INPUT_FILE, "r", encoding="latin1", newline="") as file:
        reader = csv.reader(file)

        next(reader)

        for row in reader:
            if len(row) == len(header):
                valid_rows.append(row)

    df = pd.DataFrame(valid_rows, columns=header)

    print(f"Rows before cleaning: {len(df)}")

    # Remove completely empty rows.
    df = df.dropna(how="all")

    # Remove duplicate records.
    before_duplicates = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_duplicates - len(df)

    # Convert label to numeric.
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Remove rows with invalid labels.
    before_labels = len(df)
    df = df[df["label"].isin([0, 1])]
    invalid_labels_removed = before_labels - len(df)

    # Convert numeric feature columns.
    feature_columns = [
        column for column in EXPECTED_COLUMNS
        if column != "domain" and column != "label"
    ]

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows where required values are missing.
    before_missing = len(df)
    df = df.dropna()
    missing_removed = before_missing - len(df)

    # Save cleaned dataset.
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaning completed.")
    print(f"Malformed rows removed: {len(bad_rows)}")
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Invalid-label rows removed: {invalid_labels_removed}")
    print(f"Missing-value rows removed: {missing_removed}")
    print(f"Final dataset size: {len(df)}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
    