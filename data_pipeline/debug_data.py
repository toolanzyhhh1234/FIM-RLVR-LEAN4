import pandas as pd


def main():
    parquet_path = "data/NuminaMath-LEAN/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path)

    # inspect algebra_871 or just a random row
    row = df[df["formal_statement"].str.contains("algebra_871")].iloc[0]

    print("KEYS:", row.keys())
    print("STMT:", row["formal_statement"])
    print("PROOF:", row["formal_proof"])
    print("ANSWER:", row["answer"])


if __name__ == "__main__":
    main()
