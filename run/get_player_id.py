import pandas as pd
import sys
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/projection_all_metrics.csv")


def get_player_id(name: str) -> None:
    df = pd.read_csv(DATA_FILE, usecols=["ID", "Name"])
    matches = df[df["Name"].str.contains(name, case=False, na=False)]

    if matches.empty:
        print(f"No player found matching '{name}'")
    elif len(matches) == 1:
        row = matches.iloc[0]
        print(f"{row['Name']}: {int(row['ID'])}")
    else:
        print(f"Multiple matches for '{name}':")
        for _, row in matches.iterrows():
            print(f"  {row['Name']}: {int(row['ID'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_player_id.py <player_name>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    get_player_id(query)
