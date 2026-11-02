
import kagglehub
import pandas as pd
import csv
from pathlib import Path

# 1. Download only if the dataset is not already present
data_path = Path(__file__).resolve().parent / "data"
data_path.mkdir(exist_ok=True)

file = data_path / "games.csv"

if not file.exists():
    kagglehub.dataset_download(
        "fronkongames/steam-games-dataset",
        path="games.csv",
        output_dir=str(data_path)
    )

# 2. Fix the known CSV header issue
with open(file, encoding="utf-8-sig", newline="") as f:
    columns = next(csv.reader(f))

if "DiscountDLC count" in columns:
    i = columns.index("DiscountDLC count")
    columns[i:i+1] = ["Discount", "DLC count"]

# 3. Read only the 10 columns we need
keep = [
    "AppID",
    "Name",
    "Genres",
    "Tags",
    "Price",
    "Average playtime forever",
    "Median playtime forever",
    "Positive",
    "Negative",
    "Estimated owners"
]

games = pd.read_csv(
    file,
    header=0,
    names=columns,
    usecols=keep,
    low_memory=False
)

print("Original shape:", games.shape)

# 4. Rename columns
games = games.rename(columns={
    "AppID": "app_id",
    "Name": "name",
    "Genres": "genres",
    "Tags": "tags",
    "Price": "price",
    "Average playtime forever": "avg_hours",
    "Median playtime forever": "median_hours",
    "Positive": "positive",
    "Negative": "negative",
    "Estimated owners": "estimated_owners"
})

# 5. Clean data
def report(step, before, after):
    print(f"  {step:<45} {before:>7} -> {after:>7}  (removed {before - after})")

print("\nCleaning steps:")

n = len(games)
games = games.drop_duplicates(subset="app_id")
report("drop duplicate app_id", n, len(games))

n = len(games)
games = games.dropna(subset=["name", "genres", "price", "avg_hours"])
report("drop missing name/genres/price/playtime", n, len(games))

n = len(games)
games = games[games["price"] > 0].copy()
report("price > 0 (excludes free-to-play)", n, len(games))

n = len(games)
games = games[games["avg_hours"] > 0].copy()
report("avg playtime > 0 (has tracked playtime)", n, len(games))

# Keep only real game genres; drop software apps and non-genre labels
GAME_GENRES = {
    "Action", "Adventure", "Casual", "Indie", "Massively Multiplayer",
    "RPG", "Racing", "Simulation", "Sports", "Strategy",
}

def clean_genres(text):
    genres = [g.strip() for g in str(text).split(",")]
    return ",".join(g for g in genres if g in GAME_GENRES)

games["genres"] = games["genres"].apply(clean_genres)

n = len(games)
games = games[games["genres"] != ""].copy()
report("has at least one game genre (drops software)", n, len(games))

games["avg_hours"] = games["avg_hours"] / 60
games["median_hours"] = games["median_hours"] / 60

# 6. Save the cleaned dataset
games.to_csv(
    "data/steam_games_clean.csv",
    index=False,
    header=True
)

print("Cleaned shape:", games.shape)
print(games.head(10).to_string(index=False))


print(games[["price", "avg_hours", "median_hours"]].describe())

print("\nGames with no reviews:")
print(((games["positive"] + games["negative"]) == 0).sum())

print("\nGames without tags:")
print((games["tags"].str.strip() == "").sum())

print("\nTop 10 games by average playtime:")
print(
    games.nlargest(10, "avg_hours")[
        ["name", "price", "avg_hours", "positive", "negative"]
    ].to_string(index=False)
)


