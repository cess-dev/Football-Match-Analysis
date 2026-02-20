import pandas as pd

# Load dataset
df = pd.read_csv("results.csv")

df = pd.read_csv("results.csv")
print(df.head())

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Create useful extra columns
df["total_goals"] = df["home_score"] + df["away_score"]
df["year"] = df["date"].dt.year


# 1. How many matches are in the dataset?
print("1. Total matches:", df.shape[0])


# 2. Earliest and latest year
print("2. Earliest year:", df["year"].min())
print("   Latest year:", df["year"].max())


# 3. Number of unique countries
print("3. Unique countries:", df["country"].nunique())


# 4. Most frequent home team
print("4. Most frequent home team:")
print(df["home_team"].value_counts().idxmax())


# 5. Average goals per match
print("5. Average goals per match:", df["total_goals"].mean())


# 6. Highest scoring match
max_goals = df["total_goals"].max()
highest_match = df[df["total_goals"] == max_goals]
print("6. Highest scoring match:")
print(highest_match[["date", "home_team", "away_team", "home_score", "away_score"]])


# 7. More goals at home or away?
home_goals = df["home_score"].sum()
away_goals = df["away_score"].sum()

print("7. Total home goals:", home_goals)
print("   Total away goals:", away_goals)

if home_goals > away_goals:
    print("   More goals scored at home.")
else:
    print("   More goals scored away.")


# 8. Most common total goals value
print("8. Most common total goals value:")
print(df["total_goals"].mode()[0])


# 9. Percentage of home wins
home_wins = df[df["home_score"] > df["away_score"]].shape[0]
percentage_home_wins = (home_wins / df.shape[0]) * 100

print("9. Percentage of home wins:", percentage_home_wins)


# 10. Does home advantage exist?
away_wins = df[df["away_score"] > df["home_score"]].shape[0]

if home_wins > away_wins:
    print("10. Yes, home advantage exists.")
else:
    print("10. No strong home advantage detected.")


# 11. Country with most historical wins
home_win_teams = df[df["home_score"] > df["away_score"]]["home_team"]
away_win_teams = df[df["away_score"] > df["home_score"]]["away_team"]

all_winners = pd.concat([home_win_teams, away_win_teams])

print("11. Country with most wins:")
print(all_winners.value_counts().idxmax())
