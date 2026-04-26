import pandas as pd
import numpy as np
from datetime import datetime

# Load dataset
df = pd.read_csv("results.csv")

print("=" * 60)
print("FOOTBALL MATCH RESULTS - COMPREHENSIVE ANALYSIS")
print("=" * 60)

# Display basic info
print("\n--- DATASET OVERVIEW ---")
print(f"Total matches: {df.shape[0]}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Create useful extra columns
df["total_goals"] = df["home_score"] + df["away_score"]
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["goal_difference"] = df["home_score"] - df["away_score"]

# ============================================================
# DATA QUALITY ASSESSMENT & GAP IDENTIFICATION
# ============================================================
print("\n" + "=" * 60)
print("DATA QUALITY & GAP ANALYSIS")
print("=" * 60)

# 1. Check for missing values
print("\n--- MISSING VALUES ---")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
for col in df.columns:
    if missing_values[col] > 0:
        print(f"{col}: {missing_values[col]} ({missing_percent[col]:.2f}%)")

if missing_values.sum() == 0:
    print("✓ No missing values detected")

# 2. Check for duplicate matches
print("\n--- DUPLICATE CHECK ---")
duplicates = df.duplicated(subset=['date', 'home_team', 'away_team'], keep=False)
if duplicates.any():
    print(f"⚠ Found {duplicates.sum()} potential duplicate matches")
    print(df[duplicates].head())
else:
    print("✓ No duplicate matches found")

# 3. Validate scores
print("\n--- SCORE VALIDATION ---")
negative_scores = df[(df["home_score"] < 0) | (df["away_score"] < 0)]
if len(negative_scores) > 0:
    print(f"⚠ Found {len(negative_scores)} matches with negative scores")
else:
    print("✓ All scores are non-negative")

# Check for unusually high scores
high_scoring = df[df["total_goals"] > 20]
if len(high_scoring) > 0:
    print(f"⚠ Found {len(high_scoring)} matches with >20 total goals (potential outliers)")
    print(high_scoring[["date", "home_team", "away_team", "total_goals"]].head())

# 4. Date range and temporal gaps
print("\n--- TEMPORAL ANALYSIS ---")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Total years covered: {(df['date'].max() - df['date'].min()).days / 365.25:.1f} years")

# Find temporal gaps > 1 year
df_sorted = df.sort_values("date")
date_diffs = df_sorted["date"].diff()
large_gaps = date_diffs[date_diffs > pd.Timedelta(days=365)]
if len(large_gaps) > 0:
    print(f"\n⚠ Found {len(large_gaps)} temporal gaps > 1 year:")
    for idx in large_gaps.index[:5]:  # Show first 5 gaps
        prev_idx = df_sorted.index.get_loc(idx) - 1
        if prev_idx >= 0:
            prev_date = df_sorted.iloc[prev_idx]["date"]
            curr_date = df_sorted.loc[idx, "date"]
            gap_days = (curr_date - prev_date).days
            print(f"  - {gap_days} days between {prev_date.strftime('%Y-%m-%d')} and {curr_date.strftime('%Y-%m-%d')}")

# 5. Check for future dates
future_matches = df[df["date"] > datetime.now()]
if len(future_matches) > 0:
    print(f"\n⚠ Found {len(future_matches)} matches with future dates")
else:
    print("✓ No future dates detected")

# 6. Team and location consistency
print("\n--- CONSISTENCY CHECKS ---")
# Check if home_team always matches country
team_country_mismatch = df[df["home_team"] != df["country"]]
print(f"Matches where home_team != country: {len(team_country_mismatch)} (this is normal for tournaments)")

# Check neutral flag consistency
neutral_matches = df[df["neutral"] == "TRUE"]
non_neutral_matches = df[df["neutral"] == "FALSE"]
print(f"Neutral venue matches: {len(neutral_matches)} ({len(neutral_matches)/len(df)*100:.1f}%)")
print(f"Non-neutral venue matches: {len(non_neutral_matches)} ({len(non_neutral_matches)/len(df)*100:.1f}%)")

# ============================================================
# GAP FILLING & DATA ENRICHMENT
# ============================================================
print("\n" + "=" * 60)
print("GAP FILLING & DATA ENRICHMENT")
print("=" * 60)

# Gap 1: Add match outcome column
print("\n✓ Adding match outcome column...")
df["outcome"] = np.where(
    df["home_score"] > df["away_score"], "home_win",
    np.where(df["away_score"] > df["home_score"], "away_win", "draw")
)

# Gap 2: Add decade column for temporal analysis
print("✓ Adding decade column...")
df["decade"] = (df["year"] // 10) * 10

# Gap 3: Add goal bins for categorization
print("✓ Adding goal category column...")
df["goal_category"] = pd.cut(
    df["total_goals"],
    bins=[-1, 0, 2, 4, 6, 100],
    labels=["0_goals", "1-2_goals", "3-4_goals", "5-6_goals", "7+_goals"]
)

# Gap 4: Calculate rolling statistics for teams
print("✓ Calculating team performance statistics...")

# Home performance
home_stats = df.groupby("home_team").agg(
    home_matches=("home_team", "count"),
    home_wins=("outcome", lambda x: (x == "home_win").sum()),
    home_draws=("outcome", lambda x: (x == "draw").sum()),
    home_losses=("outcome", lambda x: (x == "away_win").sum()),
    home_goals_scored=("home_score", "sum"),
    home_goals_conceded=("away_score", "sum")
).reset_index()

home_stats["home_win_rate"] = home_stats["home_wins"] / home_stats["home_matches"]

# Away performance
away_stats = df.groupby("away_team").agg(
    away_matches=("away_team", "count"),
    away_wins=("outcome", lambda x: (x == "away_win").sum()),
    away_draws=("outcome", lambda x: (x == "draw").sum()),
    away_losses=("outcome", lambda x: (x == "home_win").sum()),
    away_goals_scored=("away_score", "sum"),
    away_goals_conceded=("home_score", "sum")
).reset_index()

away_stats["away_win_rate"] = away_stats["away_wins"] / away_stats["away_matches"]

# Gap 5: Identify teams with incomplete data
print("\n--- TEAM DATA COMPLETENESS ---")
all_teams = pd.concat([df["home_team"], df["away_team"]])
team_counts = all_teams.value_counts()
rare_teams = team_counts[team_counts < 5]
print(f"Teams with < 5 matches: {len(rare_teams)}")
print(f"Most active teams: {team_counts.head(10)}")

# ============================================================
# COMPREHENSIVE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("COMPREHENSIVE ANALYSIS")
print("=" * 60)

# 1. Total matches
print(f"\n1. Total matches in dataset: {df.shape[0]}")

# 2. Date range
print(f"\n2. Date Range:")
print(f"   Earliest year: {df['year'].min()}")
print(f"   Latest year: {df['year'].max()}")
print(f"   Decades covered: {df['decade'].nunique()}")

# 3. Geographic distribution
print(f"\n3. Geographic Distribution:")
print(f"   Unique countries: {df['country'].nunique()}")
print(f"   Top 5 countries by matches:")
for country, count in df["country"].value_counts().head(5).items():
    print(f"   - {country}: {count} matches")

# 4. Most frequent teams
print(f"\n4. Most Frequent Teams:")
print(f"   Most frequent home team: {df['home_team'].value_counts().idxmax()}")
print(f"   Most frequent away team: {df['away_team'].value_counts().idxmax()}")

# 5. Goal statistics
print(f"\n5. Goal Statistics:")
print(f"   Average goals per match: {df['total_goals'].mean():.2f}")
print(f"   Average home goals: {df['home_score'].mean():.2f}")
print(f"   Average away goals: {df['away_score'].mean():.2f}")
print(f"   Most common total goals: {df['total_goals'].mode()[0]}")

# 6. Highest scoring matches
print(f"\n6. Highest Scoring Matches:")
max_goals = df["total_goals"].max()
highest_matches = df[df["total_goals"] == max_goals]
for _, match in highest_matches.head(3).iterrows():
    print(f"   {match['date'].strftime('%Y-%m-%d')}: {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']} ({match['tournament']})")

# 7. Home vs Away performance
print(f"\n7. Home vs Away Performance:")
home_goals = df["home_score"].sum()
away_goals = df["away_score"].sum()
print(f"   Total home goals: {home_goals}")
print(f"   Total away goals: {away_goals}")
print(f"   Home goals per match: {home_goals/len(df):.2f}")
print(f"   Away goals per match: {away_goals/len(df):.2f}")

# 8. Match outcomes
print(f"\n8. Match Outcomes:")
home_wins = (df["outcome"] == "home_win").sum()
away_wins = (df["outcome"] == "away_win").sum()
draws = (df["outcome"] == "draw").sum()
print(f"   Home wins: {home_wins} ({home_wins/len(df)*100:.1f}%)")
print(f"   Away wins: {away_wins} ({away_wins/len(df)*100:.1f}%)")
print(f"   Draws: {draws} ({draws/len(df)*100:.1f}%)")

# 9. Home advantage analysis
print(f"\n9. Home Advantage Analysis:")
if home_wins > away_wins:
    advantage = ((home_wins - away_wins) / len(df)) * 100
    print(f"   ✓ YES - Home advantage exists (+{advantage:.1f}% home win advantage)")
else:
    print(f"   ✗ NO strong home advantage detected")

# 10. Tournament analysis
print(f"\n10. Tournament Distribution:")
print(f"    Unique tournaments: {df['tournament'].nunique()}")
print(f"    Top 5 tournaments:")
for tournament, count in df["tournament"].value_counts().head(5).items():
    avg_goals = df[df["tournament"] == tournament]["total_goals"].mean()
    print(f"    - {tournament}: {count} matches (avg {avg_goals:.2f} goals)")

# 11. Top performing teams
print(f"\n11. Top Performing Teams (by total wins):")
home_win_teams = df[df["outcome"] == "home_win"]["home_team"]
away_win_teams = df[df["outcome"] == "away_win"]["away_team"]
all_winners = pd.concat([home_win_teams, away_win_teams])
top_teams = all_winners.value_counts().head(10)
for rank, (team, wins) in enumerate(top_teams.items(), 1):
    print(f"    {rank}. {team}: {wins} wins")

# 12. Decade-wise trends
print(f"\n12. Decade-wise Trends:")
decade_stats = df.groupby("decade").agg(
    matches=("date", "count"),
    avg_goals=("total_goals", "mean"),
    home_win_pct=("outcome", lambda x: (x == "home_win").sum() / len(x) * 100)
).round(2)
print(decade_stats.tail(10))

# ============================================================
# SAVE ENRICHED DATASET
# ============================================================
print("\n" + "=" * 60)
print("SAVING ENRICHED DATASET")
print("=" * 60)

output_file = "results_enriched.csv"
df.to_csv(output_file, index=False)
print(f"✓ Enriched dataset saved to {output_file}")
print(f"  New columns added: outcome, decade, goal_category, goal_difference")
print(f"  Total columns: {len(df.columns)}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
