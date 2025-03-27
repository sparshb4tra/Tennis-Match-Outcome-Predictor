"""
*this file consists of every step starting from data loading to eda, i made several changes as well you may check them in their respective files, open to contribution.*

import pandas as pd

# Load the dataset
df = pd.read_csv('atp_matches_2024.csv')  # Replace with your file path if needed

# Quick look at the data
print(df.head())  # Displays first 5 rows
print(df.info())  # Shows column names, data types, and non-null counts



# Check for missing values
print(df.isnull().sum())

# Handle missing seeds and entries (fill with 'None' or drop if irrelevant)
df['winner_seed'] = df['winner_seed'].fillna('None')
df['loser_seed'] = df['loser_seed'].fillna('None')
df['winner_entry'] = df['winner_entry'].fillna('None')
df['loser_entry'] = df['loser_entry'].fillna('None')

# Drop rows with critical missing data (e.g., no winner or loser)
df = df.dropna(subset=['winner_name', 'loser_name', 'score'])

# Check for duplicates
duplicates = df.duplicated(subset=['tourney_id', 'match_num'])
print(f"Number of duplicates: {duplicates.sum()}")
df = df.drop_duplicates(subset=['tourney_id', 'match_num'])

# Fix data types
df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce')
df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce')
df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')

# Quick sanity check
print(df.describe())  # Stats like min/max to spot outliers (e.g., negative minutes)




from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le = LabelEncoder()
df['surface'] = le.fit_transform(df['surface'])
df['winner_hand'] = le.fit_transform(df['winner_hand'])
df['loser_hand'] = le.fit_transform(df['loser_hand'])
df['round'] = le.fit_transform(df['round'])

# Feature engineering
df['rank_diff'] = df['winner_rank'] - df['loser_rank']
df['ace_pct_winner'] = df['w_ace'] / df['w_svpt'].replace(0, 1)  # Avoid division by zero
df['ace_pct_loser'] = df['l_ace'] / df['l_svpt'].replace(0, 1)

# Scaling numerical features (optional, depends on model)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['winner_rank', 'loser_rank', 'minutes']] = scaler.fit_transform(df[['winner_rank', 'loser_rank', 'minutes']])






import matplotlib.pyplot as plt
import seaborn as sns

# Win rate by surface
surface_wins = df.groupby('surface')['winner_name'].count() / df['surface'].value_counts()
plt.bar(surface_wins.index, surface_wins.values)
plt.xlabel('Surface')
plt.ylabel('Win Proportion')
plt.title('Win Proportion by Surface')
plt.show()

# Rank difference distribution
sns.histplot(df['rank_diff'], bins=20)
plt.title('Distribution of Rank Difference (Winner - Loser)')
plt.xlabel('Rank Difference')
plt.show()

# Correlation heatmap for match stats
stats_cols = ['w_ace', 'w_df', 'w_1stWon', 'w_2ndWon', 'l_ace', 'l_df', 'l_1stWon', 'l_2ndWon']
corr = df[stats_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation of Match Stats')
plt.show()


    # Line Plot: Rank vs. Win Count (Top Players)
    rank_wins = df['winner_rank'].value_counts().sort_index().head(20)  # Top 20 ranks
    plt.figure(figsize=(10, 6))
    plt.plot(rank_wins.index, rank_wins.values, marker='o')
    plt.title('Wins by Player Rank (Top 20)')
    plt.xlabel('Winner Rank')
    plt.ylabel('Number of Wins')
    plt.grid(True)
    plt.show()

    # Violin Plot: Break Points Saved by Round
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='round', y='w_bpSaved', data=df)
    plt.title('Break Points Saved by Winner Across Rounds')
    plt.xlabel('Round')
    plt.ylabel('Break Points Saved')
    plt.xticks(rotation=45)
    plt.show()
"""
