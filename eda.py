import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    # Scatter Plot: Rank Difference vs. Match Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='rank_diff', y='minutes', data=df, hue='surface', alpha=0.6)
    plt.title('Rank Difference vs. Match Duration by Surface')
    plt.xlabel('Rank Difference (Winner - Loser)')
    plt.ylabel('Match Duration (Minutes)')
    plt.show()

    # Box Plot: Aces by Surface
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='surface', y='w_ace', data=df)
    plt.title('Winner Aces by Surface')
    plt.xlabel('Surface')
    plt.ylabel('Aces by Winner')
    plt.xticks(ticks=[0], labels=['Hard'])  # Update labels based on encoding
    plt.show()

    # Bar Plot: Win Percentage by Player Hand
    hand_wins = df['winner_hand'].value_counts() / (df['winner_hand'].value_counts() + df['loser_hand'].value_counts())
    plt.figure(figsize=(8, 5))
    hand_wins.plot(kind='bar')
    plt.title('Win Percentage by Player Hand')
    plt.xlabel('Hand (0=Left, 1=Right)')  # Adjust based on encoding
    plt.ylabel('Win Proportion')
    plt.xticks(ticks=[0, 1], labels=['Left', 'Right'], rotation=0)
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