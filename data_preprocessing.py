from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
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
    scaler = StandardScaler()
    df[['winner_rank', 'loser_rank', 'minutes']] = scaler.fit_transform(df[['winner_rank', 'loser_rank', 'minutes']])

    # Check the transformed data
    print(df[['rank_diff', 'ace_pct_winner', 'ace_pct_loser']].head())
    
    return df