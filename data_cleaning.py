import pandas as pd 
def clean_data(df):
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
    
    return df