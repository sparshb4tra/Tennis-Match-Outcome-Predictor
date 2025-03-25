import pandas as pd

def load_data(file_path = 'D:/downloads/apps/aiml lab/atp_matches_2024.csv'):
    # Load the dataset
    df = pd.read_csv('atp_matches_2024.csv')  # Replace with your file path if needed
    
    # Quick look at the data
    print(df.head())  # Displays first 5 rows
    print(df.info())  # Shows column names, data types, and non-null counts
    
    return df

# Note: We won't call this function here; main.py will use it