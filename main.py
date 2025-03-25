from data_loading import load_data
from data_cleaning import clean_data
from data_preprocessing import preprocess_data
from eda import perform_eda

# Define your CSV file path
file_path = 'atp_matches_2024.csv'  # Replace with your actual file path

# Load the data once
df = load_data(file_path)

# Clean the data
df_cleaned = clean_data(df)

# Preprocess the data
df_preprocessed = preprocess_data(df_cleaned)

# Perform EDA
perform_eda(df_preprocessed)