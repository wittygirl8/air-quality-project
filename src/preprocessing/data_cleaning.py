import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the data by handling missing values appropriately."""
    # Select only numeric columns for filling missing values with median
    numeric_cols = df.select_dtypes(include=['number'])
    df[numeric_cols.columns] = numeric_cols.apply(lambda x: x.fillna(x.median()))
    return df

if __name__ == "__main__":
    # Load the raw data
    train = load_data("data/raw/val_data_new.csv")
    
    # Clean the data
    train = clean_data(train)
    
    # Save the cleaned data
    train.to_csv("data/processed/cleaned_val_data_new.csv", index=False)
