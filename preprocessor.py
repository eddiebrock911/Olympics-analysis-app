import pandas as pd

def preprocess(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    # Merge NOC -> region mapping
    if 'NOC' in df.columns and 'NOC' in region_df.columns:
        df = df.merge(region_df, on='NOC', how='left')
    else:
        # If region_df doesn't have NOC or merge fails, continue but warn
        pass

    # Only Summer Olympics
    if 'Season' in df.columns:
        df = df[df['Season'] == 'Summer']

    # Numeric conversions
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    if 'Height' in df.columns:
        df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
    if 'Weight' in df.columns:
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

    # Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    return df
