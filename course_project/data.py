import pandas as pd


def data_preparation():
    df = pd.read_csv('result_table.csv', low_memory=False)
    df['D_REG'] = pd.to_datetime(df['D_REG'], format='%d.%m.%Y', errors='coerce')
    # Extract year from the registration date
    df['year'] = df['D_REG'].dt.year
    # Replace commas with periods in float columns
    float_columns = ['CAPACITY', 'OWN_WEIGHT', 'TOTAL_WEIGHT']
    for col in float_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)
    return df
