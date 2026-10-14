import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATA_FILE


def load_and_preprocess():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError("Mall_Customers.csv not found")

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    gender_col = 'Genre' if 'Genre' in df.columns else 'Gender'
    df['Gender_Encoded'] = df[gender_col].map({'Male': 0, 'Female': 1})

    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']
    X_raw = df[features].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return df, X_scaled, features, scaler
