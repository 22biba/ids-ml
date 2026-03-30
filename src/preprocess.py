import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from load_data import load_all_data
import os


def preprocess(df):
    print("\n" + "=" * 50)
    print("🔧 DUKE PASTRUAR TË DHËNAT...")
    print("=" * 50)

    # 1. Pastro emrat e kolonave (hiq hapësirat)
    df.columns = df.columns.str.strip()
    print("✅ Emrat e kolonave u pastruan")

    # 2. Hiq vlerat null
    before = len(df)
    df = df.dropna()
    print(f"✅ Vlerat null u hoqën: {before - len(df)} rreshta")

    # 3. Hiq vlerat Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"✅ Vlerat Inf u hoqën. Rreshta mbetën: {len(df):,}")

    # 4. Enkodo Label (BENIGN=0, sulme=1 per binary)
    le = LabelEncoder()
    df['Label_encoded'] = le.fit_transform(df['Label'])

    # Binary classification: BENIGN=0, çdo sulm=1
    df['Label_binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    print(f"\n✅ Etiketat u enkoduan:")
    print(f"   BENIGN (0): {(df['Label_binary'] == 0).sum():,}")
    print(f"   SULM   (1): {(df['Label_binary'] == 1).sum():,}")

    # 5. Zgjidh features (hiq Label)
    feature_cols = [c for c in df.columns if c not in ['Label', 'Label_encoded', 'Label_binary']]
    X = df[feature_cols]
    y = df['Label_binary']

    # 6. Normalizim
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    print(f"\n✅ Features u normalizuan: {len(feature_cols)} kolona")

    return X_scaled, y, le, scaler


if __name__ == "__main__":
    df = load_all_data()
    X, y, le, scaler = preprocess(df)
    print(f"\n✅ X shape: {X.shape}")
    print(f"✅ y shape: {y.shape}")
    print("\n🎯 Preprocessing u krye me sukses!")