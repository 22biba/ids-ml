import pandas as pd
import os

# Rruga e datasetit
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'MachineLearningCVE')


def load_all_data():
    """Ngarkon të gjitha CSV-të dhe i bashkon në një DataFrame"""
    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]

    print(f"✅ Gjeta {len(all_files)} skedarë CSV:")
    for f in all_files:
        print(f"   - {f}")

    dataframes = []
    for file in all_files:
        filepath = os.path.join(DATA_PATH, file)
        print(f"\n📂 Duke ngarkuar: {file}")
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        print(f"   Rreshta: {len(df):,} | Kolona: {len(df.columns)}")
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)
    print(f"\n✅ TOTAL: {len(combined):,} rreshta të bashkuara")
    return combined


def explore_data(df):
    """Shfaq informacion bazë për datasetin"""
    print("\n" + "=" * 50)
    print("📊 INFORMACION PËR DATASETIN")
    print("=" * 50)
    print(f"Dimensionet: {df.shape}")
    print(f"\nKolonat:\n{list(df.columns)}")
    print(f"\nLlojet e sulmeve:\n{df[' Label'].value_counts()}")
    print(f"\nVlera null:\n{df.isnull().sum().sum()} total")


if __name__ == "__main__":
    df = load_all_data()
    explore_data(df)