import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import nga modulet tona
import sys
sys.path.append(os.path.dirname(__file__))
from load_data import load_all_data
from preprocess import preprocess

def train(X, y):
    print("\n" + "="*50)
    print("🤖 DUKE TRAJNUAR MODELIN...")
    print("="*50)

    # 1. Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 2. Perdor vetem 20% te train per shpejtesi (RAM)
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train, y_train, test_size=0.8, random_state=42, stratify=y_train
    )
    print(f"✅ Duke trajnuar me: {len(X_train_small):,} rreshta (sample)")

    # 3. Random Forest
    print("\n⏳ Duke trajnuar Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,       # perdor te gjitha CPU cores
        verbose=1
    )
    model.fit(X_train_small, y_train_small)
    print("✅ Trajnimi perfundoi!")

    return model, X_test, y_test

def evaluate(model, X_test, y_test):
    print("\n" + "="*50)
    print("📊 VLERESIMI I MODELIT")
    print("="*50)

    y_pred = model.predict(X_test)

    # Metrikat
    report = classification_report(y_test, y_pred, target_names=['BENIGN', 'SULM'])
    print(report)

    f1 = f1_score(y_test, y_pred)
    print(f"🎯 F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BENIGN', 'SULM'],
                yticklabels=['BENIGN', 'SULM'])
    plt.title('Confusion Matrix — Random Forest IDS')
    plt.ylabel('Aktual')
    plt.xlabel('Parashikuar')
    plt.tight_layout()

    # Ruaj grafiken
    output_path = os.path.join(os.path.dirname(__file__), '..', 'confusion_matrix.png')
    plt.savefig(output_path)
    print(f"\n✅ Confusion Matrix u ruajt: confusion_matrix.png")
    plt.show()

    return f1

def save_model(model, scaler):
    """Ruaj modelin per perdorim me vone"""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Modeli u ruajt në: models/rf_model.pkl")

if __name__ == "__main__":
    # Ngarko dhe preproceso
    df = load_all_data()
    X, y, le, scaler = preprocess(df)

    # Trajno
    model, X_test, y_test = train(X, y)

    # Vlerëso
    f1 = evaluate(model, X_test, y_test)

    # Ruaj
    save_model(model, scaler)

    print("\n🎉 GATI! Modeli është trajnuar dhe ruajtur!")