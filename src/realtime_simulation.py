import pandas as pd
import numpy as np
import pickle
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
from load_data import load_all_data
from preprocess import preprocess
from sklearn.model_selection import train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model():
    """Ngarko modelin e trajnuar"""
    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Modeli u ngarkua nga disku")
    return model, scaler

def simulate_realtime(model, X_test, y_test, n_packets=20):
    """Simulon inspektim trafiku në kohë reale"""
    print("\n" + "="*55)
    print("🔴 SIMULIM TRAFIKU NË KOHË REALE")
    print("="*55)
    print(f"{'#':<5} {'STATUSI':<12} {'REAL':<12} {'REZULTATI'}")
    print("-"*55)

    correct = 0
    attacks_found = 0

    # Merr 20 paketa të rastësishme
    indices = np.random.choice(len(X_test), n_packets, replace=False)

    for i, idx in enumerate(indices):
        packet = X_test.iloc[[idx]]
        actual = y_test.iloc[idx]
        prediction = model.predict(packet)[0]
        proba = model.predict_proba(packet)[0][1]

        status = "🔴 SULM" if prediction == 1 else "🟢 NORMAL"
        actual_label = "SULM" if actual == 1 else "NORMAL"
        result = "✅ SAKTË" if prediction == actual else "❌ GABIM"

        if prediction == actual:
            correct += 1
        if prediction == 1:
            attacks_found += 1

        print(f"{i+1:<5} {status:<14} {actual_label:<12} {result}  (conf: {proba:.2f})")
        time.sleep(0.1)  # Simulon vonesë rrjeti

    print("-"*55)
    print(f"\n📊 REZULTATE SIMULIMIT:")
    print(f"   Paketa të analizuara : {n_packets}")
    print(f"   Sulme të zbuluara    : {attacks_found}")
    print(f"   Saktësi              : {correct/n_packets*100:.1f}%")

def save_summary_report(results_dict):
    """Ruan raportin final si CSV"""
    df = pd.DataFrame(results_dict)
    path = os.path.join(RESULTS_DIR, 'simulation_report.csv')
    df.to_csv(path, index=False)
    print(f"\n✅ Raporti u ruajt: results/simulation_report.csv")

if __name__ == "__main__":
    # Ngarko dataset dhe preproceso
    df = load_all_data()
    X, y, le, scaler = preprocess(df)

    # Merr vetëm test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Ngarko modelin
    model, _ = load_model()

    # Simulim
    simulate_realtime(model, X_test, y_test, n_packets=20)

    # Ruaj raport
    save_summary_report({
        'Algoritmi': ['Random Forest', 'Decision Tree', 'Logistic Regression'],
        'F1':        [0.9967, 0.9965, 0.8496],
        'Precision': [0.9970, 0.9956, 0.8576],
        'Recall':    [0.9963, 0.9975, 0.8417],
        'Accuracy':  [0.9987, 0.9986, 0.9413],
        'AUC':       [0.9999, 0.9986, 0.9800]
    })

    print("\n🎉 SISTEMI IDS ËSHTË FUNKSIONAL!")
    print("📁 Të gjitha rezultatet janë në folderin: results/")