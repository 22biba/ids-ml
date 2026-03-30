import requests
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))
from load_data import load_all_data
from preprocess import preprocess
from sklearn.model_selection import train_test_split

API_URL = "http://localhost:5000"

def load_real_packets():
    """Ngarko paketa reale nga dataseti"""
    print("📂 Duke ngarkuar paketa reale nga CICIDS2017...")
    df = load_all_data()
    X, y, le, scaler = preprocess(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"✅ {len(X_test):,} paketa gati për simulim!\n")
    return X_test, y_test

def simulate_live_traffic(X_test, y_test, n_packets=50):
    """Dërgon paketa reale tek API dhe shfaq rezultatin live"""
    print("="*60)
    print("🔴 SIMULIM TRAFIKU LIVE — CICIDS2017")
    print("="*60)
    print(f"{'#':<5} {'PARASHIKIM':<16} {'REAL':<12} {'CONF':>6}  {'REZULTAT'}")
    print("-"*60)

    stats = {"correct": 0, "sulme": 0, "normale": 0, "total": 0}

    # Merr paketa të balancuara (mix sulmesh dhe normale)
    sulm_idx   = y_test[y_test == 1].index[:n_packets//2].tolist()
    normal_idx = y_test[y_test == 0].index[:n_packets//2].tolist()
    indices    = sulm_idx + normal_idx
    np.random.shuffle(indices)

    for i, idx in enumerate(indices):
        packet   = X_test.iloc[idx].tolist()
        actual   = y_test.iloc[idx]

        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"features": packet},
                timeout=5
            )
            data = resp.json()

            pred       = data['prediction']
            conf       = data['confidence']
            timestamp  = data['timestamp'].split(' ')[1][:8]

            actual_lbl = "SULM" if actual == 1 else "NORMAL"
            correct    = pred == actual_lbl
            icon_pred  = "🔴 SULM  " if pred == "SULM" else "🟢 NORMAL"
            icon_res   = "✅" if correct else "❌"

            if correct:
                stats["correct"] += 1
            if pred == "SULM":
                stats["sulme"] += 1
            else:
                stats["normale"] += 1
            stats["total"] += 1

            print(f"{i+1:<5} {icon_pred:<16} {actual_lbl:<12} {conf:>6.2f}  {icon_res} [{timestamp}]")

        except Exception as e:
            print(f"{i+1:<5} ❌ Gabim: {e}")

        time.sleep(0.3)  # Simulon vonesë rrjeti

    # Rezultatet finale
    print("="*60)
    print(f"\n📊 REZULTATE FINALE:")
    print(f"   Paketa të analizuara : {stats['total']}")
    print(f"   🔴 Sulme të zbuluara : {stats['sulme']}")
    print(f"   🟢 Normale           : {stats['normale']}")
    acc = stats['correct'] / stats['total'] * 100
    print(f"   ✅ Saktësi live      : {acc:.1f}%")

    # Merr statistikat nga API
    print("\n📡 STATISTIKAT E API-T:")
    r = requests.get(f"{API_URL}/stats").json()
    print(f"   Total të analizuara  : {r['total_analizuar']}")
    print(f"   Sulme të zbuluara    : {r['sulme_zbuluar']}")
    print(f"   Përqindja sulmeve    : {r['perqindja_sulmeve']}%")
    print(f"   Modeli               : {r['model']}")
    print(f"   Accuracy             : {r['accuracy']}")
    print(f"   F1-Score             : {r['f1_score']}")

if __name__ == "__main__":
    print("\n🚀 LIVE CLIENT — SISTEMI IDS\n")

    # Kontrollo nese API eshte aktive
    try:
        requests.get(f"{API_URL}/health", timeout=3)
        print("✅ API është aktive!\n")
    except:
        print("❌ API nuk është aktive! Ekzekuto: python src/api.py")
        sys.exit(1)

    # Ngarko paketa reale
    X_test, y_test = load_real_packets()

    # Simulim me 50 paketa
    simulate_live_traffic(X_test, y_test, n_packets=50)

    print("\n🎉 SIMULIMI PËRFUNDOI!")