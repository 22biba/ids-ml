import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, roc_curve, auc, precision_recall_curve)

sys.path.append(os.path.dirname(__file__))
from load_data import load_all_data
from preprocess import preprocess

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compare_models(X_train, X_test, y_train, y_test):
    """Krahason 3 algoritme ML"""
    print("\n" + "="*50)
    print("📊 KRAHASIM ALGORITMESH")
    print("="*50)

    models = {
        'Random Forest':    RandomForestClassifier(n_estimators=100, max_depth=20,
                                                    random_state=42, n_jobs=-1),
        'Decision Tree':    DecisionTreeClassifier(max_depth=20, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in models.items():
        print(f"\n⏳ Duke trajnuar: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1  = f1_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred,
                                    target_names=['BENIGN','SULM'],
                                    output_dict=True)
        results[name] = {
            'model':     model,
            'f1':        f1,
            'precision': rep['SULM']['precision'],
            'recall':    rep['SULM']['recall'],
            'accuracy':  rep['accuracy']
        }
        print(f"   ✅ F1={f1:.4f} | Acc={rep['accuracy']:.4f}")

    return results

def plot_comparison(results):
    """Grafik krahasimi"""
    names   = list(results.keys())
    metrics = ['f1', 'precision', 'recall', 'accuracy']
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Krahasim Algoritmesh ML — IDS CICIDS2017', fontsize=14, fontweight='bold')

    for ax, metric, color in zip(axes.flat, metrics, colors):
        values = [results[n][metric] for n in names]
        bars   = ax.bar(names, values, color=color, alpha=0.8, edgecolor='black')
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_ylim(0.85, 1.01)
        ax.set_ylabel('Vlera')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150)
    print(f"\n✅ Grafiku u ruajt: results/model_comparison.png")
    plt.close()

def plot_roc_curves(results, X_test, y_test):
    """ROC Curves për të 3 modelet"""
    plt.figure(figsize=(8, 6))

    for name, data in results.items():
        model = data['model']
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.4f})')

    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Krahasim Algoritmesh')
    plt.legend(loc='lower right')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150)
    print(f"✅ ROC Curves u ruajt: results/roc_curves.png")
    plt.close()

def plot_feature_importance(results, feature_names):
    """Top 15 features më të rëndësishme"""
    rf = results['Random Forest']['model']
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.bar(range(15), importances[indices], color='#2196F3', alpha=0.8, edgecolor='black')
    plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Top 15 Features më të Rëndësishme — Random Forest')
    plt.ylabel('Importance Score')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    print(f"✅ Feature Importance u ruajt: results/feature_importance.png")
    plt.close()

if __name__ == "__main__":
    # Ngarko të dhënat
    df = load_all_data()
    X, y, le, scaler = preprocess(df)

    # Sample për shpejtësi (20% e të dhënave — mjafton shkencërisht)
    X_s, _, y_s, _ = train_test_split(X, y, test_size=0.8,
                                       random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
    )
    print(f"✅ Sample: Train={len(X_train):,} | Test={len(X_test):,}")

    # Krahaso modelet
    results = compare_models(X_train, X_test, y_train, y_test)

    # Grafiqet
    plot_comparison(results)
    plot_roc_curves(results, X_test, y_test)
    plot_feature_importance(results, list(X.columns))

    # Tabela finale
    print("\n" + "="*50)
    print("🏆 TABELA FINALE")
    print("="*50)
    print(f"{'Algoritmi':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
    print("-"*65)
    for name, data in results.items():
        print(f"{name:<25} {data['f1']:>8.4f} {data['precision']:>10.4f} "
              f"{data['recall']:>8.4f} {data['accuracy']:>10.4f}")

    print("\n🎉 Evaluimi i plotë përfundoi! Shiko folderin results/")