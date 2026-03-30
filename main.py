import os
import sys

from src.evaluate_model import compare_models, plot_comparison, plot_roc_curves, plot_feature_importance
from src.load_data import load_all_data
from src.preprocess import preprocess
from src.realtime_simulation import simulate_realtime
from src.train_model import evaluate, save_model, train

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from sklearn.model_selection import train_test_split

def main():
    print("\n" + "🔷"*25)
    print("   SISTEMI IDS — ZBULIMI I ANOMALIVE ME ML")
    print("🔷"*25)

    # 1. Ngarko
    df = load_all_data()

    # 2. Preproceso
    X, y, le, scaler = preprocess(df)

    # 3. Split
    X_s, _, y_s, _ = train_test_split(X, y, test_size=0.8,
                                       random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
    )

    # 4. Trajno & Vlerëso
    model, X_test_full, y_test_full = train(X, y)
    evaluate(model, X_test_full, y_test_full)
    save_model(model, scaler)

    # 5. Krahasim algoritmesh
    results = compare_models(X_train, X_test, y_train, y_test)
    plot_comparison(results)
    plot_roc_curves(results, X_test, y_test)
    plot_feature_importance(results, list(X.columns))

    # 6. Simulim live
    X_test_full = X_test_full.reset_index(drop=True)
    y_test_full = y_test_full.reset_index(drop=True)
    simulate_realtime(model, X_test_full, y_test_full, n_packets=20)

    print("\n" + "🎉"*20)
    print("   IMPLEMENTIMI I PLOTË PËRFUNDOI!")
    print("🎉"*20)

if __name__ == "__main__":
    main()