from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Ngarko modelin
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Ruaj historikun e kërkesave
history = []

FEATURE_NAMES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean',
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
    'Idle Std', 'Idle Max', 'Idle Min'
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "sistem": "IDS - Zbulimi i Anomalive me ML",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Parashiko një paketë",
            "POST /predict_batch": "Parashiko shumë paketa",
            "GET /history": "Shiko historikun",
            "GET /stats": "Statistika të sistemit",
            "GET /health": "Gjendja e sistemit"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running",
        "model": "Random Forest",
        "timestamp": str(datetime.datetime.now())
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Mungojnë 'features' në kërkesë"}), 400

        features = data['features']
        if len(features) != 78:
            return jsonify({
                "error": f"Duhen 78 features, morëm {len(features)}"
            }), 400

        # Krijo DataFrame
        df = pd.DataFrame([features], columns=FEATURE_NAMES)
        df_scaled = scaler.transform(df)

        # Parashiko
        prediction = model.predict(df_scaled)[0]
        confidence = model.predict_proba(df_scaled)[0]

        result = "SULM" if prediction == 1 else "NORMAL"
        conf_score = float(max(confidence))

        # Ruaj në histori
        record = {
            "timestamp": str(datetime.datetime.now()),
            "prediction": result,
            "confidence": round(conf_score, 4)
        }
        history.append(record)

        return jsonify({
            "prediction": result,
            "confidence": round(conf_score, 4),
            "status": "🔴 SULM DETECTED" if prediction == 1 else "🟢 TRAFIK NORMAL",
            "timestamp": record["timestamp"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        if not data or 'packets' not in data:
            return jsonify({"error": "Mungon 'packets'"}), 400

        packets = data['packets']
        df = pd.DataFrame(packets, columns=FEATURE_NAMES)
        df_scaled = scaler.transform(df)

        predictions = model.predict(df_scaled)
        probas = model.predict_proba(df_scaled)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probas)):
            result = "SULM" if pred == 1 else "NORMAL"
            results.append({
                "paketa": i + 1,
                "prediction": result,
                "confidence": round(float(max(prob)), 4)
            })
            history.append({
                "timestamp": str(datetime.datetime.now()),
                "prediction": result,
                "confidence": round(float(max(prob)), 4)
            })

        sulme = sum(1 for r in results if r['prediction'] == 'SULM')

        return jsonify({
            "total_paketa": len(packets),
            "sulme_zbuluar": sulme,
            "normale": len(packets) - sulme,
            "rezultatet": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({
        "total_kerkesa": len(history),
        "historiku": history[-20:]  # Fundit 20
    })

@app.route('/stats', methods=['GET'])
def stats():
    if not history:
        return jsonify({"mesazh": "Asnjë kërkesë ende"})

    sulme = sum(1 for h in history if h['prediction'] == 'SULM')
    normale = len(history) - sulme

    return jsonify({
        "total_analizuar": len(history),
        "sulme_zbuluar": sulme,
        "normale": normale,
        "perqindja_sulmeve": round(sulme / len(history) * 100, 2),
        "model": "Random Forest",
        "accuracy": "99.87%",
        "f1_score": "0.9967"
    })

if __name__ == '__main__':
    import os
    print("\n" + "="*50)
    print("🚀 SISTEMI IDS API PO STARTON...")
    print("="*50)
    port = int(os.environ.get('PORT', 5000))
    print(f"📡 URL: http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
@app.route('/panel', methods=['GET'])
def panel():
    return render_template('test_panel.html')