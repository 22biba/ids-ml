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
    return '''<!DOCTYPE html>
<html>
<head>
    <title>IDS System — Cloud Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: "Segoe UI", Arial; background: #0a0e1a; color: #e0e0e0; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #1a237e, #0d47a1); padding: 20px; text-align: center; border-bottom: 2px solid #00d4ff; }
        .header h1 { font-size: 28px; color: #00d4ff; letter-spacing: 2px; }
        .header p { color: #90caf9; margin-top: 5px; font-size: 13px; }
        .container { max-width: 1100px; margin: 30px auto; padding: 0 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }
        .stat-card { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 10px; padding: 18px; text-align: center; }
        .stat-card .value { font-size: 28px; font-weight: bold; color: #00d4ff; }
        .stat-card .label { font-size: 12px; color: #90caf9; margin-top: 5px; text-transform: uppercase; }
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel-box { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 10px; padding: 20px; }
        .panel-box h3 { color: #00d4ff; margin-bottom: 15px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #1e3a5f; padding-bottom: 10px; }
        .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
        button { padding: 12px; border: none; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: bold; transition: all 0.2s; }
        .btn-blue { background: #1565c0; color: white; }
        .btn-blue:hover { background: #1976d2; transform: translateY(-1px); }
        .btn-green { background: #2e7d32; color: white; }
        .btn-green:hover { background: #388e3c; transform: translateY(-1px); }
        .btn-red { background: #c62828; color: white; }
        .btn-red:hover { background: #d32f2f; transform: translateY(-1px); }
        .btn-purple { background: #6a1b9a; color: white; }
        .btn-purple:hover { background: #7b1fa2; transform: translateY(-1px); }
        .btn-full { grid-column: span 2; background: #0277bd; color: white; }
        .btn-full:hover { background: #0288d1; }
        .result-box { background: #050d1a; border-radius: 8px; padding: 15px; min-height: 120px; font-family: monospace; font-size: 13px; border: 1px solid #1e3a5f; }
        .status-normal { background: #1b5e20; color: #69f0ae; padding: 12px; border-radius: 8px; text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 10px; }
        .status-attack { background: #b71c1c; color: #ff8a80; padding: 12px; border-radius: 8px; text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 10px; animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.7; } }
        .history-list { max-height: 300px; overflow-y: auto; }
        .history-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; background: #050d1a; border: 1px solid #1e3a5f; font-size: 12px; }
        .badge-normal { background: #1b5e20; color: #69f0ae; padding: 3px 10px; border-radius: 12px; font-size: 11px; }
        .badge-attack { background: #b71c1c; color: #ff8a80; padding: 3px 10px; border-radius: 12px; font-size: 11px; }
        .live-dot { width: 10px; height: 10px; background: #00e676; border-radius: 50%; display: inline-block; margin-right: 6px; animation: blink 1s infinite; }
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }
        .confidence-bar { background: #1e3a5f; border-radius: 4px; height: 8px; margin-top: 4px; }
        .confidence-fill { background: linear-gradient(90deg, #00d4ff, #00e676); height: 8px; border-radius: 4px; transition: width 0.5s; }
        .spinner { display: none; text-align: center; padding: 20px; color: #00d4ff; }
        .footer { text-align: center; padding: 20px; color: #37474f; font-size: 12px; margin-top: 30px; border-top: 1px solid #1e3a5f; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ IDS SYSTEM — CLOUD DASHBOARD</h1>
        <p><span class="live-dot"></span>LIVE — ids-ml-6che.onrender.com &nbsp;|&nbsp; Model: Random Forest &nbsp;|&nbsp; Dataset: CICIDS2017</p>
    </div>

    <div class="container">
        <!-- Stats Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value" id="total">—</div>
                <div class="label">Paketa të Analizuara</div>
            </div>
            <div class="stat-card">
                <div class="value" id="sulme" style="color:#ff5252">—</div>
                <div class="label">Sulme të Zbuluara</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color:#69f0ae">99.87%</div>
                <div class="label">Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color:#ffab40">0.9967</div>
                <div class="label">F1-Score</div>
            </div>
        </div>

        <div class="main-grid">
            <!-- Panel Testimi -->
            <div class="panel-box">
                <h3>🔬 Panel Testimi</h3>
                <div class="btn-grid">
                    <button class="btn-blue" onclick="testHealth()">Health Check</button>
                    <button class="btn-purple" onclick="testStats()">Statistika</button>
                    <button class="btn-green" onclick="testNormal()">Paketë NORMALE</button>
                    <button class="btn-red" onclick="testAttack()">Paketë SULM</button>
                    <button class="btn-full" onclick="testBatch()">Batch Test (5 paketa njëherësh)</button>
                </div>
                <div id="status-box"></div>
                <div id="conf-bar" style="display:none">
                    <div style="font-size:11px; color:#90caf9; margin-bottom:3px">Besueshmëria:</div>
                    <div class="confidence-bar"><div class="confidence-fill" id="conf-fill" style="width:0%"></div></div>
                </div>
                <div class="spinner" id="spinner">⏳ Duke analizuar paketën...</div>
                <div class="result-box" id="output">Klikoni një buton për të testuar sistemin...</div>
            </div>

            <!-- Historia Live -->
            <div class="panel-box">
                <h3>📜 Historia Live <button onclick="loadHistory()" style="float:right; background:#1e3a5f; color:#00d4ff; border:none; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px;">🔄 Refresh</button></h3>
                <div class="history-list" id="history-list">
                    <div style="color:#37474f; text-align:center; padding:20px">Asnjë kërkesë ende...</div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        🎓 Master Thesis — Zbulimi i Anomalive në Trafik Cloud duke përdorur ML &nbsp;|&nbsp; CICIDS2017 Dataset &nbsp;|&nbsp; 2.8M paketa të trajnuara
    </div>

    <script>
        async function callAPI(url, method="GET", body=null) {
            document.getElementById("spinner").style.display = "block";
            document.getElementById("output").textContent = "";
            try {
                const opts = { method, headers: {"Content-Type": "application/json"} };
                if (body) opts.body = JSON.stringify(body);
                const res = await fetch(url, opts);
                const data = await res.json();

                if (data.prediction) {
                    const isAttack = data.prediction === "SULM";
                    document.getElementById("status-box").innerHTML =
                        "<div class='" + (isAttack ? "status-attack" : "status-normal") + "'>" +
                        (isAttack ? "🔴 SULM I ZBULUAR!" : "🟢 TRAFIK NORMAL") + "</div>";
                    const conf = Math.round(data.confidence * 100);
                    document.getElementById("conf-bar").style.display = "block";
                    document.getElementById("conf-fill").style.width = conf + "%";
                } else {
                    document.getElementById("status-box").innerHTML = "";
                    document.getElementById("conf-bar").style.display = "none";
                }
                document.getElementById("output").textContent = JSON.stringify(data, null, 2);
                loadStats();
                loadHistory();
            } catch(e) {
                document.getElementById("output").textContent = "❌ Gabim: " + e.message;
            }
            document.getElementById("spinner").style.display = "none";
        }

        async function loadStats() {
            try {
                const res = await fetch("/stats");
                const data = await res.json();
                document.getElementById("total").textContent = data.total_analizuar || 0;
                document.getElementById("sulme").textContent = data.sulme_zbuluar || 0;
            } catch(e) {}
        }

        async function loadHistory() {
            try {
                const res = await fetch("/history");
                const data = await res.json();
                const list = document.getElementById("history-list");
                if (!data.historiku || data.historiku.length === 0) {
                    list.innerHTML = "<div style=\\"color:#37474f; text-align:center; padding:20px\\">Asnjë kërkesë ende...</div>";
                    return;
                }
                list.innerHTML = data.historiku.slice().reverse().map(h =>
                    "<div class=\\"history-item\\">" +
                    "<span class=\\"badge-" + (h.prediction === "SULM" ? "attack\\">" + "🔴 SULM" : "normal\\">" + "🟢 NORMAL") + "</span>" +
                    "<span style=\\"color:#90caf9\\">" + Math.round(h.confidence * 100) + "% conf</span>" +
                    "<span style=\\"color:#546e7a; font-size:11px\\">" + h.timestamp.split(" ")[1].split(".")[0] + "</span>" +
                    "</div>"
                ).join("");
            } catch(e) {}
        }

        const testHealth  = () => callAPI("/health");
        const testStats   = () => callAPI("/stats");
        const testNormal  = () => callAPI("/predict", "POST", {
            features: [80,100000,5,3,1500,800,1500,40,300,200,800,20,150,100,50000,10,5000,2000,10000,100,8000,2000,1500,6000,200,5000,1500,1200,4000,100,0,0,0,0,40,32,5,3,40,1500,300,200,40000,0,1,0,1,1,0,0,0,1,300,300,150,40,0,0,0,0,0,0,5,1500,3,800,65535,256,5,20,0,0,0,0,0,0,0,0]
        });
        const testAttack  = () => callAPI("/predict", "POST", {
            features: [0,5,1000,0,44000,0,44,44,44,0,0,0,0,0,8800000,200000,5,0,5,5,5,5,0,5,5,0,0,0,0,0,0,0,0,0,20,0,200000,0,44,44,44,0,0,0,0,0,0,0,0,0,0,0,44,44,0,20,0,0,0,0,0,0,1000,44000,0,0,0,0,1000,20,0,0,0,0,0,0,0,0]
        });
        async function testBatch() {
            await callAPI("/predict_batch", "POST", { packets: [
                [80,100000,5,3,1500,800,1500,40,300,200,800,20,150,100,50000,10,5000,2000,10000,100,8000,2000,1500,6000,200,5000,1500,1200,4000,100,0,0,0,0,40,32,5,3,40,1500,300,200,40000,0,1,0,1,1,0,0,0,1,300,300,150,40,0,0,0,0,0,0,5,1500,3,800,65535,256,5,20,0,0,0,0,0,0,0,0],
                [0,5,1000,0,44000,0,44,44,44,0,0,0,0,0,8800000,200000,5,0,5,5,5,5,0,5,5,0,0,0,0,0,0,0,0,0,20,0,200000,0,44,44,44,0,0,0,0,0,0,0,0,0,0,0,44,44,0,20,0,0,0,0,0,0,1000,44000,0,0,0,0,1000,20,0,0,0,0,0,0,0,0],
                [80,100000,5,3,1500,800,1500,40,300,200,800,20,150,100,50000,10,5000,2000,10000,100,8000,2000,1500,6000,200,5000,1500,1200,4000,100,0,0,0,0,40,32,5,3,40,1500,300,200,40000,0,1,0,1,1,0,0,0,1,300,300,150,40,0,0,0,0,0,0,5,1500,3,800,65535,256,5,20,0,0,0,0,0,0,0,0],
                [0,5,1000,0,44000,0,44,44,44,0,0,0,0,0,8800000,200000,5,0,5,5,5,5,0,5,5,0,0,0,0,0,0,0,0,0,20,0,200000,0,44,44,44,0,0,0,0,0,0,0,0,0,0,0,44,44,0,20,0,0,0,0,0,0,1000,44000,0,0,0,0,1000,20,0,0,0,0,0,0,0,0],
                [80,100000,5,3,1500,800,1500,40,300,200,800,20,150,100,50000,10,5000,2000,10000,100,8000,2000,1500,6000,200,5000,1500,1200,4000,100,0,0,0,0,40,32,5,3,40,1500,300,200,40000,0,1,0,1,1,0,0,0,1,300,300,150,40,0,0,0,0,0,0,5,1500,3,800,65535,256,5,20,0,0,0,0,0,0,0,0]
            ]});
        }
        // Auto-refresh stats çdo 10 sekonda
        loadStats();
        loadHistory();
        setInterval(() => { loadStats(); loadHistory(); }, 10000);
    </script>
</body>
</html>'''