from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)
scaler = joblib.load("../models/scaler.pkl")
model = joblib.load("../models/best_model.pkl")

# liste des features dans le bon ordre
FEATURES = [f"var_{i}" for i in range(200)]
@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Body JSON manquant"}), 400

    #Vérifier que toutes les features sont présentes
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Features manquantes", "missing": missing[:10], "missing_count": len(missing)}), 400

    #Transformer les données en array numpy
    x = np.array([data[f] for f in FEATURES], dtype=float).reshape(1, -1)

    #Appliquer le scaler UNIQUEMENT si le modèle est une Logistic Regression
    if model.__class__.__name__ == "LogisticRegression":
        x = scaler.transform(x)

    #Faire la prédiction
    proba = float(model.predict_proba(x)[0, 1])

    decision = int(proba >= 0.5)

    return jsonify({
        "proba_target_1": proba,
        "decision": decision,
        "threshold": 0.5
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

