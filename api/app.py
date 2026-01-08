# api/app.py
from __future__ import annotations

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np


# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "../models/best_model.pkl"
SCALER_PATH = "../models/scaler.pkl"
FEATURES = [f"var_{i}" for i in range(200)]
DEFAULT_THRESHOLD = 0.5


# -----------------------------
# App init + load artifacts once
# -----------------------------
app = Flask(__name__)
CORS(app)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle: {MODEL_PATH}. Détail: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None  # ok si le modèle n'en a pas besoin

MODEL_TYPE = model.__class__.__name__
USES_SCALER = (MODEL_TYPE == "LogisticRegression") and (scaler is not None)


# -----------------------------
# Helpers
# -----------------------------
def _json_error(message: str, status_code: int = 400, **extra):
    payload = {"error": message, **extra}
    return jsonify(payload), status_code


def _extract_features(payload: dict) -> np.ndarray:
    """
    Extrait les 200 features dans le bon ordre et retourne un array shape (1, 200).
    """
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise KeyError(missing)

    x = np.array([payload[f] for f in FEATURES], dtype=float).reshape(1, -1)

    # Vérif NaN/Inf
    if not np.isfinite(x).all():
        raise ValueError("NaN/Inf détecté")

    return x


def _predict_proba(x: np.ndarray) -> float:
    """
    Retourne la proba de la classe 1.
    """
    x_in = scaler.transform(x) if USES_SCALER else x

    # predict_proba attendu
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Le modèle ne supporte pas predict_proba().")

    return float(model.predict_proba(x_in)[0, 1])


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/model-info")
def model_info():
    return jsonify({
        "model_type": MODEL_TYPE,
        "n_features": len(FEATURES),
        "feature_names": FEATURES,          # tu peux enlever si tu veux plus léger
        "uses_scaler": USES_SCALER,
        "threshold_default": DEFAULT_THRESHOLD
    })


@app.post("/predict")
def predict():
    """
    Body JSON attendu:
      - soit un dict avec var_0..var_199
      - soit {"features": {var_0..var_199}, "threshold": 0.6}
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return _json_error("Body JSON manquant", 400)

        # Threshold optionnel
        threshold = data.get("threshold", DEFAULT_THRESHOLD)
        try:
            threshold = float(threshold)
        except Exception:
            return _json_error("threshold doit être un nombre", 400)

        if not (0.0 <= threshold <= 1.0):
            return _json_error("threshold doit être compris entre 0 et 1", 400)

        # Supporte deux formats: direct ou via clé "features"
        features_payload = data.get("features", data)

        try:
            x = _extract_features(features_payload)
        except KeyError as ke:
            missing = list(ke.args[0])
            return _json_error(
                "Features manquantes",
                400,
                missing=missing[:10],
                missing_count=len(missing)
            )
        except ValueError:
            return _json_error("Valeurs invalides : NaN/inf ou valeurs non numériques", 400)

        proba = _predict_proba(x)
        decision = int(proba >= threshold)

        return jsonify({
            "proba_target_1": proba,
            "decision": decision,
            "threshold": threshold,
            "model_type": MODEL_TYPE
        })

    except ValueError:
        return _json_error("Valeurs invalides : toutes les features doivent être numériques", 400)
    except AttributeError as e:
        return _json_error(str(e), 500)
    except Exception as e:
        # En prod on ne renvoie pas le détail, mais pour ton projet (debug) c'est ok
        return _json_error("Erreur interne serveur", 500, details=str(e))


@app.post("/predict-batch")
def predict_batch():
    """
    Body JSON attendu:
      {
        "rows": [
          {"var_0":..., ..., "var_199":...},
          {"var_0":..., ..., "var_199":...}
        ],
        "threshold": 0.5
      }
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return _json_error("Body JSON manquant", 400)

        rows = data.get("rows")
        if not isinstance(rows, list) or len(rows) == 0:
            return _json_error("rows doit être une liste non vide", 400)

        threshold = data.get("threshold", DEFAULT_THRESHOLD)
        try:
            threshold = float(threshold)
        except Exception:
            return _json_error("threshold doit être un nombre", 400)
        if not (0.0 <= threshold <= 1.0):
            return _json_error("threshold doit être compris entre 0 et 1", 400)

        X = []
        for idx, row in enumerate(rows):
            try:
                x = _extract_features(row)  # (1,200)
                X.append(x[0])
            except KeyError as ke:
                missing = list(ke.args[0])
                return _json_error(
                    f"Features manquantes sur la ligne {idx}",
                    400,
                    row_index=idx,
                    missing=missing[:10],
                    missing_count=len(missing)
                )
            except ValueError:
                return _json_error(
                    f"Valeurs invalides (NaN/inf ou non numérique) sur la ligne {idx}",
                    400,
                    row_index=idx
                )

        X = np.array(X, dtype=float)
        X_in = scaler.transform(X) if USES_SCALER else X

        probas = model.predict_proba(X_in)[:, 1].astype(float)
        decisions = (probas >= threshold).astype(int)

        return jsonify({
            "threshold": threshold,
            "model_type": MODEL_TYPE,
            "predictions": [
                {"proba_target_1": float(p), "decision": int(d)}
                for p, d in zip(probas, decisions)
            ]
        })

    except Exception as e:
        return _json_error("Erreur interne serveur", 500, details=str(e))


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)