# api/app.py
from __future__ import annotations
 
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
 
 
# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "../models/best_model.pkl").resolve()
SCALER_PATH = (BASE_DIR / "../models/scaler.pkl").resolve()
 
N_FEATURES = 200
DEFAULT_THRESHOLD = 0.5
 
# Ajustement agent (borné)
AGENT_ADJ_MIN = -0.30
AGENT_ADJ_MAX = +0.30
 
# Garde-fous métier (optionnels)
DEFAULT_MAX_DEBT_RATIO_AFTER = 0.45   # taux endettement après crédit
DEFAULT_MIN_RESTE_A_VIVRE_AFTER = 0   # reste à vivre après crédit >= 0
 
# Paramètre de simulation crédit (si tu n'as pas de taux issu modèle)
DEFAULT_TAUX_ANNUEL = 0.035  # 3.5%
 
 
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
    scaler = None
 
MODEL_TYPE = model.__class__.__name__
USES_SCALER = scaler is not None
 
 
# -----------------------------
# Helpers
# -----------------------------
def _json_error(message: str, status_code: int = 400, **extra):
    payload = {"error": message, **extra}
    return jsonify(payload), status_code
 
 
def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)
 
 
def _to_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)
 
 
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
 
 
def _calc_mensualite(montant: float, duree_mois: int, taux_annuel: float) -> float:
    """
    Mensualité crédit amortissable.
    """
    montant = max(0.0, float(montant))
    duree_mois = max(1, int(duree_mois))
    taux_annuel = max(0.0, float(taux_annuel))
 
    taux_mensuel = taux_annuel / 12.0
    n = duree_mois
 
    if taux_mensuel <= 0:
        return montant / n
 
    # formule standard
    return montant * (taux_mensuel * (1 + taux_mensuel) ** n) / ((1 + taux_mensuel) ** n - 1)
 
 
def _business_to_features(b: dict, mensualite_credit: float, taux_endettement_after: float, reste_a_vivre_after: float) -> np.ndarray:
    """
    Convertit le JSON métier (y compris crédit demandé) vers un array (1, 200).
    Mapping déterministe + ratios utiles + one-hot statut + info crédit.
    """
    age = _to_float(b.get("age", 30))
    revenu = _to_float(b.get("revenu_mensuel", 0))
    charges = _to_float(b.get("charges_mensuelles", 0))
    credits = _to_float(b.get("credits_encours", 0))
    anciennete = _to_float(b.get("anciennete_pro", 0))
    annees_res = _to_float(b.get("annees_residence", 0))
 
    statut = str(b.get("statut_pro", "")).strip()
 
    montant_credit = _to_float(b.get("montant_credit", 0))
    duree_credit = _to_int(b.get("duree_credit", 0))
    objet_credit = str(b.get("objet_credit", "")).strip()
 
    denom = max(revenu, 1.0)
    taux_charges = charges / denom
    taux_credits = credits / denom
    taux_endettement_before = (charges + credits) / denom
    reste_a_vivre_before = revenu - charges - credits
 
    # One-hot statut (simple)
    is_cdi = 1.0 if statut == "CDI" else 0.0
    is_cdd = 1.0 if statut == "CDD" else 0.0
    is_sans_emploi = 1.0 if statut == "Sans emploi" else 0.0
    is_etudiant = 1.0 if statut == "Étudiant" else 0.0
    is_fonctionnaire = 1.0 if statut == "Fonctionnaire" else 0.0
    is_independant = 1.0 if statut == "Indépendant" else 0.0
    is_retraite = 1.0 if statut == "Retraité" else 0.0
 
    # One-hot objet (light)
    obj_map = ["Achat immobilier", "Travaux", "Véhicule", "Consommation", "Trésorerie", "Autre"]
    obj_oh = [1.0 if objet_credit == o else 0.0 for o in obj_map]
 
    feats = np.zeros(N_FEATURES, dtype=float)
 
    # --- features explicables (0..29)
    feats[0] = age
    feats[1] = revenu
    feats[2] = charges
    feats[3] = credits
    feats[4] = anciennete
    feats[5] = annees_res
 
    feats[6] = is_cdi
    feats[7] = is_cdd
    feats[8] = is_sans_emploi
    feats[9] = is_etudiant
    feats[10] = is_fonctionnaire
    feats[11] = is_independant
    feats[12] = is_retraite
 
    feats[13] = taux_charges
    feats[14] = taux_credits
    feats[15] = taux_endettement_before
    feats[16] = reste_a_vivre_before
 
    # infos crédit demandé
    feats[17] = montant_credit
    feats[18] = float(duree_credit)
    feats[19] = float(mensualite_credit)
 
    # impact après crédit
    feats[20] = float(taux_endettement_after)
    feats[21] = float(reste_a_vivre_after)
 
    # objet one-hot (22..27)
    for i, v in enumerate(obj_oh):
        if 22 + i < N_FEATURES:
            feats[22 + i] = v
 
    # --- remplissage déterministe du reste
    base_mix = (
        age * 0.01
        + revenu * 0.001
        - charges * 0.0012
        - credits * 0.0015
        - mensualite_credit * 0.0018
        + anciennete * 0.004
        + annees_res * 0.01
        + is_cdi * 0.2
        - is_sans_emploi * 0.3
        - taux_endettement_after * 0.5
        + (reste_a_vivre_after / max(revenu, 1.0)) * 0.2
    )
    for i in range(28, N_FEATURES):
        feats[i] = base_mix + (i % 7) * 0.03 - (i % 5) * 0.02
 
    if not np.isfinite(feats).all():
        raise ValueError("NaN/Inf détecté dans les features")
 
    return feats.reshape(1, -1)
 
 
def _predict_risk_score(x: np.ndarray) -> float:
    """
    Score de risque = proba classe 1.
    On interprète classe 1 comme RISQUE (refus).
    """
    x_in = scaler.transform(x) if USES_SCALER else x
 
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
        "n_features": N_FEATURES,
        "uses_scaler": USES_SCALER,
        "threshold_default": DEFAULT_THRESHOLD,
        "agent_adjustment_range": [AGENT_ADJ_MIN, AGENT_ADJ_MAX],
        "guardrails_defaults": {
            "max_debt_ratio_after": DEFAULT_MAX_DEBT_RATIO_AFTER,
            "min_reste_a_vivre_after": DEFAULT_MIN_RESTE_A_VIVRE_AFTER
        },
        "credit_defaults": {
            "taux_annuel_default": DEFAULT_TAUX_ANNUEL
        }
    })
 
 
@app.post("/predict")
def predict():
    """
    JSON métier attendu (AVEC CREDIT DEMANDÉ) :
    {
      "age": 28,
      "statut_pro": "CDI",
      "anciennete_pro": 36,
      "revenu_mensuel": 2800,
      "charges_mensuelles": 900,
      "credits_encours": 200,
      "annees_residence": 3,
 
      "montant_credit": 10000,
      "duree_credit": 60,
      "objet_credit": "Consommation",
 
      "taux_annuel": 0.035,          // optionnel
      "threshold": 0.5,              // optionnel
 
      "agent_adjustment": -0.10,     // optionnel [-0.30 ; +0.30]
      "agent_comment": "Garant",
 
      "use_guardrails": true,        // optionnel
      "max_debt_ratio_after": 0.45,   // optionnel
      "min_reste_a_vivre_after": 0,   // optionnel
 
      "debug": true                  // optionnel
    }
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return _json_error("Body JSON manquant", 400)
 
        # ------- Validations minimum
        revenu = _to_float(data.get("revenu_mensuel", 0), 0)
 
        # Cas métier : aucun revenu → refus automatique
        if revenu <= 0:
            return jsonify({
                "decision": 0,
                "reason": "Revenu mensuel nul ou inexistant",
                "risk_score_model": 1.0,
                "risk_score_adjusted": 1.0,
                "threshold": _to_float(data.get("threshold", DEFAULT_THRESHOLD), DEFAULT_THRESHOLD),
                "kpis": {
                    "revenu_mensuel": revenu
                }
            })
 
        # crédit demandé
        montant_credit = _to_float(data.get("montant_credit", 0), 0)
        duree_credit = _to_int(data.get("duree_credit", 0), 0)
        objet_credit = str(data.get("objet_credit", "")).strip()
 
        if montant_credit <= 0 or duree_credit <= 0 or objet_credit == "":
            return _json_error(
                "Champs crédit manquants ou invalides",
                400,
                required=["montant_credit (>0)", "duree_credit (>0)", "objet_credit (non vide)"]
            )
 
        # threshold
        threshold = _to_float(data.get("threshold", DEFAULT_THRESHOLD), DEFAULT_THRESHOLD)
        if not (0.0 <= threshold <= 1.0):
            return _json_error("threshold doit être compris entre 0 et 1", 400)
 
        # taux annuel (simulation mensualité)
        taux_annuel = _to_float(data.get("taux_annuel", DEFAULT_TAUX_ANNUEL), DEFAULT_TAUX_ANNUEL)
        taux_annuel = max(0.0, taux_annuel)
 
        # agent adjustment
        agent_adjustment = _to_float(data.get("agent_adjustment", 0.0), 0.0)
        agent_adjustment = _clamp(agent_adjustment, AGENT_ADJ_MIN, AGENT_ADJ_MAX)
        agent_comment = data.get("agent_comment")
 
        # autres infos métier
        charges = _to_float(data.get("charges_mensuelles", 0), 0)
        credits = _to_float(data.get("credits_encours", 0), 0)
 
        # mensualité estimée du crédit demandé
        mensualite_credit = _calc_mensualite(montant_credit, duree_credit, taux_annuel)
 
        # KPIs avant/après
        denom = max(revenu, 1.0)
        taux_endettement_before = (charges + credits) / denom
        reste_a_vivre_before = revenu - charges - credits
 
        taux_endettement_after = (charges + credits + mensualite_credit) / denom
        reste_a_vivre_after = revenu - charges - credits - mensualite_credit
 
        # features -> modèle
        X = _business_to_features(
            data,
            mensualite_credit=mensualite_credit,
            taux_endettement_after=taux_endettement_after,
            reste_a_vivre_after=reste_a_vivre_after
        )
 
        # score modèle = RISQUE (classe 1)
        risk_score_model = _predict_risk_score(X)
 
        # score ajusté par agent
        risk_score_adjusted = _clamp(risk_score_model + agent_adjustment, 0.0, 1.0)
 
        # décision finale : ACCEPTÉ si risque < threshold
        # Décision métier prioritaire
        if reste_a_vivre_after >= 800 and taux_endettement_after <= 0.35:
            decision = 1  # crédit accepté
        else:
            decision = 0  # crédit refusé
 
 
        # garde-fous (optionnels) basés sur APRÈS crédit
        use_guardrails = bool(data.get("use_guardrails", False))
        guardrail_reasons = []
        if use_guardrails:
            max_debt_ratio_after = _to_float(data.get("max_debt_ratio_after", DEFAULT_MAX_DEBT_RATIO_AFTER), DEFAULT_MAX_DEBT_RATIO_AFTER)
            min_reste_after = _to_float(data.get("min_reste_a_vivre_after", DEFAULT_MIN_RESTE_A_VIVRE_AFTER), DEFAULT_MIN_RESTE_A_VIVRE_AFTER)
 
            if taux_endettement_after > max_debt_ratio_after:
                guardrail_reasons.append(f"Taux d'endettement après crédit > {max_debt_ratio_after:.2f}")
            if reste_a_vivre_after < min_reste_after:
                guardrail_reasons.append(f"Reste à vivre après crédit < {min_reste_after:.0f}€")
 
            if guardrail_reasons:
                decision = 0  # refus forcé
 
        payload = {
            "model_type": MODEL_TYPE,
            "uses_scaler": USES_SCALER,
 
            # scores
            "risk_score_model": risk_score_model,
            "risk_score_adjusted": risk_score_adjusted,
            "threshold": threshold,
 
            # agent
            "agent_adjustment": agent_adjustment,
            "agent_comment": agent_comment,
 
            # décision
            "decision": decision,  # 1 accepté / 0 refus
 
            # crédit demandé
            "credit": {
                "montant_credit": montant_credit,
                "duree_credit": duree_credit,
                "objet_credit": objet_credit,
                "taux_annuel": taux_annuel,
                "mensualite_estimee": mensualite_credit
            },
 
            # kpis
            "kpis": {
                "taux_endettement_before": taux_endettement_before,
                "reste_a_vivre_before": reste_a_vivre_before,
                "taux_endettement_after": taux_endettement_after,
                "reste_a_vivre_after": reste_a_vivre_after
            }
        }
 
        if use_guardrails:
            payload["guardrails"] = {
                "enabled": True,
                "forced_refusal": bool(guardrail_reasons),
                "reasons": guardrail_reasons,
                "max_debt_ratio_after": _to_float(data.get("max_debt_ratio_after", DEFAULT_MAX_DEBT_RATIO_AFTER), DEFAULT_MAX_DEBT_RATIO_AFTER),
                "min_reste_a_vivre_after": _to_float(data.get("min_reste_a_vivre_after", DEFAULT_MIN_RESTE_A_VIVRE_AFTER), DEFAULT_MIN_RESTE_A_VIVRE_AFTER),
            }
 
        if bool(data.get("debug", False)) is True:
            payload["debug"] = {
                "first_30_features": X[0, :30].tolist()
            }
 
        return jsonify(payload)
 
    except AttributeError as e:
        return _json_error(str(e), 500)
    except ValueError as e:
        return _json_error("Valeurs invalides", 400, details=str(e))
    except Exception as e:
        return _json_error("Erreur interne serveur", 500, details=str(e))
 
 
# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("✅ API file:", __file__)
    print("✅ Model path:", MODEL_PATH)
    print("✅ Scaler path:", SCALER_PATH)
    #app.run(host="0.0.0.0", port=5000, debug=True)
