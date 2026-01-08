# frontend/streamlit_app.py
import streamlit as st
import math
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
APP_TITLE = "Simulation d’éligibilité au crédit"
APP_SUBTITLE = "Formulaire multi-étapes • Résultat immédiat"

PREDICT_URL = "http://localhost:5000/predict"

FEATURES = [f"var_{i}" for i in range(200)]

st.set_page_config(page_title=APP_TITLE, page_icon="💳", layout="wide")

# =========================
# STYLE (simple + propre)
# =========================
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px;
    }
    .muted {opacity: .75;}
    .step-pill {
        display:inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        margin-right: 8px;
        font-size: 14px;
    }
    .step-active {
        border-color: rgba(255, 99, 99, 0.60);
        box-shadow: 0 0 0 2px rgba(255,99,99,0.15) inset;
    }
    .big-title {font-size: 28px; font-weight: 750; margin: 0;}
    .sub-title {margin-top: 6px; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# STATE INIT
# =========================
def init_state():
    defaults = {
        "step": 0,

        # Personnel
        "full_name": "Jean Dupont",
        "age": 30,
        "city": "Paris",
        "housing_status": "Locataire",
        "years_at_address": 2,

        # Professionnel
        "employment_status": "CDI",
        "sector": "",
        "job_seniority": 3,

        # Financier
        "monthly_income": 2000,
        "monthly_charges": 600,
        "existing_loans": 0,
        "loan_amount": 5000,
        "loan_duration_months": 12,

        # UI
        "threshold": 0.50,
        "result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# HELPERS
# =========================
def set_step(i: int):
    st.session_state.step = int(i)

def step_header():
    steps = ["Personnel", "Professionnel", "Financier", "Révision"]
    cols = st.columns([1, 1, 1, 1])
    for i, name in enumerate(steps):
        active = "step-pill step-active" if st.session_state.step == i else "step-pill"
        with cols[i]:
            if st.button(name, use_container_width=True, key=f"nav_{i}"):
                set_step(i)
    st.markdown("---")

def business_rules_decision():
    """
    Règles simples 'crédit' pour éviter les cas incohérents.
    Retourne (eligible, message_court)
    """
    income = float(st.session_state.monthly_income)
    charges = float(st.session_state.monthly_charges)
    amount = float(st.session_state.loan_amount)
    months = float(st.session_state.loan_duration_months)

    if income <= 0:
        return False, "Revenu mensuel manquant."

    if months <= 0:
        return False, "Durée invalide."

    # mensualité estimée (sans intérêts)
    monthly_payment = amount / months
    debt_ratio = (charges + monthly_payment) / income

    if debt_ratio > 0.45:
        return False, "Capacité de remboursement insuffisante."

    # Exemple : sans emploi + revenu faible
    if st.session_state.employment_status == "Sans emploi" and income < 1200:
        return False, "Situation trop instable pour cette simulation."

    return True, "Profil cohérent pour une analyse."

def build_payload_from_form():
    """
    On doit envoyer var_0..var_199.
    Comme le dataset Santander est anonymisé, on simule un mapping.
    """
    payload = {f: 0.0 for f in FEATURES}

    payload["var_0"] = float(st.session_state.age)
    payload["var_1"] = float(st.session_state.years_at_address)
    payload["var_2"] = float(st.session_state.job_seniority)
    payload["var_3"] = float(st.session_state.monthly_income)
    payload["var_4"] = float(st.session_state.monthly_charges)
    payload["var_5"] = float(st.session_state.existing_loans)
    payload["var_6"] = float(st.session_state.loan_amount)
    payload["var_7"] = float(st.session_state.loan_duration_months)

    status_map = {"CDI": 3, "CDD": 2, "Freelance": 2, "Étudiant": 1, "Sans emploi": 0}
    payload["var_8"] = float(status_map.get(st.session_state.employment_status, 0))

    return payload

def call_prediction(payload: dict):
    r = requests.post(PREDICT_URL, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

# =========================
# SIDEBAR (config)
# =========================
with st.sidebar:
    st.markdown("## Configuration")
    st.session_state.threshold = st.slider(
        "Seuil de décision",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.threshold),
        step=0.01,
        key="threshold_slider"
    )
    st.caption("Plus le seuil est élevé, plus la décision devient stricte.")
    st.markdown("---")
    if st.button("Réinitialiser la simulation"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# =========================
# HEADER
# =========================
st.markdown(f"<p class='big-title'>💳 {APP_TITLE}</p>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
st.write("")
step_header()

# =========================
# STEP 1 — PERSONNEL
# =========================
if st.session_state.step == 0:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Informations personnelles")
    c1, c2 = st.columns(2)

    with c1:
        st.text_input("Nom complet", key="full_name")
        st.number_input("Âge", min_value=18, max_value=99, step=1, key="age")
        st.text_input("Ville", key="city")

    with c2:
        st.selectbox("Résidence", ["Locataire", "Propriétaire", "Hébergé"], key="housing_status")
        st.number_input("Années à l’adresse actuelle", min_value=0, max_value=50, step=1, key="years_at_address")

    st.write("")
    colA, colB = st.columns([1, 1])
    with colB:
        if st.button("Suivant →", use_container_width=True):
            set_step(1)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# STEP 2 — PROFESSIONNEL
# =========================
elif st.session_state.step == 1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Situation professionnelle")

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Statut", ["CDI", "CDD", "Freelance", "Étudiant", "Sans emploi"], key="employment_status")
        st.number_input("Ancienneté (années)", min_value=0, max_value=50, step=1, key="job_seniority")

    with c2:
        st.text_input("Secteur (optionnel)", key="sector")

    st.write("")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("← Retour", use_container_width=True):
            set_step(0)
    with colB:
        if st.button("Suivant →", use_container_width=True):
            set_step(2)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# STEP 3 — FINANCIER
# =========================
elif st.session_state.step == 2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Informations financières")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Revenu mensuel net (€)", min_value=0, step=50, key="monthly_income")
        st.number_input("Charges mensuelles (€)", min_value=0, step=50, key="monthly_charges")
    with c2:
        st.number_input("Crédits en cours (nombre)", min_value=0, step=1, key="existing_loans")
        st.number_input("Montant demandé (€)", min_value=0, step=100, key="loan_amount")
    with c3:
        st.number_input("Durée (mois)", min_value=1, step=1, key="loan_duration_months")

    st.write("")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("← Retour", use_container_width=True):
            set_step(1)
    with colB:
        if st.button("Suivant →", use_container_width=True):
            set_step(3)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# STEP 4 — REVIEW & RESULT
# =========================
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Révision")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("**Personnel**")
        st.write(f"• Nom : {st.session_state.full_name}")
        st.write(f"• Âge : {st.session_state.age} ans")
        st.write(f"• Ville : {st.session_state.city}")
        st.write(f"• Résidence : {st.session_state.housing_status} — {st.session_state.years_at_address} an(s)")

        st.write("")
        st.markdown("**Professionnel**")
        st.write(f"• Statut : {st.session_state.employment_status}")
        st.write(f"• Ancienneté : {st.session_state.job_seniority} an(s)")
        if st.session_state.sector.strip():
            st.write(f"• Secteur : {st.session_state.sector}")

    with right:
        st.markdown("**Financier**")
        st.write(f"• Revenus : {st.session_state.monthly_income} €")
        st.write(f"• Charges : {st.session_state.monthly_charges} €")
        st.write(f"• Crédits en cours : {st.session_state.existing_loans}")
        st.write(f"• Demande : {st.session_state.loan_amount} € sur {st.session_state.loan_duration_months} mois")

        st.write("")
        st.markdown("### Résultat")

        if st.button("✅ Calculer mon éligibilité", use_container_width=True):
            # 1) Décision "banque" (ne pas afficher 'ok', juste le verdict)
            eligible_rules, msg = business_rules_decision()
            if not eligible_rules:
                st.session_state.result = {
                    "decision": 0,
                    "proba_target_1": None,
                    "message": msg
                }
            else:
                # 2) Appel modèle (en arrière-plan, message user-friendly)
                payload = build_payload_from_form()
                try:
                    res = call_prediction(payload)
                    proba = float(res.get("proba_target_1", 0.0))
                    threshold = float(st.session_state.threshold)
                    decision = 1 if proba >= threshold else 0
                    st.session_state.result = {
                        "decision": decision,
                        "proba_target_1": proba,
                        "message": None
                    }
                except Exception:
                    st.session_state.result = {
                        "decision": 0,
                        "proba_target_1": None,
                        "message": "Service indisponible. Réessayez."
                    }

        # Affichage résultat utilisateur final
        if st.session_state.result:
            decision = st.session_state.result["decision"]
            proba = st.session_state.result["proba_target_1"]
            msg = st.session_state.result["message"]

            if decision == 1:
                st.success("✅ Éligible")
                if proba is not None:
                    st.caption(f"Score de la simulation : {proba:.3f}")
            else:
                st.error("❌ Non éligible")
                if msg:
                    st.caption(msg)
                elif proba is not None:
                    st.caption(f"Score de la simulation : {proba:.3f}")

    st.write("")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("← Retour", use_container_width=True):
            set_step(2)
    with colB:
        if st.button("Nouvelle simulation", use_container_width=True):
            st.session_state.result = None
            set_step(0)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='muted'>Note : le modèle s’appuie sur des variables anonymisées (var_0…var_199). "
                "Cette interface réalise une simulation en mappant quelques champs vers ces variables.</div>",
                unsafe_allow_html=True)

# =========================
# (OPTION) Upload CSV (1 ligne var_0..var_199)
# Tu peux le mettre où tu veux si tu en as besoin.
# =========================
with st.expander("Importer un fichier (optionnel)", expanded=False):
    st.caption("Vous pouvez importer un fichier contenant une seule ligne avec les colonnes var_0..var_199.")
    file = st.file_uploader("Importer un CSV", type=["csv"])
    if file is not None:
        df_up = pd.read_csv(file)
        missing_cols = [c for c in FEATURES if c not in df_up.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes : {missing_cols[:10]} ... ({len(missing_cols)} au total)")
        else:
            st.success("Fichier valide.")
            row0 = df_up.iloc[0]
            st.write("Aperçu (premières variables) :", row0[FEATURES[:10]].to_dict())
            if st.button("Utiliser cette ligne pour calculer"):
                payload = {f: float(row0[f]) for f in FEATURES}
                try:
                    res = call_prediction(payload)
                    proba = float(res.get("proba_target_1", 0.0))
                    decision = 1 if proba >= float(st.session_state.threshold) else 0
                    st.session_state.result = {
                        "decision": decision,
                        "proba_target_1": proba,
                        "message": None
                    }
                    st.success("Calcul terminé. Allez à l’étape Révision.")
                except Exception as e:
                    st.error(f"Erreur de calcul : {e}")
