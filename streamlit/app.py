# frontend/streamlit_app.py
import streamlit as st
import requests
from typing import Dict, Any

# ==================== CONFIG PAGE ====================
st.set_page_config(
    page_title="Santander - Simulation de Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS (AMÉLIORÉ MAIS MÊME UI) ====================
PRIMARY = "#EC0000"
BORDER = "#E6E8F0"
BG = "#F5F6FA"
CARD = "#FFFFFF"
TXT = "#111827"
MUTED = "#6B7280"
SUCCESS_BG = "#EAF7EE"
SUCCESS_BORDER = "#22C55E"
DANGER_BG = "#FDECEC"
DANGER_BORDER = "#EF4444"
INFO_BG = "#F0F2F6"

st.markdown(f"""
<style>
.stApp {{
    background: {BG};
}}
.block-container {{
    max-width: 1200px;
    padding-top: 1.2rem;
}}
.main-header {{
    font-size: 2.6rem;
    color: {PRIMARY};
    text-align: center;
    margin-bottom: 1.2rem;
    margin-top: 3rem;
    font-weight: 900;
    letter-spacing: -0.5px;
}}
.step-header {{
    font-size: 1.6rem;
    color: {TXT};
    margin-top: 1.4rem;
    margin-bottom: 0.8rem;
    border-bottom: 2px solid {BORDER};
    padding-bottom: 0.5rem;
    font-weight: 800;
}}
.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,.04);
}}
.info-box {{
    background-color: {INFO_BG};
    padding: 1.1rem;
    border-radius: 14px;
    border-left: 6px solid {PRIMARY};
    margin: 1rem 0;
    color: {TXT};
}}
.success-box {{
    background-color: {SUCCESS_BG};
    color: {TXT};
    padding: 1.4rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.1rem;
    border: 2px solid {SUCCESS_BORDER};
}}
.danger-box {{
    background-color: {DANGER_BG};
    color: {TXT};
    padding: 1.4rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.1rem;
    border: 2px solid {DANGER_BORDER};
}}
.small-note {{
    font-size: 0.92rem;
    color: {MUTED};
}}
/* Boutons un peu plus modernes */
div.stButton > button {{
    border-radius: 14px;
    padding: .65rem 1rem;
    font-weight: 700;
}}
div.stButton > button[kind="primary"] {{
    background: {PRIMARY};
    border: 1px solid {PRIMARY};
}}
/* Sidebar plus clean */
section[data-testid="stSidebar"] {{
    background: #fff;
    border-right: 1px solid {BORDER};
}}
</style>
""", unsafe_allow_html=True)

# ==================== API CONFIG ====================
API_URL = "http://localhost:5000"

# ==================== SESSION STATE ====================
if "etape" not in st.session_state:
    st.session_state.etape = 0  # 0 = Accueil
if "donnees" not in st.session_state:
    st.session_state.donnees = {}

# ==================== API CALL ====================
def appeler_api_prediction(donnees: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "age": donnees.get("age"),
        "statut_pro": donnees.get("statut_pro"),
        "anciennete_pro": donnees.get("anciennete_pro"),
        "revenu_mensuel": donnees.get("revenu_mensuel"),
        "charges_mensuelles": donnees.get("charges_mensuelles"),
        "credits_encours": donnees.get("credits_encours"),
        "annees_residence": donnees.get("annees_residence"),

        # crédit demandé (OBLIGATOIRE)
        "montant_credit": donnees.get("montant_credit"),
        "duree_credit": donnees.get("duree_credit"),
        "objet_credit": donnees.get("objet_credit"),

        # optionnels
        "taux_annuel": donnees.get("taux_annuel", 0.035),
        "threshold": donnees.get("threshold", 0.5),
        "agent_adjustment": donnees.get("agent_adjustment", 0.0),
        "agent_comment": donnees.get("agent_comment", ""),
        "use_guardrails": donnees.get("use_guardrails", False),
        "max_debt_ratio_after": donnees.get("max_debt_ratio_after", 0.45),
        "min_reste_a_vivre_after": donnees.get("min_reste_a_vivre_after", 0),
        "debug": donnees.get("debug", False),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            try:
                err = response.json()
            except Exception:
                err = {"error": response.text}
            return {"success": False, "error": err, "status_code": response.status_code}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": {"error": f"Impossible de se connecter à l'API ({API_URL}). Lance Flask."}}
    except Exception as e:
        return {"success": False, "error": {"error": str(e)}}

# ==================== FALLBACK RAISON (si API n'envoie pas reason) ====================
def generer_raison_fallback(d: Dict[str, Any], api: Dict[str, Any]) -> str:
    decision = api.get("decision", 0)
    kpis = api.get("kpis", {}) or {}
    revenu = float(d.get("revenu_mensuel", 0) or 0)
    charges = float(d.get("charges_mensuelles", 0) or 0)
    credits = float(d.get("credits_encours", 0) or 0)
    mensualite = float(d.get("mensualite_estimee", 0) or 0)
    statut = str(d.get("statut_pro", "")).strip()

    taux_after = kpis.get("taux_endettement_after", None)
    reste_after = kpis.get("reste_a_vivre_after", None)

    if taux_after is None and revenu > 0:
        taux_after = (charges + credits + mensualite) / revenu
    if reste_after is None:
        reste_after = revenu - charges - credits - mensualite

    if decision == 1:
        return "Votre dossier est compatible avec nos critères."

    if revenu <= 0:
        return "Revenu mensuel nul ou inexistant."
    if taux_after is not None and taux_after > 0.45:
        return "Taux d’endettement après crédit trop élevé (> 45%)."
    if statut == "Sans emploi":
        return "Situation professionnelle jugée instable."
    if reste_after is not None and reste_after < 200:
        return "Reste à vivre après crédit insuffisant."

    return "Critères d’éligibilité non atteints pour ce montant/durée."

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🏦 Santander - Simulation de Crédit</h1>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### Navigation")

    etapes = {
        0: "🏁 Accueil",
        1: "👤 Informations Personnelles",
        2: "💼 Situation Professionnelle",
        3: "💰 Situation Financière",
        4: "📋 Détails du Crédit",
        5: "🎯 Résultat"
    }

    for num, titre in etapes.items():
        if num == st.session_state.etape:
            st.markdown(f"**➤ {titre}**")
        elif num < st.session_state.etape:
            st.markdown(f"✅ {titre}")
        else:
            st.markdown(f"⚪ {titre}")

    st.markdown("---")
    st.markdown("### 🧑‍💼 Paramètres Agent")

    threshold = st.slider(
        "Seuil décision (threshold)",
        0.0, 1.0,
        float(st.session_state.donnees.get("threshold", 0.5)),
        0.01
    )

    agent_adjustment = st.slider(
        "Ajustement agent",
        -0.30, 0.30,
        float(st.session_state.donnees.get("agent_adjustment", 0.0)),
        0.01
    )

    st.session_state.donnees.update({
        "threshold": threshold,
        "agent_adjustment": agent_adjustment,
    })

    st.markdown("---")
    if st.button("🔄 Recommencer", use_container_width=True):
        st.session_state.etape = 0
        st.session_state.donnees = {}
        st.rerun()

# ==================== PROGRESS ====================
if st.session_state.etape == 0:
    st.progress(0.0)
else:
    progress = (st.session_state.etape - 1) / 4
    st.progress(progress)
st.markdown(f"**Étape {st.session_state.etape} sur 5**" if st.session_state.etape else "**Accueil**")

# ==================== ACCUEIL ====================
if st.session_state.etape == 0:
    st.markdown("## Bienvenue dans la simulation de crédit !")
    st.markdown(
        "<span class='small-note'>Cliquez sur le bouton ci-dessous pour commencer votre simulation.</span>",
        unsafe_allow_html=True
    )
    st.write("")
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        if st.button("🚀 Commencer la simulation", type="primary", use_container_width=True):
            st.session_state.etape = 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ==================== ETAPE 1 ====================
if st.session_state.etape == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-header">👤 Informations Personnelles</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        nom = st.text_input("Nom *", value=st.session_state.donnees.get("nom", ""), placeholder="Dupont")
        prenom = st.text_input("Prénom *", value=st.session_state.donnees.get("prenom", ""), placeholder="Jean")
        age = st.number_input("Âge *", 18, 100, int(st.session_state.donnees.get("age", 30)))
    with col2:
        residence = st.selectbox(
            "Type de résidence *",
            ["Propriétaire", "Locataire", "Hébergé gratuitement", "Autre"],
            index=["Propriétaire", "Locataire", "Hébergé gratuitement", "Autre"].index(
                st.session_state.donnees.get("residence", "Locataire")
            )
        )
        annees_residence = st.number_input(
            "Années à l'adresse actuelle *",
            0, 50, int(st.session_state.donnees.get("annees_residence", 2))
        )

    st.markdown(
        "<div class='info-box'>ℹ️ Tous les champs marqués d’un * sont obligatoires.</div>",
        unsafe_allow_html=True
    )

    if st.button("Suivant ➡️", type="primary", use_container_width=True):
        if nom and prenom:
            st.session_state.donnees.update({
                "nom": nom,
                "prenom": prenom,
                "age": age,
                "residence": residence,
                "annees_residence": annees_residence
            })
            st.session_state.etape = 2
            st.rerun()
        else:
            st.error("❌ Veuillez remplir tous les champs obligatoires")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== ETAPE 2 ====================
elif st.session_state.etape == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-header">💼 Situation Professionnelle</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        statut_pro = st.selectbox(
            "Statut professionnel *",
            ["CDI", "CDD", "Intérimaire", "Indépendant", "Fonctionnaire", "Retraité", "Sans emploi", "Étudiant"],
            index=["CDI", "CDD", "Intérimaire", "Indépendant", "Fonctionnaire", "Retraité", "Sans emploi", "Étudiant"].index(
                st.session_state.donnees.get("statut_pro", "CDI")
            )
        )
        secteur = st.selectbox(
            "Secteur d'activité *",
            ["Agriculture", "Commerce", "Construction", "Éducation", "Finance", "Industrie",
             "Santé", "Services", "Technologies", "Transport", "Autre"],
            index=["Agriculture", "Commerce", "Construction", "Éducation", "Finance", "Industrie",
                   "Santé", "Services", "Technologies", "Transport", "Autre"].index(
                st.session_state.donnees.get("secteur", "Services")
            )
        )
    with col2:
        anciennete_pro = st.number_input(
            "Ancienneté professionnelle (en mois) *",
            0, 600, int(st.session_state.donnees.get("anciennete_pro", 24))
        )

    colA, colB = st.columns(2)
    with colA:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.etape = 1
            st.rerun()
    with colB:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.donnees.update({
                "statut_pro": statut_pro,
                "secteur": secteur,
                "anciennete_pro": anciennete_pro
            })
            st.session_state.etape = 3
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== ETAPE 3 ====================
elif st.session_state.etape == 3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-header">💰 Situation Financière</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        revenu_mensuel = st.number_input(
            "Revenu mensuel net (€) *",
            0, 50000, int(st.session_state.donnees.get("revenu_mensuel", 2000)),
            step=100
        )
        credits_encours = st.number_input(
            "Crédits en cours (mensualités) (€) *",
            0, 500000, int(st.session_state.donnees.get("credits_encours", 0)),
            step=100
        )
    with col2:
        charges_mensuelles = st.number_input(
            "Charges mensuelles (€) *",
            0, 10000, int(st.session_state.donnees.get("charges_mensuelles", 800)),
            step=50
        )

    st.markdown("---")
    reste_a_vivre = revenu_mensuel - credits_encours - charges_mensuelles
    taux_endettement = (credits_encours / revenu_mensuel * 100) if revenu_mensuel > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("💵 Reste à vivre", f"{reste_a_vivre:.0f} €")
    m2.metric("📈 Taux endettement (crédits)", f"{taux_endettement:.1f} %")
    m3.metric("💪 Capacité indicative", f"{max(0, revenu_mensuel*0.33-credits_encours):.0f} €/mois")

    colA, colB = st.columns(2)
    with colA:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.etape = 2
            st.rerun()
    with colB:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.donnees.update({
                "revenu_mensuel": revenu_mensuel,
                "credits_encours": credits_encours,
                "charges_mensuelles": charges_mensuelles
            })
            st.session_state.etape = 4
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== ETAPE 4 ====================
elif st.session_state.etape == 4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-header">📋 Détails du Crédit Demandé</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        montant_credit = st.number_input(
            "Montant du crédit demandé (€) *",
            1000, 500000, int(st.session_state.donnees.get("montant_credit", 10000)),
            step=1000
        )
        duree_credit = st.selectbox(
            "Durée du crédit (mois) *",
            [12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 300],
            index=[12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 300].index(
                int(st.session_state.donnees.get("duree_credit", 60))
            )
        )
    with col2:
        objet_credit = st.selectbox(
            "Objet du crédit *",
            ["Achat immobilier", "Travaux", "Véhicule", "Consommation", "Trésorerie", "Autre"],
            index=["Achat immobilier", "Travaux", "Véhicule", "Consommation", "Trésorerie", "Autre"].index(
                st.session_state.donnees.get("objet_credit", "Consommation")
            )
        )

    st.markdown("---")
    st.markdown("<b>💳 Simulation de mensualité</b>", unsafe_allow_html=True)
    taux_annuel = float(st.session_state.donnees.get("taux_annuel", 0.035))
    taux_mensuel = taux_annuel / 12.0
    n = duree_credit

    if taux_mensuel > 0:
        mensualite = montant_credit * (taux_mensuel * (1 + taux_mensuel) ** n) / ((1 + taux_mensuel) ** n - 1)
    else:
        mensualite = montant_credit / n

    st.session_state.donnees["mensualite_estimee"] = mensualite
    st.session_state.donnees["taux_annuel"] = taux_annuel

    c1, c2, c3 = st.columns(3)
    c1.metric("Mensualité estimée", f"{mensualite:.2f} €")
    c2.metric("Total à rembourser", f"{mensualite*n:.2f} €")
    c3.metric("Coût du crédit", f"{mensualite*n - montant_credit:.2f} €")

    colA, colB = st.columns(2)
    with colA:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.etape = 3
            st.rerun()
    with colB:
        if st.button("🎯 Lancer la Simulation", type="primary", use_container_width=True):
            st.session_state.donnees.update({
                "montant_credit": montant_credit,
                "duree_credit": duree_credit,
                "objet_credit": objet_credit,
            })
            st.session_state.etape = 5
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== ETAPE 5 ====================
elif st.session_state.etape == 5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="step-header">🎯 Résultat de votre Simulation</h2>', unsafe_allow_html=True)

    d = st.session_state.donnees
    res = appeler_api_prediction(d)

    if not res["success"]:
        st.error("❌ Erreur lors de l'appel à l'API /predict")
        st.json(res.get("error", {}))
        colA, colB = st.columns(2)
        with colA:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.etape = 4
                st.rerun()
        with colB:
            if st.button("🔄 Nouvelle simulation", use_container_width=True):
                st.session_state.etape = 0
                st.session_state.donnees = {}
                st.rerun()
        st.stop()

    api = res["data"]
    decision = api.get("decision", 0)

    # ✅ PRIORITÉ À reason venant de l'API
    raison = api.get("reason") or generer_raison_fallback(d, api)

    if decision == 1:
        st.markdown(f"""
        <div class="success-box">
        ✅ <b>DEMANDE ÉLIGIBLE</b><br><br>
        {raison}
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div class="danger-box">
        ❌ <b>DEMANDE NON ÉLIGIBLE</b><br><br>
        {raison}
        </div>
        """, unsafe_allow_html=True)

    # Récap
    st.markdown("---")
    st.markdown("### 📋 Récapitulatif")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**👤 Personnel**")
        st.write(f"Nom : {d.get('nom')} {d.get('prenom')}")
        st.write(f"Âge : {d.get('age')} ans")
        st.write(f"Résidence : {d.get('residence')}")
        st.write(f"Années à l'adresse : {d.get('annees_residence')}")
        st.markdown("**💼 Professionnel**")
        st.write(f"Statut : {d.get('statut_pro')}")
        st.write(f"Secteur : {d.get('secteur')}")
        st.write(f"Ancienneté : {d.get('anciennete_pro')} mois")

    with col2:
        st.markdown("**💰 Financier**")
        st.write(f"Revenu : {d.get('revenu_mensuel')} €")
        st.write(f"Charges : {d.get('charges_mensuelles')} €")
        st.write(f"Crédits en cours : {d.get('credits_encours')} €")
        st.write(f"Mensualité estimée : {d.get('mensualite_estimee'):.2f} €")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("⬅️ Modifier le crédit", use_container_width=True):
            st.session_state.etape = 4
            st.rerun()
    with colB:
        if st.button("🔄 Nouvelle simulation", use_container_width=True):
            st.session_state.etape = 0
            st.session_state.donnees = {}
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
