# frontend/streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
import json

# Configuration de la page
st.set_page_config(
    page_title="Santander - Simulation de Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #EC0000;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .step-header {
        font-size: 1.8rem;
        color: #EC0000;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #EC0000;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #EC0000;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        border: 2px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        border: 2px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration API
API_URL = "http://localhost:5000"

# Initialisation de session_state
if 'etape' not in st.session_state:
    st.session_state.etape = 1
if 'donnees' not in st.session_state:
    st.session_state.donnees = {}

def generer_features_aleatoires() -> Dict[str, float]:
    """
    Génère 200 features aléatoires pour la prédiction.
    En production, ces valeurs seraient calculées à partir des données du formulaire.
    """
    import numpy as np
    features = {}
    for i in range(200):
        # Génération de valeurs réalistes basées sur les statistiques du dataset
        if i < 50:
            features[f"var_{i}"] = np.random.normal(10, 3)
        elif i < 100:
            features[f"var_{i}"] = np.random.normal(5, 2)
        elif i < 150:
            features[f"var_{i}"] = np.random.normal(15, 4)
        else:
            features[f"var_{i}"] = np.random.normal(8, 2.5)
    return features

def appeler_api_prediction(features: Dict[str, float], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Appelle l'API de prédiction
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features, "threshold": threshold},
            timeout=10
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Erreur API: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Impossible de se connecter à l'API. Vérifiez qu'elle est démarrée."}
    except Exception as e:
        return {"success": False, "error": f"Erreur: {str(e)}"}

def calculer_score_risque(donnees: Dict) -> float:
    """
    Calcule un score de risque basé sur les données du formulaire
    (Ceci est une simplification - en production, le ML fait ce calcul)
    """
    score = 0.5  # Score de base
    
    # Facteurs positifs
    if donnees.get('statut_pro') == 'CDI':
        score += 0.15
    if donnees.get('anciennete_pro', 0) > 24:
        score += 0.1
    if donnees.get('annees_residence', 0) > 3:
        score += 0.05
    
    # Facteurs négatifs
    taux_endettement = donnees.get('credits_encours', 0) / max(donnees.get('revenu_mensuel', 1), 1)
    if taux_endettement > 0.33:
        score -= 0.2
    if donnees.get('charges_mensuelles', 0) / max(donnees.get('revenu_mensuel', 1), 1) > 0.5:
        score -= 0.15
    
    return max(0.0, min(1.0, score))

# ==================== INTERFACE PRINCIPALE ====================

st.markdown('<h1 class="main-header">🏦 Santander - Simulation de Crédit</h1>', unsafe_allow_html=True)

# Barre de progression
progress = (st.session_state.etape - 1) / 4
st.progress(progress)
st.markdown(f"**Étape {st.session_state.etape} sur 5**")

# Sidebar - Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Santander_Logo.svg/1200px-Santander_Logo.svg.png", width=200)
    st.markdown("### Navigation")
    
    etapes = {
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
    if st.button("🔄 Recommencer", use_container_width=True):
        st.session_state.etape = 1
        st.session_state.donnees = {}
        st.rerun()

# ==================== ÉTAPE 1 : INFORMATIONS PERSONNELLES ====================
if st.session_state.etape == 1:
    st.markdown('<h2 class="step-header">👤 Informations Personnelles</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        nom = st.text_input("Nom *", value=st.session_state.donnees.get('nom', ''), placeholder="Dupont")
        prenom = st.text_input("Prénom *", value=st.session_state.donnees.get('prenom', ''), placeholder="Jean")
        age = st.number_input("Âge *", min_value=18, max_value=100, value=st.session_state.donnees.get('age', 30))
    
    with col2:
        residence = st.selectbox(
            "Type de résidence *",
            ["Propriétaire", "Locataire", "Hébergé gratuitement", "Autre"],
            index=["Propriétaire", "Locataire", "Hébergé gratuitement", "Autre"].index(
                st.session_state.donnees.get('residence', 'Locataire')
            )
        )
        annees_residence = st.number_input(
            "Années à l'adresse actuelle *",
            min_value=0,
            max_value=50,
            value=st.session_state.donnees.get('annees_residence', 2)
        )
    
    st.markdown('<div class="info-box">ℹ️ <b>Informations importantes :</b><br>• Tous les champs marqués d\'un * sont obligatoires<br>• Vos données sont sécurisées et confidentielles<br>• La simulation est gratuite et sans engagement</div>', unsafe_allow_html=True)
    
    if st.button("Suivant ➡️", type="primary", use_container_width=True):
        if nom and prenom:
            st.session_state.donnees.update({
                'nom': nom,
                'prenom': prenom,
                'age': age,
                'residence': residence,
                'annees_residence': annees_residence
            })
            st.session_state.etape = 2
            st.rerun()
        else:
            st.error("❌ Veuillez remplir tous les champs obligatoires")

# ==================== ÉTAPE 2 : SITUATION PROFESSIONNELLE ====================
elif st.session_state.etape == 2:
    st.markdown('<h2 class="step-header">💼 Situation Professionnelle</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        statut_pro = st.selectbox(
            "Statut professionnel *",
            ["CDI", "CDD", "Intérimaire", "Indépendant", "Fonctionnaire", "Retraité", "Sans emploi", "Étudiant"],
            index=["CDI", "CDD", "Intérimaire", "Indépendant", "Fonctionnaire", "Retraité", "Sans emploi", "Étudiant"].index(
                st.session_state.donnees.get('statut_pro', 'CDI')
            )
        )
        
        secteur = st.selectbox(
            "Secteur d'activité *",
            ["Agriculture", "Commerce", "Construction", "Éducation", "Finance", "Industrie", 
             "Santé", "Services", "Technologies", "Transport", "Autre"],
            index=["Agriculture", "Commerce", "Construction", "Éducation", "Finance", "Industrie", 
                   "Santé", "Services", "Technologies", "Transport", "Autre"].index(
                st.session_state.donnees.get('secteur', 'Services')
            )
        )
    
    with col2:
        anciennete_pro = st.number_input(
            "Ancienneté professionnelle (en mois) *",
            min_value=0,
            max_value=600,
            value=st.session_state.donnees.get('anciennete_pro', 24),
            help="Nombre de mois dans votre emploi actuel"
        )
        
        st.markdown("### 📊 Indicateur de stabilité")
        if statut_pro == "CDI" and anciennete_pro >= 12:
            st.success("✅ Très bonne stabilité professionnelle")
        elif statut_pro in ["CDI", "Fonctionnaire"] and anciennete_pro >= 6:
            st.info("ℹ️ Bonne stabilité professionnelle")
        else:
            st.warning("⚠️ Stabilité professionnelle à renforcer")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.etape = 1
            st.rerun()
    
    with col_btn2:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.donnees.update({
                'statut_pro': statut_pro,
                'secteur': secteur,
                'anciennete_pro': anciennete_pro
            })
            st.session_state.etape = 3
            st.rerun()

# ==================== ÉTAPE 3 : SITUATION FINANCIÈRE ====================
elif st.session_state.etape == 3:
    st.markdown('<h2 class="step-header">💰 Situation Financière</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        revenu_mensuel = st.number_input(
            "Revenu mensuel net (€) *",
            min_value=0,
            max_value=50000,
            value=st.session_state.donnees.get('revenu_mensuel', 2000),
            step=100
        )
        
        credits_encours = st.number_input(
            "Crédits en cours (€) *",
            min_value=0,
            max_value=500000,
            value=st.session_state.donnees.get('credits_encours', 0),
            step=100,
            help="Montant total des mensualités de vos crédits actuels"
        )
    
    with col2:
        charges_mensuelles = st.number_input(
            "Charges mensuelles (€) *",
            min_value=0,
            max_value=10000,
            value=st.session_state.donnees.get('charges_mensuelles', 800),
            step=50,
            help="Loyer, assurances, abonnements, etc."
        )
    
    # Calcul du taux d'endettement
    st.markdown("---")
    st.markdown("### 📊 Analyse de votre capacité d'emprunt")
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    reste_a_vivre = revenu_mensuel - credits_encours - charges_mensuelles
    taux_endettement = (credits_encours / revenu_mensuel * 100) if revenu_mensuel > 0 else 0
    
    with col_metric1:
        st.metric("💵 Reste à vivre", f"{reste_a_vivre:.0f} €")
    
    with col_metric2:
        st.metric("📈 Taux d'endettement", f"{taux_endettement:.1f} %")
    
    with col_metric3:
        capacite = revenu_mensuel * 0.33 - credits_encours
        st.metric("💪 Capacité d'emprunt", f"{max(0, capacite):.0f} €/mois")
    
    # Indicateurs visuels
    if taux_endettement < 33:
        st.success("✅ Votre taux d'endettement est excellent (< 33%)")
    elif taux_endettement < 40:
        st.warning("⚠️ Votre taux d'endettement est élevé (33-40%)")
    else:
        st.error("❌ Votre taux d'endettement est trop élevé (> 40%)")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.etape = 2
            st.rerun()
    
    with col_btn2:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.donnees.update({
                'revenu_mensuel': revenu_mensuel,
                'credits_encours': credits_encours,
                'charges_mensuelles': charges_mensuelles
            })
            st.session_state.etape = 4
            st.rerun()

# ==================== ÉTAPE 4 : DÉTAILS DU CRÉDIT ====================
elif st.session_state.etape == 4:
    st.markdown('<h2 class="step-header">📋 Détails du Crédit Demandé</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        montant_credit = st.number_input(
            "Montant du crédit demandé (€) *",
            min_value=1000,
            max_value=500000,
            value=st.session_state.donnees.get('montant_credit', 10000),
            step=1000
        )
        
        duree_credit = st.selectbox(
            "Durée du crédit (mois) *",
            [12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 300],
            index=[12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 300].index(
                st.session_state.donnees.get('duree_credit', 60)
            )
        )
    
    with col2:
        objet_credit = st.selectbox(
            "Objet du crédit *",
            ["Achat immobilier", "Travaux", "Véhicule", "Consommation", "Trésorerie", "Autre"],
            index=["Achat immobilier", "Travaux", "Véhicule", "Consommation", "Trésorerie", "Autre"].index(
                st.session_state.donnees.get('objet_credit', 'Consommation')
            )
        )
    
    # Simulation de mensualité (taux fictif pour démo)
    st.markdown("---")
    st.markdown("### 💳 Simulation de la mensualité")
    
    taux_annuel = 0.035  # 3.5% (exemple)
    taux_mensuel = taux_annuel / 12
    n_mois = duree_credit
    
    if taux_mensuel > 0:
        mensualite = montant_credit * (taux_mensuel * (1 + taux_mensuel)**n_mois) / ((1 + taux_mensuel)**n_mois - 1)
    else:
        mensualite = montant_credit / n_mois
    
    cout_total = mensualite * n_mois
    cout_credit = cout_total - montant_credit
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        st.metric("💰 Mensualité estimée", f"{mensualite:.2f} €")
    
    with col_sim2:
        st.metric("💸 Coût total du crédit", f"{cout_credit:.2f} €")
    
    with col_sim3:
        st.metric("📊 Total à rembourser", f"{cout_total:.2f} €")
    
    # Vérification de la capacité
    revenu = st.session_state.donnees.get('revenu_mensuel', 0)
    credits_actuels = st.session_state.donnees.get('credits_encours', 0)
    
    nouveau_taux = ((credits_actuels + mensualite) / revenu * 100) if revenu > 0 else 100
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"**📊 Votre nouveau taux d'endettement serait de {nouveau_taux:.1f}%**")
    if nouveau_taux < 33:
        st.markdown("✅ Ce crédit est compatible avec votre situation financière")
    elif nouveau_taux < 40:
        st.markdown("⚠️ Ce crédit représente un endettement important")
    else:
        st.markdown("❌ Ce crédit risque de dépasser votre capacité de remboursement")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.etape = 3
            st.rerun()
    
    with col_btn2:
        if st.button("🎯 Lancer la Simulation", type="primary", use_container_width=True):
            st.session_state.donnees.update({
                'montant_credit': montant_credit,
                'duree_credit': duree_credit,
                'objet_credit': objet_credit,
                'mensualite_estimee': mensualite
            })
            st.session_state.etape = 5
            st.rerun()
# ==================== ÉTAPE 5 : RÉSULTAT ====================
elif st.session_state.etape == 5:
    st.markdown('<h2 class="step-header">🎯 Résultat de votre Simulation</h2>', unsafe_allow_html=True)

    d = st.session_state.donnees

    # ================= RÈGLES MÉTIER (DÉCISION RÉELLE) =================
    revenu = d.get("revenu_mensuel", 0)
    charges = d.get("charges_mensuelles", 0)
    credits = d.get("credits_encours", 0)
    mensualite = d.get("mensualite_estimee", 0)
    statut = d.get("statut_pro", "")

    decision = 1
    raisons = []

    if revenu <= 0:
        decision = 0
        raisons.append("Revenu invalide")

    taux_endettement = (charges + credits + mensualite) / revenu if revenu > 0 else 1

    if taux_endettement > 0.45:
        decision = 0
        raisons.append("Taux d’endettement trop élevé (> 45%)")

    if statut == "Sans emploi":
        decision = 0
        raisons.append("Situation professionnelle instable")

    # ================= AFFICHAGE =================
    st.markdown("---")

    if decision == 1:
        st.markdown("""
        <div class="success-box">
        ✅ <b>DEMANDE ÉLIGIBLE</b><br><br>
        Votre situation est compatible avec l’octroi du crédit.
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="danger-box">
        ❌ <b>DEMANDE NON ÉLIGIBLE</b><br><br>
        {}
        </div>
        """.format("<br>".join(f"• {r}" for r in raisons)), unsafe_allow_html=True)

    # ================= RÉCAP =================
    st.markdown("### 📋 Récapitulatif")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**👤 Personnel**")
        st.write(f"Nom : {d.get('nom')} {d.get('prenom')}")
        st.write(f"Âge : {d.get('age')} ans")
        st.write(f"Résidence : {d.get('residence')}")

        st.markdown("**💼 Professionnel**")
        st.write(f"Statut : {d.get('statut_pro')}")
        st.write(f"Secteur : {d.get('secteur')}")
        st.write(f"Ancienneté : {d.get('anciennete_pro')} mois")

    with col2:
        st.markdown("**💰 Financier**")
        st.write(f"Revenu : {revenu} €")
        st.write(f"Charges : {charges} €")
        st.write(f"Crédits : {credits} €")
        st.write(f"Mensualité estimée : {mensualite:.2f} €")

    st.markdown("---")

    if st.button("🔄 Nouvelle simulation", use_container_width=True):
        st.session_state.etape = 1
        st.session_state.donnees = {}
        st.rerun()
