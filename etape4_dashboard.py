# ============================================================
#   ÉTAPE 4 — DASHBOARD STREAMLIT
#   Projet : BiasDetect_RH
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#   CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="BiasDetect RH",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style CSS personnalisé ────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #4FC3F7;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #AAAAAA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-box {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.3rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .kpi-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .biais-box {
        background: #C00000;
        border-left: 6px solid #FF4444;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #FFFFFF !important;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .biais-box b {
        color: #FFD700 !important;
        font-size: 1.05rem;
    }
    .ok-box {
        background: #1D6A3A;
        border-left: 6px solid #44FF88;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #FFFFFF !important;
        font-size: 0.95rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #4FC3F7;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#   CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data
def charger_donnees():
    with open('data_pretraitee.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('modele_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    return data, model

try:
    data, model = charger_donnees()
    df       = data['df_original']
    X_test   = data['X_test']
    y_test   = data['y_test']
    X_train  = data['X_train']
    y_train  = data['y_train']
    features = data['features']
except:
    st.error("❌ Fichiers introuvables ! Lance d'abord etape1 et etape2.")
    st.stop()

BLEU  = "#1F4E79"
ROUGE = "#C00000"
VERT  = "#1D6A3A"
GRIS  = "#F2F2F2"

# ============================================================
#   SIDEBAR — NAVIGATION
# ============================================================
st.sidebar.image("https://img.icons8.com/color/96/diversity.png", width=80)
st.sidebar.title("BiasDetect RH")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil",
     "📊 Vue Générale",
     "⚠️ Détection des Biais",
     "🔍 Analyser un Candidat",
     "📈 Performance du Modèle"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset :** {len(df)} candidats")
st.sidebar.markdown(f"**Features :** {len(features)} variables")
taux = (df['Décision Finale'] == 'Retenu').mean()
st.sidebar.markdown(f"**Taux global :** {taux:.1%} retenus")

# ============================================================
#   PAGE 1 — ACCUEIL
# ============================================================
if page == "🏠 Accueil":
    st.markdown('<div class="main-title">🔍 BiasDetect RH</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explicabilité & Détection des Biais dans les Modèles de Recrutement</div>', unsafe_allow_html=True)

    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value">{len(df)}</div>
            <div class="kpi-label">Candidats analysés</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        retenus = (df['Décision Finale'] == 'Retenu').sum()
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value">{retenus}</div>
            <div class="kpi-label">Candidats retenus</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value">{len(features)}</div>
            <div class="kpi-label">Variables analysées</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value">3</div>
            <div class="kpi-label">Biais détectés</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ Biais détectés dans ce dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="biais-box">
            <b>⚠️ Biais Genre</b><br>
            Hommes : 61.8% retenus<br>
            Femmes : 50.0% retenus<br>
            <b>Écart : 11.8 points</b>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="biais-box">
            <b>⚠️ Biais Âge</b><br>
            Femmes > 40 ans pénalisées<br>
            Taux : 42.3% seulement<br>
            <b>Double pénalité détectée</b>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="biais-box">
            <b>⚠️ Biais Statut Marital</b><br>
            Mariés : 60.0% retenus<br>
            Célibataires : 51.2%<br>
            <b>Écart : 8.8 points</b>
        </div>""", unsafe_allow_html=True)

# ============================================================
#   PAGE 2 — VUE GÉNÉRALE
# ============================================================
elif page == "📊 Vue Générale":
    st.markdown('<div class="section-title">📊 Vue Générale du Dataset</div>', unsafe_allow_html=True)

    # Filtres
    st.markdown("#### 🔧 Filtres")
    col1, col2, col3 = st.columns(3)
    with col1:
        genre_filtre = st.multiselect("Genre", df['Genre'].unique(),
                                       default=list(df['Genre'].unique()))
    with col2:
        diplome_filtre = st.multiselect("Diplôme", df['Diplôme'].unique(),
                                         default=list(df['Diplôme'].unique()))
    with col3:
        secteur_filtre = st.multiselect("Secteur", df['Secteur'].unique(),
                                         default=list(df['Secteur'].unique()))

    df_filtre = df[
        df['Genre'].isin(genre_filtre) &
        df['Diplôme'].isin(diplome_filtre) &
        df['Secteur'].isin(secteur_filtre)
    ]

    st.markdown(f"**{len(df_filtre)} candidats** correspondent aux filtres")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        taux = df_filtre.groupby('Genre')['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100)
        bars = ax.bar(taux.index, taux.values, width=0.4,
                      color=[BLEU if g=='Homme' else ROUGE for g in taux.index],
                      edgecolor='white')
        for bar, val in zip(bars, taux.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')
        ax.set_title("Taux de recrutement par Genre", fontweight='bold')
        ax.set_ylabel("% Retenu"); ax.set_ylim(0, 80)
        ax.set_facecolor(GRIS)
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        taux_d = df_filtre.groupby('Diplôme')['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100).sort_values()
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(taux_d)))
        bars = ax.barh(taux_d.index, taux_d.values,
                       color=colors, edgecolor='white')
        for bar, val in zip(bars, taux_d.values):
            ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontweight='bold', fontsize=9)
        ax.set_title("Taux de recrutement par Diplôme", fontweight='bold')
        ax.set_xlabel("% Retenu")
        ax.set_facecolor(GRIS)
        st.pyplot(fig); plt.close()

    # Tableau des données
    st.markdown("#### 📋 Données filtrées")
    cols_afficher = ['Nom','Prénom','Genre','Âge','Statut Marital',
                     'Diplôme','Score Test (/100)','Score Entretien (/100)',
                     'Secteur','Décision Finale']
    st.dataframe(
        df_filtre[cols_afficher].reset_index(drop=True),
        use_container_width=True,
        height=300
    )

# ============================================================
#   PAGE 3 — DÉTECTION DES BIAIS
# ============================================================
elif page == "⚠️ Détection des Biais":
    st.markdown('<div class="section-title">⚠️ Détection des Biais</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Genre", "Âge", "Statut Marital", "Biais Croisé"])

    with tab1:
        st.markdown("### ⚠️ Biais de Genre")
        taux = df.groupby('Genre')['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100)
        ecart = abs(taux['Homme'] - taux['Femme'])
        col1, col2, col3 = st.columns(3)
        col1.metric("Hommes retenus", f"{taux['Homme']:.1f}%", "+11.8%")
        col2.metric("Femmes retenues", f"{taux['Femme']:.1f}%", "-11.8%")
        col3.metric("Écart détecté", f"{ecart:.1f} pts", "⚠️ Biais")

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(taux.index, taux.values, width=0.4,
                      color=[BLEU, ROUGE], edgecolor='white')
        for bar, val in zip(bars, taux.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=14)
        moy = (df['Décision Finale']=='Retenu').mean()*100
        ax.axhline(y=moy, color='gray', linestyle='--',
                   label=f'Moyenne : {moy:.1f}%')
        ax.set_ylim(0, 80); ax.legend()
        ax.set_facecolor(GRIS)
        ax.set_title("Taux de recrutement Homme vs Femme", fontweight='bold', color=ROUGE)
        st.pyplot(fig); plt.close()
        st.error(f"⚠️ Les hommes ont {ecart:.1f}% de chances supplémentaires d'être retenus à compétences égales !")

    with tab2:
        st.markdown("### ⚠️ Biais d'Âge")
        df['Tranche Âge'] = pd.cut(df['Âge'], bins=[20,30,40,55],
                                    labels=['20-30 ans','31-40 ans','41-55 ans'])
        taux_a = df.groupby('Tranche Âge', observed=True)['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [VERT, BLEU, ROUGE]
        bars = ax.bar(taux_a.index.astype(str), taux_a.values,
                      width=0.5, color=colors, edgecolor='white')
        for bar, val in zip(bars, taux_a.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=13)
        ax.set_ylim(0, 80); ax.set_facecolor(GRIS)
        ax.set_title("Taux de recrutement par tranche d'âge", fontweight='bold', color=ROUGE)
        st.pyplot(fig); plt.close()
        st.warning("⚠️ Les candidats de 41-55 ans montrent un taux différent selon le genre !")

    with tab3:
        st.markdown("### ⚠️ Biais de Statut Marital")
        taux_s = df.groupby('Statut Marital')['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors_s = [BLEU, ROUGE, VERT]
        bars = ax.bar(taux_s.index, taux_s.values,
                      width=0.5, color=colors_s, edgecolor='white')
        for bar, val in zip(bars, taux_s.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=13)
        ax.set_ylim(0, 80); ax.set_facecolor(GRIS)
        ax.set_title("Taux de recrutement par Statut Marital", fontweight='bold', color=ROUGE)
        st.pyplot(fig); plt.close()
        st.error("⚠️ Les personnes mariées ont 8.8% de chances supplémentaires d'être retenues !")

    with tab4:
        st.markdown("### ⚠️ Biais Croisé — Genre × Âge")
        pivot = df.groupby(['Genre','Tranche Âge'], observed=True)['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100).unstack()

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                    ax=ax, linewidths=2, linecolor='white',
                    annot_kws={'size':14,'weight':'bold'},
                    vmin=30, vmax=80)
        ax.set_title("% Retenu par Genre et Tranche d'Âge",
                     fontweight='bold', color=ROUGE)
        st.pyplot(fig); plt.close()
        st.error("⚠️ Pire cas : Femmes de 20-30 ans → seulement 41.9% retenues !")

# ============================================================
#   PAGE 4 — ANALYSER UN CANDIDAT
# ============================================================
elif page == "🔍 Analyser un Candidat":
    st.markdown('<div class="section-title">🔍 Analyser un Candidat</div>', unsafe_allow_html=True)
    st.info("Remplis les informations du candidat pour obtenir une prédiction et l'explication des facteurs.")

    # ── Nom & Prénom (affichage uniquement) ──────────────────
    col_nom, col_prenom = st.columns(2)
    with col_nom:
        nom_candidat    = st.text_input("📝 Nom", placeholder="Ex : Benali")
    with col_prenom:
        prenom_candidat = st.text_input("📝 Prénom", placeholder="Ex : Mohamed")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Informations personnelles**")
        genre    = st.selectbox("Genre", ["Homme", "Femme"])
        age      = st.slider("Âge", 20, 60, 30)
        statut   = st.selectbox("Statut Marital",
                                 ["Célibataire", "Marié(e)", "Divorcé(e)"])

    with col2:
        st.markdown("**🎓 Formation & Compétences**")
        diplome   = st.selectbox("Diplôme",
                                  ["Bac","Bac+2","Bac+3","Bac+5","Doctorat"])
        experience = st.slider("Expérience (ans)", 0, 35, 5)
        nb_langues = st.slider("Nombre de langues", 1, 4, 2)

    with col3:
        st.markdown("**💼 Poste & Scores**")
        secteur   = st.selectbox("Secteur",
                                  ["Informatique","Finance","Marketing","RH","Commercial"])
        score_test = st.slider("Score Test (/100)", 0, 100, 70)
        score_ent  = st.slider("Score Entretien (/100)", 0, 100, 65)

    col4, col5, col6 = st.columns(3)
    with col4:
        nb_ent   = st.slider("Nombre d'entretiens", 1, 5, 2)
    with col5:
        distance = st.slider("Distance domicile-bureau (km)", 1, 80, 20)
    with col6:
        salaire  = st.number_input("Salaire attendu (MAD/an)",
                                    20000, 120000, 45000, step=1000)

    st.markdown("**📋 Historique**")
    col7, col8 = st.columns(2)
    with col7:
        deja   = st.selectbox("Déjà recruté ?", ["Non","Oui"])
    with col8:
        duree  = st.slider("Durée en poste (mois)", 0, 84, 0)

    # ── Encodage ──────────────────────────────────────────
    enc_statut  = {'Célibataire':0,'Divorcé(e)':1,'Marié(e)':2}
    enc_diplome = {'Bac':0,'Bac+2':1,'Bac+3':2,'Bac+5':3,'Doctorat':4}
    enc_secteur = {'Commercial':0,'Finance':1,'Informatique':2,'Marketing':3,'RH':4}

    candidat = pd.DataFrame([{
        'Genre':                  1 if genre=='Homme' else 0,
        'Âge':                    age,
        'Statut Marital':         enc_statut[statut],
        'Expérience (ans)':       experience,
        'Diplôme':                enc_diplome[diplome],
        'Nb Langues':             nb_langues,
        'Secteur':                enc_secteur[secteur],
        'Score Test (/100)':      score_test,
        'Score Entretien (/100)': score_ent,
        'Nb Entretiens':          nb_ent,
        'Distance (km)':          distance,
        'Salaire Attendu (MAD)':  salaire,
        'Déjà Recruté':           1 if deja=='Oui' else 0,
        'Durée Poste (mois)':     duree,
    }])

    if st.button("🔍 Analyser ce candidat", type="primary"):
        proba    = model.predict_proba(candidat)[0, 1]
        decision = "✅ Retenu" if proba >= 0.5 else "❌ Rejeté"
        couleur  = "success" if proba >= 0.5 else "error"

        st.markdown("---")

        # Afficher le nom du candidat s'il est renseigné
        nom_complet = f"{prenom_candidat} {nom_candidat}".strip()
        if nom_complet:
            st.markdown(f"### 📊 Résultat pour : **{nom_complet}**")
        else:
            st.markdown("### 📊 Résultat")

        col1, col2, col3 = st.columns(3)
        col1.metric("Décision", decision)
        col2.metric("Probabilité", f"{proba:.0%}")
        col3.metric("Confiance",
                    "Élevée" if abs(proba-0.5) > 0.2 else "Faible")

        # Barre de probabilité
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh([""], [proba], color=VERT if proba>=0.5 else ROUGE,
                height=0.5)
        ax.barh([""], [1-proba], left=[proba], color="#EEEEEE", height=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)
        ax.set_xlim(0,1); ax.set_xlabel("Probabilité d'être retenu")
        ax.text(proba/2, 0, f"{proba:.0%}", ha='center',
                va='center', fontweight='bold', color='white', fontsize=12)
        ax.set_facecolor(GRIS)
        st.pyplot(fig); plt.close()

        # Facteurs influents
        st.markdown("### 🔍 Facteurs qui influencent la décision")
        moyennes = X_train.mean()
        facteurs = []
        labels_lisibles = {
            'Genre':                  f"Genre : {genre}",
            'Âge':                    f"Âge : {age} ans",
            'Statut Marital':         f"Statut : {statut}",
            'Score Test (/100)':      f"Score Test : {score_test}/100",
            'Score Entretien (/100)': f"Score Entretien : {score_ent}/100",
            'Expérience (ans)':       f"Expérience : {experience} ans",
            'Distance (km)':          f"Distance : {distance} km",
            'Diplôme':                f"Diplôme : {diplome}",
        }
        for feat in features[:8]:
            val  = candidat[feat].values[0]
            moy  = moyennes[feat]
            diff = val - moy
            facteurs.append({'label': labels_lisibles.get(feat, feat),
                             'diff': diff})

        facteurs_df = pd.DataFrame(facteurs).sort_values('diff')
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_f = [VERT if d > 0 else ROUGE for d in facteurs_df['diff']]
        bars = ax.barh(facteurs_df['label'], facteurs_df['diff'],
                       color=colors_f, edgecolor='white', height=0.6)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_title("Impact par rapport à la moyenne des candidats",
                     fontweight='bold')
        ax.set_facecolor(GRIS)
        ax.legend(handles=[
            mpatches.Patch(color=VERT,  label='Au-dessus de la moyenne ✅'),
            mpatches.Patch(color=ROUGE, label='En-dessous de la moyenne ⚠️')
        ], fontsize=9)
        st.pyplot(fig); plt.close()

        # Alertes biais
        st.markdown("### ⚠️ Alertes Biais Détectées")
        biais_detectes = []
        if genre == 'Femme':
            biais_detectes.append("⚠️ **Genre** : Femme → pénalisée de -11.8% en moyenne")
        if age > 40:
            biais_detectes.append("⚠️ **Âge** : Plus de 40 ans → pénalité détectée")
        if statut == 'Divorcé(e)':
            biais_detectes.append("⚠️ **Statut Marital** : Divorcé(e) → légèrement pénalisé")
        if statut == 'Célibataire':
            biais_detectes.append("⚠️ **Statut Marital** : Célibataire → moins favorisé que les mariés")

        if biais_detectes:
            for b in biais_detectes:
                st.error(b)
        else:
            st.success("✅ Aucun biais majeur détecté pour ce profil")

# ============================================================
#   PAGE 5 — PERFORMANCE DU MODÈLE
# ============================================================
elif page == "📈 Performance du Modèle":
    st.markdown('<div class="section-title">📈 Performance du Modèle</div>', unsafe_allow_html=True)

    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    f1      = f1_score(y_test, y_pred)
    acc     = (y_pred == y_test).mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC",   f"{auc:.3f}")
    col2.metric("F1 Score",  f"{f1:.3f}")
    col3.metric("Accuracy",  f"{acc:.1%}")
    col4.metric("Candidats testés", len(y_test))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matrice de Confusion")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Rejeté','Retenu'],
                    yticklabels=['Rejeté','Retenu'],
                    linewidths=2, linecolor='white',
                    annot_kws={'size':16,'weight':'bold'})
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Courbe ROC")
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=BLEU, lw=2.5,
                label=f'Modèle (AUC={auc:.3f})')
        ax.plot([0,1],[0,1], color='gray', lw=1.5,
                linestyle='--', label='Aléatoire (0.5)')
        ax.fill_between(fpr, tpr, alpha=0.1, color=BLEU)
        ax.set_xlabel("Taux Faux Positifs")
        ax.set_ylabel("Taux Vrais Positifs")
        ax.legend(); ax.set_facecolor(GRIS)
        st.pyplot(fig); plt.close()

    # Importance des variables
    st.markdown("#### Importance des Variables")
    importances = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=True)
    biais_vars = ['Genre', 'Âge', 'Statut Marital']
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [ROUGE if v in biais_vars else BLEU for v in importances.index]
    ax.barh(importances.index, importances.values,
            color=colors, edgecolor='white', height=0.6)
    ax.set_facecolor(GRIS)
    ax.legend(handles=[
        mpatches.Patch(color=ROUGE, label='⚠️ Variable de biais'),
        mpatches.Patch(color=BLEU,  label='Variable de mérite')
    ], fontsize=9)
    ax.set_title("Importance des Variables — Random Forest",
                 fontweight='bold', color=BLEU)
    st.pyplot(fig); plt.close()

    st.info(f"""
    **Interprétation du modèle :**
    - AUC = {auc:.3f} → Le modèle est **au-dessus du hasard** (0.5)
    - Pour un dataset de 500 candidats, c'est un résultat **acceptable**
    - L'objectif principal est la **détection des biais**, pas la performance maximale
    """)