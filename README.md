# 🔍 BiasDetect RH
## Explicabilité & Détection des Biais dans les Modèles de Recrutement basés sur l'Apprentissage Automatique

---

## 📋 Description du projet

**BiasDetect RH** est un système d'intelligence artificielle qui permet de :
- Construire un modèle de recrutement basé sur le Machine Learning
- Détecter automatiquement les biais cachés (genre, âge, statut marital)
- Expliquer les décisions du modèle pour chaque candidat
- Visualiser les résultats dans un dashboard interactif pour les responsables RH

---

## 🎯 Objectif

> Prouver par les données que certains modèles de recrutement discriminent les candidats non pas selon leurs compétences, mais selon des critères injustes comme le genre, l'âge ou le statut marital.

---

## ⚠️ Biais détectés

| Biais | Groupe favorisé | Écart |
|---|---|---|
| **Genre** | Hommes (61.8%) vs Femmes (50.0%) | 11.8 points |
| **Statut Marital** | Mariés (60.0%) vs Célibataires (51.2%) | 8.8 points |
| **Âge** | Pénalité après 40 ans pour les femmes | Variable |

---

## 🗂️ Structure du projet

```
📁 BiasDetect_RH/
│
├── 📊 Dataset
│   └── Dataset_RH_Organise_v2.xlsx     ← 500 candidats, 21 colonnes
│
├── 🐍 Scripts Python
│   ├── etape1_pretraitement.py          ← Nettoyage & encodage des données
│   ├── etape2_modele_ML.py              ← Modèle Random Forest
│   ├── etape3_detection_biais.py        ← Analyse & visualisation des biais
│   ├── etape4_dashboard.py              ← Dashboard Streamlit interactif
│   └── etape5_multi_agent.py            ← Système multi-agent complet
│
├── 💾 Fichiers générés
│   ├── data_pretraitee.pkl              ← Données prêtes pour le ML
│   ├── modele_random_forest.pkl         ← Modèle entraîné sauvegardé
│   ├── rapport_final.txt                ← Rapport automatique des biais
│   ├── resultats_modele_RF.png          ← Graphiques du modèle
│   └── resultats_biais.png              ← Graphiques des biais
│
└── 📖 README.md                         ← Ce fichier
```

---

## 🚀 Installation & Lancement

### 1. Prérequis
- Python 3.11 (recommandé)
- Windows / Mac / Linux

### 2. Créer l'environnement virtuel
```bash
python -m venv env
```

### 3. Activer l'environnement
```bash
# Windows
env\Scripts\activate

# Mac / Linux
source env/bin/activate
```

### 4. Installer les bibliothèques
```bash
pip install pandas scikit-learn openpyxl matplotlib seaborn streamlit
```

### 5. Lancer les étapes dans l'ordre

```bash
# Étape 1 — Prétraitement
python etape1_pretraitement.py

# Étape 2 — Modèle ML
python etape2_modele_ML.py

# Étape 3 — Détection des biais
python etape3_detection_biais.py

# Étape 4 — Dashboard interactif
streamlit run etape4_dashboard.py

# Étape 5 — Système multi-agent (tout en une seule commande)
python etape5_multi_agent.py
```

---

## 🤖 Système Multi-Agent

Le projet implémente un système de **5 agents autonomes** qui travaillent en pipeline :

```
┌─────────────────┐
│  Agent 1        │  Lit le fichier Excel
│  Collecte       │  → 500 candidats chargés
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 2        │  Encode texte → chiffres
│  Nettoyage      │  → 14 features prêtes
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 3        │  Entraîne Random Forest
│  Modèle ML      │  → AUC : 0.599 | F1 : 0.614
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 4        │  Mesure les écarts par groupe
│  Détection      │  → 3 biais détectés
│  Biais          │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 5        │  Génère rapport_final.txt
│  Rapport        │  → Recommandations RH
└─────────────────┘
```

---

## 📊 Dataset

| Colonne | Description | Type |
|---|---|---|
| Genre | Homme / Femme | Biais ⚠️ |
| Âge | 22 à 55 ans | Biais ⚠️ |
| Statut Marital | Célibataire / Marié(e) / Divorcé(e) | Biais ⚠️ |
| Diplôme | Bac → Doctorat | Mérite |
| Score Test | Note technique /100 | Mérite |
| Score Entretien | Note entretien /100 | Mérite |
| Expérience | Années d'expérience | Mérite |
| Distance | Domicile → Bureau (km) | Neutre |
| Salaire Attendu | Prétention salariale (MAD) | Neutre |
| Décision Finale | Retenu / Rejeté | **Cible** 🎯 |

---

## 🧠 Modèle utilisé

| Paramètre | Valeur |
|---|---|
| Algorithme | Random Forest Classifier |
| Nombre d'arbres | 100 |
| Profondeur max | 8 niveaux |
| AUC-ROC | 0.599 |
| F1 Score | 0.614 |

---

## 🖥️ Dashboard — Pages disponibles

| Page | Description |
|---|---|
| 🏠 Accueil | KPIs globaux + résumé des biais |
| 📊 Vue Générale | Statistiques filtrables |
| ⚠️ Détection des Biais | Analyse Genre, Âge, Statut Marital |
| 🔍 Analyser un Candidat | Prédiction individuelle + explication |
| 📈 Performance du Modèle | AUC, F1, Matrice de confusion |

---

## 📚 Bibliothèques utilisées

| Bibliothèque | Usage |
|---|---|
| `pandas` | Manipulation des données |
| `scikit-learn` | Modèle ML + prétraitement |
| `matplotlib` | Graphiques statiques |
| `seaborn` | Heatmaps et visualisations |
| `streamlit` | Dashboard interactif |
| `openpyxl` | Lecture des fichiers Excel |
| `pickle` | Sauvegarde du modèle |

---

## 💡 Recommandations finales

Suite à l'analyse des biais détectés, nous recommandons aux responsables RH de :

1. **Supprimer** le genre et le statut marital des critères de sélection
2. **Baser** les décisions uniquement sur : Score Test, Expérience, Diplôme
3. **Auditer** régulièrement les décisions du modèle
4. **Former** les recruteurs aux biais inconscients
5. **Utiliser** ce dashboard comme outil de contrôle continu

---

## 👨‍💻 Projet réalisé dans le cadre de

> **Explicabilité et détection des biais dans les modèles de recrutement basés sur l'apprentissage automatique**
> Compétences visées : ML, Deep Learning Frameworks, Système Multi-Agent

---

*BiasDetect RH — Pour un recrutement plus juste et transparent* 🎯