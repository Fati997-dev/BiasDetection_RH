# ============================================================
#   ÉTAPE 1 — PRÉTRAITEMENT DES DONNÉES
#   Projet : Explicabilité & Détection des Biais RH
# ============================================================

# ── BIBLIOTHÈQUES NÉCESSAIRES ────────────────────────────────
import pandas as pd                          # manipulation des données
import numpy as np                           # calculs numériques
from sklearn.preprocessing import LabelEncoder   # encodage texte → chiffres
from sklearn.model_selection import train_test_split  # séparation train/test
import pickle                                # sauvegarder les données traitées
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#   1. CHARGEMENT DU DATASET
# ============================================================
print("=" * 60)
print("  ÉTAPE 1 — PRÉTRAITEMENT DES DONNÉES")
print("=" * 60)

print("\n📂 Chargement du dataset...")

df = pd.read_excel(
    'Dataset_RH_Organise_v2.xlsx',   # ← mettre le chemin de ton fichier
    sheet_name='📋 Candidats',
    header=1                          # la ligne 1 est le vrai en-tête
)

print(f"   ✅ Dataset chargé avec succès !")
print(f"   → {df.shape[0]} candidats")
print(f"   → {df.shape[1]} colonnes")
print(f"\n   Aperçu des 3 premières lignes :")
print(df[['Nom','Prénom','Genre','Âge','Score Test (/100)','Décision Finale']].head(3).to_string())

# ============================================================
#   2. SÉLECTION DES COLONNES UTILES
# ============================================================
print("\n" + "-" * 60)
print("  2. SÉLECTION DES COLONNES")
print("-" * 60)

# Colonnes qu'on donne au modèle ML (les "features")
features = [
    'Genre',                   # variable de biais potentiel
    'Âge',                     # variable de biais potentiel
    'Statut Marital',          # variable de biais potentiel
    'Expérience (ans)',        # compétence réelle
    'Diplôme',                 # compétence réelle
    'Nb Langues',              # compétence réelle
    'Secteur',                 # secteur visé
    'Score Test (/100)',       # compétence réelle (objectif)
    'Score Entretien (/100)',  # compétence réelle (subjectif)
    'Nb Entretiens',           # processus de recrutement
    'Distance (km)',           # proximité géographique
    'Salaire Attendu (MAD)',   # prétention salariale
    'Déjà Recruté',            # historique
    'Durée Poste (mois)',      # historique
]

# La colonne qu'on veut prédire (la "cible")
cible = 'Décision Finale'

print("\n   ✅ Colonnes UTILISÉES pour le modèle :")
for i, f in enumerate(features, 1):
    print(f"      {i:2d}. {f}")

print("\n   ❌ Colonnes EXCLUES du modèle :")
exclus = {
    'ID'               : "Simple numéro — sans signification",
    'Nom / Prénom'     : "Identité personnelle — inutile pour le ML",
    'Langues'          : "Texte libre — remplacé par Nb Langues",
    'Date Recrutement' : "Trop de valeurs manquantes (N/A)",
    'Encore en Poste'  : "Trop de valeurs manquantes (N/A)",
}
for col, raison in exclus.items():
    print(f"      ✗ {col:22s} → {raison}")

# Créer le dataframe avec seulement les colonnes utiles
df_model = df[features + [cible]].copy()

# ============================================================
#   3. NETTOYAGE — VALEURS MANQUANTES
# ============================================================
print("\n" + "-" * 60)
print("  3. NETTOYAGE DES VALEURS MANQUANTES")
print("-" * 60)

print("\n   Valeurs manquantes par colonne (avant nettoyage) :")
manquantes = df_model.isnull().sum()
manquantes_cols = manquantes[manquantes > 0]
if len(manquantes_cols) == 0:
    print("   → Aucune valeur manquante trouvée ✅")
else:
    for col, nb in manquantes_cols.items():
        print(f"   → {col} : {nb} valeurs manquantes")

# Remplir Durée Poste manquante par 0
# (si N/A = le candidat n'a jamais été recruté = 0 mois d'expérience en poste)
df_model['Durée Poste (mois)'] = df_model['Durée Poste (mois)'].fillna(0)

print(f"\n   Valeurs manquantes après nettoyage : "
      f"{df_model.isnull().sum().sum()} ✅")

# ============================================================
#   4. ENCODAGE — TEXTE → CHIFFRES
# ============================================================
print("\n" + "-" * 60)
print("  4. ENCODAGE DES VARIABLES")
print("-" * 60)

print("\n   A) Encodage Manuel (colonnes binaires) :")

# Genre : Homme=1, Femme=0
df_model['Genre'] = df_model['Genre'].map({'Homme': 1, 'Femme': 0})
print("   ✅ Genre          : Homme → 1  |  Femme → 0")

# Déjà Recruté : Oui=1, Non=0
df_model['Déjà Recruté'] = df_model['Déjà Recruté'].map({'Oui': 1, 'Non': 0})
print("   ✅ Déjà Recruté   : Oui → 1    |  Non → 0")

# Cible : Retenu=1, Rejeté=0
df_model[cible] = df_model[cible].map({'Retenu': 1, 'Rejeté': 0})
print("   ✅ Décision Finale: Retenu → 1 |  Rejeté → 0")

print("\n   B) Encodage Automatique (colonnes multi-valeurs) :")

encoders = {}  # on sauvegarde les encodeurs pour les réutiliser plus tard

# Statut Marital
le_statut = LabelEncoder()
df_model['Statut Marital'] = le_statut.fit_transform(
    df_model['Statut Marital'].astype(str))
encoders['Statut Marital'] = le_statut
classes_statut = dict(zip(le_statut.classes_,
                          le_statut.transform(le_statut.classes_)))
print(f"   ✅ Statut Marital  : {classes_statut}")

# Diplôme
le_diplome = LabelEncoder()
df_model['Diplôme'] = le_diplome.fit_transform(
    df_model['Diplôme'].astype(str))
encoders['Diplôme'] = le_diplome
classes_diplome = dict(zip(le_diplome.classes_,
                            le_diplome.transform(le_diplome.classes_)))
print(f"   ✅ Diplôme         : {classes_diplome}")

# Secteur
le_secteur = LabelEncoder()
df_model['Secteur'] = le_secteur.fit_transform(
    df_model['Secteur'].astype(str))
encoders['Secteur'] = le_secteur
classes_secteur = dict(zip(le_secteur.classes_,
                            le_secteur.transform(le_secteur.classes_)))
print(f"   ✅ Secteur         : {classes_secteur}")

# ============================================================
#   5. SÉPARATION TRAIN / TEST
# ============================================================
print("\n" + "-" * 60)
print("  5. SÉPARATION TRAIN / TEST")
print("-" * 60)

X = df_model[features]   # les données d'entrée
y = df_model[cible]      # la colonne à prédire

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% pour le test
    random_state=42,   # résultats reproductibles
    stratify=y         # même proportion Retenu/Rejeté dans train ET test
)

print(f"\n   TRAIN → {len(X_train)} candidats (80%)")
print(f"   → Retenus : {y_train.sum()}  |  Rejetés : {(y_train==0).sum()}")
print(f"\n   TEST  → {len(X_test)} candidats (20%)")
print(f"   → Retenus : {y_test.sum()}   |  Rejetés : {(y_test==0).sum()}")

# ============================================================
#   6. APERÇU FINAL
# ============================================================
print("\n" + "-" * 60)
print("  6. APERÇU FINAL DU DATASET PRÉTRAITÉ")
print("-" * 60)
print("\n   5 premières lignes de X_train (données d'entraînement) :")
print(X_train.head(5).to_string())
print(f"\n   Cible correspondante (y_train) : {list(y_train[:5])}")
print("   (1 = Retenu, 0 = Rejeté)")

# ============================================================
#   7. SAUVEGARDE
# ============================================================
print("\n" + "-" * 60)
print("  7. SAUVEGARDE DES DONNÉES PRÉTRAITÉES")
print("-" * 60)

# Sauvegarder tout dans un fichier .pkl pour l'étape 2
with open('data_pretraitee.pkl', 'wb') as f:
    pickle.dump({
        'X_train':   X_train,
        'X_test':    X_test,
        'y_train':   y_train,
        'y_test':    y_test,
        'features':  features,
        'encoders':  encoders,
        'df_original': df
    }, f)

print("\n   ✅ Fichier 'data_pretraitee.pkl' sauvegardé !")
print("      → Il sera utilisé directement à l'Étape 2 (Modèle ML)")

print("\n" + "=" * 60)
print("  ✅ ÉTAPE 1 TERMINÉE — DONNÉES PRÊTES POUR LE MODÈLE !")
print("=" * 60)