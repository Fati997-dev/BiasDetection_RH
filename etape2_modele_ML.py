# ============================================================
#   ÉTAPE 2 — MODÈLE RANDOM FOREST
#   Projet : BiasDetect_RH
# ============================================================

# ── BIBLIOTHÈQUES ────────────────────────────────────────────
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#   1. CHARGEMENT DES DONNÉES PRÉTRAITÉES (Étape 1)
# ============================================================
print("=" * 60)
print("  ÉTAPE 2 — MODÈLE RANDOM FOREST")
print("=" * 60)

print("\n📂 Chargement des données prétraitées...")
with open('data_pretraitee.pkl', 'rb') as f:
    data = pickle.load(f)

X_train  = data['X_train']
X_test   = data['X_test']
y_train  = data['y_train']
y_test   = data['y_test']
features = data['features']

print(f"   ✅ Données chargées !")
print(f"   → Train : {len(X_train)} candidats")
print(f"   → Test  : {len(X_test)} candidats")
print(f"   → Features : {len(features)} colonnes")

# ============================================================
#   2. CRÉATION DU MODÈLE RANDOM FOREST
# ============================================================
print("\n" + "-" * 60)
print("  2. CRÉATION DU MODÈLE")
print("-" * 60)

# Random Forest = une forêt de 100 arbres de décision
# Chaque arbre vote → la majorité gagne
model = RandomForestClassifier(
    n_estimators=100,      # 100 arbres dans la forêt
    max_depth=8,           # profondeur max de chaque arbre
    min_samples_split=10,  # min 10 candidats pour diviser un nœud
    random_state=42,       # résultats reproductibles
    class_weight='balanced' # équilibrer Retenu/Rejeté
)

print("\n   Paramètres du modèle :")
print("   → n_estimators    : 100 arbres")
print("   → max_depth       : 8 niveaux maximum")
print("   → min_samples_split: 10 candidats minimum")
print("   → class_weight    : balanced (équilibré)")

# ============================================================
#   3. ENTRAÎNEMENT
# ============================================================
print("\n" + "-" * 60)
print("  3. ENTRAÎNEMENT DU MODÈLE")
print("-" * 60)

print("\n   ⏳ Entraînement en cours...")
model.fit(X_train, y_train)
print("   ✅ Modèle entraîné avec succès !")

# ============================================================
#   4. PRÉDICTIONS
# ============================================================
print("\n" + "-" * 60)
print("  4. PRÉDICTIONS SUR LES 100 CANDIDATS DE TEST")
print("-" * 60)

y_pred  = model.predict(X_test)          # 0 ou 1
y_proba = model.predict_proba(X_test)[:1, 1]  # probabilité entre 0 et 1

# Afficher quelques exemples
print("\n   Exemples de prédictions (10 premiers candidats) :")
print(f"   {'#':<5} {'Réel':<12} {'Prédit':<12} {'Probabilité':<12} {'Résultat'}")
print("   " + "-" * 55)
for i in range(10):
    reel   = "Retenu" if list(y_test)[i] == 1 else "Rejeté"
    predit = "Retenu" if y_pred[i] == 1 else "Rejeté"
    prob   = model.predict_proba(X_test)[i, 1]
    ok     = "✅ Correct" if list(y_test)[i] == y_pred[i] else "❌ Erreur"
    print(f"   {i+1:<5} {reel:<12} {predit:<12} {prob:.2f}         {ok}")

# ============================================================
#   5. ÉVALUATION DU MODÈLE
# ============================================================
print("\n" + "-" * 60)
print("  5. ÉVALUATION DU MODÈLE")
print("-" * 60)

# AUC-ROC
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# F1 Score
f1 = f1_score(y_test, y_pred)

# Validation croisée (5 blocs)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

print(f"\n   📊 MÉTRIQUES DE PERFORMANCE :")
print(f"   ┌─────────────────────────────────────┐")
print(f"   │  AUC-ROC          : {auc:.3f}           │")
print(f"   │  F1 Score         : {f1:.3f}           │")
print(f"   │  F1 Validation CV : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  │")
print(f"   └─────────────────────────────────────┘")

print(f"\n   Interprétation de l'AUC = {auc:.3f} :")
if auc >= 0.80:
    print("   → 🟢 Excellent — le modèle est très performant")
elif auc >= 0.70:
    print("   → 🟡 Bon — le modèle est correct")
elif auc >= 0.60:
    print("   → 🟠 Moyen — le modèle apprend mais peut mieux faire")
else:
    print("   → 🔴 Faible — dataset trop petit ou biais trop forts")

print(f"\n   Rapport détaillé :")
print(classification_report(y_test, y_pred,
      target_names=['Rejeté (0)', 'Retenu (1)']))

# ============================================================
#   6. IMPORTANCE DES VARIABLES (Détection des biais)
# ============================================================
print("\n" + "-" * 60)
print("  6. IMPORTANCE DES VARIABLES")
print("-" * 60)

importances = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n   Classement des variables par importance :")
print(f"   {'Rang':<5} {'Variable':<25} {'Importance':<12} {'Biais ?'}")
print("   " + "-" * 55)
biais_vars = ['Genre', 'Âge', 'Statut Marital']
for rang, (var, imp) in enumerate(importances.items(), 1):
    biais = "⚠️  OUI" if var in biais_vars else "—"
    barre = "█" * int(imp * 100)
    print(f"   {rang:<5} {var:<25} {imp:.4f}       {biais}")

# ============================================================
#   7. GRAPHIQUES
# ============================================================
print("\n" + "-" * 60)
print("  7. GÉNÉRATION DES GRAPHIQUES")
print("-" * 60)

BLEU  = "#1F4E79"
ROUGE = "#C00000"
VERT  = "#1D6A3A"
GRIS  = "#F2F2F2"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "BiasDetect_RH — Résultats du Modèle Random Forest",
    fontsize=14, fontweight='bold', color=BLEU, y=0.98
)
fig.patch.set_facecolor('#FAFAFA')

# ── Graphique 1 : Importance des variables ────────────────
ax1 = axes[0, 0]
imp_sorted = importances.sort_values()
colors = [ROUGE if v in biais_vars else BLEU for v in imp_sorted.index]
bars = ax1.barh(imp_sorted.index, imp_sorted.values,
                color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, imp_sorted.values):
    ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8)
ax1.set_title("Importance des Variables", fontweight='bold', color=BLEU)
ax1.set_xlabel("Importance")
ax1.set_facecolor(GRIS)
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(color=ROUGE, label='⚠️ Variable de biais'),
    Patch(color=BLEU,  label='Variable de mérite')
], fontsize=8)

# ── Graphique 2 : Matrice de confusion ───────────────────
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Rejeté','Retenu'],
            yticklabels=['Rejeté','Retenu'],
            linewidths=2, linecolor='white',
            annot_kws={'size': 16, 'weight': 'bold'})
ax2.set_title("Matrice de Confusion", fontweight='bold', color=BLEU)
ax2.set_xlabel("Prédit"); ax2.set_ylabel("Réel")

# ── Graphique 3 : Courbe ROC ─────────────────────────────
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
ax3.plot(fpr, tpr, color=BLEU, lw=2.5, label=f'Modèle (AUC = {auc:.3f})')
ax3.plot([0,1],[0,1], color='gray', lw=1.5,
         linestyle='--', label='Aléatoire (AUC = 0.5)')
ax3.fill_between(fpr, tpr, alpha=0.1, color=BLEU)
ax3.set_title("Courbe ROC", fontweight='bold', color=BLEU)
ax3.set_xlabel("Taux Faux Positifs")
ax3.set_ylabel("Taux Vrais Positifs")
ax3.legend(fontsize=10)
ax3.set_facecolor(GRIS)

# ── Graphique 4 : Biais Genre ─────────────────────────────
ax4 = axes[1, 1]
df_orig = data['df_original']
taux_genre = df_orig.groupby('Genre')['Décision Finale'].apply(
    lambda x: (x == 'Retenu').mean() * 100
)
bars4 = ax4.bar(taux_genre.index, taux_genre.values, width=0.4,
                color=[BLEU if g=='Homme' else ROUGE for g in taux_genre.index],
                edgecolor='white')
for bar, val in zip(bars4, taux_genre.values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center',
             fontweight='bold', fontsize=13)
moyenne = (df_orig['Décision Finale'] == 'Retenu').mean() * 100
ax4.axhline(y=moyenne, color='gray', linestyle='--',
            linewidth=1.5, label=f'Moyenne : {moyenne:.1f}%')
ax4.set_title("⚠️ Biais Détecté : Genre", fontweight='bold', color=ROUGE)
ax4.set_ylabel("% Retenu")
ax4.set_ylim(0, 80)
ax4.legend(fontsize=9)
ax4.set_facecolor(GRIS)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('resultats_modele_RF.png', dpi=150,
            bbox_inches='tight', facecolor='#FAFAFA')
print("\n   ✅ Graphique sauvegardé : resultats_modele_RF.png")

# ============================================================
#   8. SAUVEGARDE DU MODÈLE
# ============================================================
print("\n" + "-" * 60)
print("  8. SAUVEGARDE DU MODÈLE")
print("-" * 60)

with open('modele_random_forest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n   ✅ Modèle sauvegardé : modele_random_forest.pkl")
print("      → Sera utilisé à l'Étape 3 (SHAP)")

print("\n" + "=" * 60)
print("  ✅ ÉTAPE 2 TERMINÉE — MODÈLE PRÊT !")
print("=" * 60)