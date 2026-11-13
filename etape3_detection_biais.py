# ============================================================
#   ÉTAPE 3 — DÉTECTION DES BIAIS (sans SHAP)
#   Projet : BiasDetect_RH
#   Méthode : Permutation Importance + Analyse des biais
# ============================================================

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#   1. CHARGEMENT
# ============================================================
print("=" * 60)
print("  ÉTAPE 3 — DÉTECTION DES BIAIS")
print("=" * 60)

print("\n📂 Chargement des données et du modèle...")
with open('data_pretraitee.pkl', 'rb') as f:
    data = pickle.load(f)
with open('modele_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

X_train   = data['X_train']
X_test    = data['X_test']
y_train   = data['y_train']
y_test    = data['y_test']
features  = data['features']
df        = data['df_original']

print("   ✅ Modèle et données chargés !")

# ============================================================
#   2. IMPORTANCE DES VARIABLES (Permutation)
# ============================================================
print("\n" + "-" * 60)
print("  2. IMPORTANCE DES VARIABLES PAR PERMUTATION")
print("-" * 60)
print("\n   ⏳ Calcul en cours...")

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc'
)

imp_df = pd.DataFrame({
    'Variable':   features,
    'Importance': perm.importances_mean,
    'Std':        perm.importances_std
}).sort_values('Importance', ascending=False).reset_index(drop=True)

biais_vars = ['Genre', 'Âge', 'Statut Marital']

print("\n   Classement final des variables :")
print(f"   {'Rang':<5} {'Variable':<25} {'Importance':<12} {'Biais ?'}")
print("   " + "-" * 55)
for i, row in imp_df.iterrows():
    biais = "⚠️  BIAIS DÉTECTÉ" if row['Variable'] in biais_vars else "—"
    print(f"   {i+1:<5} {row['Variable']:<25} {row['Importance']:.4f}       {biais}")

# ============================================================
#   3. ANALYSE DES BIAIS PAR GROUPE
# ============================================================
print("\n" + "-" * 60)
print("  3. ANALYSE DES BIAIS PAR GROUPE")
print("-" * 60)

# Biais Genre
print("\n   ⚠️  BIAIS 1 — GENRE :")
taux_genre = df.groupby('Genre')['Décision Finale'].apply(
    lambda x: (x == 'Retenu').mean() * 100).round(1)
for genre, taux in taux_genre.items():
    barre = "█" * int(taux / 2)
    print(f"   {genre:<10} : {taux:>5.1f}%  {barre}")
ecart_genre = abs(taux_genre['Homme'] - taux_genre['Femme'])
print(f"   → Écart Homme/Femme : {ecart_genre:.1f} points ⚠️")

# Biais Âge
print("\n   ⚠️  BIAIS 2 — ÂGE :")
df['Tranche Âge'] = pd.cut(df['Âge'], bins=[20,30,40,55],
                            labels=['20-30 ans','31-40 ans','41-55 ans'])
taux_age = df.groupby('Tranche Âge', observed=True)['Décision Finale'].apply(
    lambda x: (x == 'Retenu').mean() * 100).round(1)
for age, taux in taux_age.items():
    barre = "█" * int(taux / 2)
    print(f"   {str(age):<12} : {taux:>5.1f}%  {barre}")

# Biais Statut Marital
print("\n   ⚠️  BIAIS 3 — STATUT MARITAL :")
taux_statut = df.groupby('Statut Marital')['Décision Finale'].apply(
    lambda x: (x == 'Retenu').mean() * 100).round(1)
for statut, taux in taux_statut.items():
    barre = "█" * int(taux / 2)
    print(f"   {str(statut):<15} : {taux:>5.1f}%  {barre}")

# Biais croisé Genre × Âge
print("\n   ⚠️  BIAIS CROISÉ — GENRE × ÂGE :")
taux_croise = df.groupby(['Genre', 'Tranche Âge'],
                          observed=True)['Décision Finale'].apply(
    lambda x: (x == 'Retenu').mean() * 100).round(1)
print(taux_croise.to_string())

# ============================================================
#   4. EXPLICATION INDIVIDUELLE
# ============================================================
print("\n" + "-" * 60)
print("  4. EXPLICATION INDIVIDUELLE — 3 EXEMPLES")
print("-" * 60)

def expliquer_candidat(idx, X_test, y_test, model, features, df):
    """Explique pourquoi un candidat est retenu ou rejeté"""
    candidat = X_test.iloc[idx]
    proba    = model.predict_proba(X_test.iloc[[idx]])[0, 1]
    decision = "Retenu" if proba >= 0.5 else "Rejeté"
    reel     = "Retenu" if list(y_test)[idx] == 1 else "Rejeté"

    print(f"\n   Candidat #{idx+1}")
    print(f"   Décision réelle  : {reel}")
    print(f"   Décision modèle  : {decision} (probabilité : {proba:.0%})")
    print(f"   Facteurs clés :")

    # Comparer chaque feature à la moyenne
    moyennes = X_test.mean()
    for feat in ['Score Test (/100)', 'Score Entretien (/100)',
                 'Expérience (ans)', 'Genre', 'Âge', 'Statut Marital']:
        val  = candidat[feat]
        moy  = moyennes[feat]
        diff = val - moy
        if feat == 'Genre':
            val_txt = "Homme" if val == 1 else "Femme"
            impact  = "→ Avantage biais" if val == 1 else "→ Désavantage biais ⚠️"
        elif feat == 'Statut Marital':
            val_txt = {0:'Célibataire', 1:'Divorcé(e)', 2:'Marié(e)'}.get(int(val), str(val))
            impact  = "→ Avantage biais" if val == 2 else ("→ Désavantage biais ⚠️" if val == 1 else "→ Neutre")
        elif feat == 'Âge':
            val_txt = f"{int(val)} ans"
            impact  = "→ Désavantage biais ⚠️" if val > 40 else "→ Neutre"
        else:
            val_txt = f"{val:.0f}"
            impact  = f"{'↑ Au-dessus' if diff > 0 else '↓ En-dessous'} de la moyenne ({moy:.0f})"
        print(f"      {feat:<25} : {val_txt:<15} {impact}")

# Afficher 3 exemples
for i in [0, 5, 10]:
    expliquer_candidat(i, X_test, y_test, model, features, df)

# ============================================================
#   5. GRAPHIQUES DES BIAIS
# ============================================================
print("\n" + "-" * 60)
print("  5. GÉNÉRATION DES GRAPHIQUES DE BIAIS")
print("-" * 60)

BLEU  = "#1F4E79"; ROUGE = "#C00000"
VERT  = "#1D6A3A"; GRIS  = "#F2F2F2"
ORANGE= "#C55A11"

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "BiasDetect_RH — Analyse des Biais dans le Recrutement",
    fontsize=15, fontweight='bold', color=BLEU, y=0.99
)
fig.patch.set_facecolor('#FAFAFA')

# ── G1 : Importance des variables ────────────────────────
ax1 = axes[0, 0]
imp_plot = imp_df.sort_values('Importance')
colors   = [ROUGE if v in biais_vars else BLEU for v in imp_plot['Variable']]
bars = ax1.barh(imp_plot['Variable'], imp_plot['Importance'],
                color=colors, edgecolor='white', height=0.6)
ax1.errorbar(imp_plot['Importance'], range(len(imp_plot)),
             xerr=imp_plot['Std'], fmt='none', color='gray',
             capsize=3, linewidth=1)
ax1.set_title("Importance des Variables\n(Permutation)", fontweight='bold', color=BLEU)
ax1.set_xlabel("Impact sur la décision")
ax1.set_facecolor(GRIS)
ax1.legend(handles=[
    mpatches.Patch(color=ROUGE, label='⚠️ Variable de biais'),
    mpatches.Patch(color=BLEU,  label='Variable de mérite')
], fontsize=8)

# ── G2 : Biais Genre ─────────────────────────────────────
ax2 = axes[0, 1]
tg = df.groupby('Genre')['Décision Finale'].apply(
    lambda x: (x=='Retenu').mean()*100)
b2 = ax2.bar(tg.index, tg.values, width=0.4,
             color=[BLEU if g=='Homme' else ROUGE for g in tg.index],
             edgecolor='white')
for bar, val in zip(b2, tg.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=13)
moy = (df['Décision Finale']=='Retenu').mean()*100
ax2.axhline(y=moy, color='gray', linestyle='--', linewidth=1.5,
            label=f'Moyenne : {moy:.1f}%')
ax2.set_title("⚠️ Biais Genre\n(Écart de {:.1f} points)".format(ecart_genre),
              fontweight='bold', color=ROUGE)
ax2.set_ylabel("% Retenu"); ax2.set_ylim(0, 80)
ax2.legend(fontsize=9); ax2.set_facecolor(GRIS)

# ── G3 : Biais Âge ───────────────────────────────────────
ax3 = axes[0, 2]
ta = df.groupby('Tranche Âge', observed=True)['Décision Finale'].apply(
    lambda x: (x=='Retenu').mean()*100)
colors_age = [VERT, BLEU, ROUGE]
b3 = ax3.bar(ta.index.astype(str), ta.values, width=0.5,
             color=colors_age, edgecolor='white')
for bar, val in zip(b3, ta.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax3.axhline(y=moy, color='gray', linestyle='--', linewidth=1.5,
            label=f'Moyenne : {moy:.1f}%')
ax3.set_title("⚠️ Biais Âge\n(Pénalité après 40 ans)",
              fontweight='bold', color=ROUGE)
ax3.set_ylabel("% Retenu"); ax3.set_ylim(0, 80)
ax3.legend(fontsize=9); ax3.set_facecolor(GRIS)

# ── G4 : Biais Statut Marital ─────────────────────────────
ax4 = axes[1, 0]
ts = df.groupby('Statut Marital')['Décision Finale'].apply(
    lambda x: (x=='Retenu').mean()*100)
colors_s = [BLEU, ROUGE, VERT]
b4 = ax4.bar(ts.index, ts.values, width=0.5,
             color=colors_s, edgecolor='white')
for bar, val in zip(b4, ts.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax4.axhline(y=moy, color='gray', linestyle='--', linewidth=1.5,
            label=f'Moyenne : {moy:.1f}%')
ax4.set_title("⚠️ Biais Statut Marital\n(Mariés favorisés)",
              fontweight='bold', color=ROUGE)
ax4.set_ylabel("% Retenu"); ax4.set_ylim(0, 80)
ax4.legend(fontsize=9); ax4.set_facecolor(GRIS)

# ── G5 : Biais croisé Genre × Âge ────────────────────────
ax5 = axes[1, 1]
pivot = df.groupby(['Genre','Tranche Âge'], observed=True)['Décision Finale'].apply(
    lambda x: (x=='Retenu').mean()*100).unstack()
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
            ax=ax5, linewidths=1, linecolor='white',
            annot_kws={'size':12,'weight':'bold'},
            vmin=30, vmax=80)
ax5.set_title("⚠️ Biais Croisé\nGenre × Âge (%Retenu)",
              fontweight='bold', color=ROUGE)
ax5.set_xlabel("Tranche d'Âge")

# ── G6 : Résumé des biais ─────────────────────────────────
ax6 = axes[1, 2]
ax6.axis('off')
resume = [
    ("BIAIS DÉTECTÉS", "", BLEU),
    ("", "", BLANC := "#FFFFFF"),
    ("⚠️  Genre",
     f"Hommes : {taux_genre.get('Homme', 0):.1f}%\nFemmes : {taux_genre.get('Femme', 0):.1f}%\nÉcart : {ecart_genre:.1f} pts",
     ROUGE),
    ("⚠️  Âge > 40 ans",
     f"20-30 : {taux_age.iloc[0]:.1f}%\n31-40 : {taux_age.iloc[1]:.1f}%\n41-55 : {taux_age.iloc[2]:.1f}%",
     ORANGE),
    ("⚠️  Statut Marital",
     f"Mariés favorisés\nvs Célibataires\nvs Divorcés",
     "#7030A0"),
]
y_pos = 0.95
for titre, detail, couleur in resume:
    if titre == "BIAIS DÉTECTÉS":
        ax6.text(0.5, y_pos, titre, transform=ax6.transAxes,
                 fontsize=13, fontweight='bold', color=BLEU,
                 ha='center', va='top')
        y_pos -= 0.08
    elif titre:
        ax6.text(0.05, y_pos, titre, transform=ax6.transAxes,
                 fontsize=11, fontweight='bold', color=couleur,
                 va='top')
        ax6.text(0.55, y_pos, detail, transform=ax6.transAxes,
                 fontsize=9, color='#333333', va='top')
        y_pos -= 0.22
ax6.set_facecolor('#F8F8F8')
ax6.set_title("Résumé des Biais", fontweight='bold', color=BLEU)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('resultats_biais.png', dpi=150,
            bbox_inches='tight', facecolor='#FAFAFA')
print("\n   ✅ Graphique sauvegardé : resultats_biais.png")

print("\n" + "=" * 60)
print("  ✅ ÉTAPE 3 TERMINÉE — BIAIS DÉTECTÉS ET VISUALISÉS !")
print("=" * 60)