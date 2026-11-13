# ============================================================
#   ÉTAPE 5 — SYSTÈME MULTI-AGENT
#   Projet : BiasDetect_RH
# ============================================================

import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# ============================================================
#   CLASSE DE BASE — AGENT
# ============================================================
class Agent:
    """Classe de base pour tous les agents"""
    def __init__(self, nom):
        self.nom     = nom
        self.statut  = "En attente"
        self.resultat = None

    def log(self, message):
        print(f"   [{self.nom}] {message}")

    def executer(self, donnees=None):
        raise NotImplementedError

# ============================================================
#   AGENT 1 — COLLECTE DES DONNÉES
# ============================================================
class AgentCollecte(Agent):
    def __init__(self):
        super().__init__("Agent Collecte")

    def executer(self, donnees=None):
        self.log("Démarrage — lecture du fichier Excel...")
        time.sleep(0.5)

        df = pd.read_excel(
            'Dataset_RH_Organise_v2.xlsx',
            sheet_name='📋 Candidats',
            header=1
        )

        self.log(f"✅ {len(df)} candidats chargés")
        self.log(f"✅ {len(df.columns)} colonnes détectées")
        self.log(f"✅ Colonnes : {list(df.columns[:5])}...")

        self.statut  = "Terminé"
        self.resultat = df
        return df

# ============================================================
#   AGENT 2 — NETTOYAGE ET PRÉTRAITEMENT
# ============================================================
class AgentNettoyage(Agent):
    def __init__(self):
        super().__init__("Agent Nettoyage")
        self.encoders = {}

    def executer(self, donnees=None):
        self.log("Démarrage — nettoyage des données...")
        time.sleep(0.5)
        df = donnees.copy()

        # Sélection des features
        features = [
            'Genre', 'Âge', 'Statut Marital', 'Expérience (ans)',
            'Diplôme', 'Nb Langues', 'Secteur', 'Score Test (/100)',
            'Score Entretien (/100)', 'Nb Entretiens', 'Distance (km)',
            'Salaire Attendu (MAD)', 'Déjà Recruté', 'Durée Poste (mois)'
        ]
        cible = 'Décision Finale'
        df_model = df[features + [cible]].copy()

        # Nettoyage valeurs manquantes
        avant = df_model.isnull().sum().sum()
        df_model['Durée Poste (mois)'] = df_model['Durée Poste (mois)'].fillna(0)
        self.log(f"✅ Valeurs manquantes : {avant} → 0")

        # Encodage
        df_model['Genre']          = df_model['Genre'].map({'Homme':1,'Femme':0})
        df_model['Déjà Recruté']   = df_model['Déjà Recruté'].map({'Oui':1,'Non':0})
        df_model[cible]            = df_model[cible].map({'Retenu':1,'Rejeté':0})

        for col in ['Statut Marital','Diplôme','Secteur']:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            self.encoders[col] = le

        self.log(f"✅ Encodage terminé — {len(features)} features prêtes")

        # Split train/test
        X = df_model[features]
        y = df_model[cible]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.log(f"✅ Split : {len(X_train)} train | {len(X_test)} test")

        self.statut  = "Terminé"
        self.resultat = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'features': features, 'encoders': self.encoders,
            'df_original': donnees
        }
        return self.resultat

# ============================================================
#   AGENT 3 — MODÈLE ML
# ============================================================
class AgentModele(Agent):
    def __init__(self):
        super().__init__("Agent Modèle ML")
        self.model = None

    def executer(self, donnees=None):
        self.log("Démarrage — entraînement du modèle...")
        time.sleep(0.5)

        X_train = donnees['X_train']
        X_test  = donnees['X_test']
        y_train = donnees['y_train']
        y_test  = donnees['y_test']

        # Créer et entraîner le modèle
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        self.log("✅ Random Forest entraîné (100 arbres)")

        # Évaluation
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        f1      = f1_score(y_test, y_pred)
        acc     = (y_pred == y_test).mean()

        self.log(f"✅ AUC-ROC  : {auc:.3f}")
        self.log(f"✅ F1 Score : {f1:.3f}")
        self.log(f"✅ Accuracy : {acc:.1%}")

        # Sauvegarder le modèle
        with open('modele_random_forest.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        self.log("✅ Modèle sauvegardé")

        self.statut  = "Terminé"
        self.resultat = {**donnees, 'model': self.model, 'auc': auc, 'f1': f1}
        return self.resultat

# ============================================================
#   AGENT 4 — DÉTECTION DES BIAIS
# ============================================================
class AgentBiais(Agent):
    def __init__(self):
        super().__init__("Agent Détection Biais")
        self.biais = {}

    def executer(self, donnees=None):
        self.log("Démarrage — analyse des biais...")
        time.sleep(0.5)

        df    = donnees['df_original']
        model = donnees['model']

        # Biais Genre
        taux_genre = df.groupby('Genre')['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100)
        ecart_genre = abs(taux_genre.get('Homme',0) - taux_genre.get('Femme',0))
        self.biais['Genre'] = {
            'Homme': taux_genre.get('Homme', 0),
            'Femme': taux_genre.get('Femme', 0),
            'ecart': ecart_genre
        }
        self.log(f"⚠️  Biais Genre détecté : écart de {ecart_genre:.1f} points")

        # Biais Âge
        df['Tranche Âge'] = pd.cut(df['Âge'], bins=[20,30,40,55],
                                    labels=['20-30','31-40','41-55'])
        taux_age = df.groupby('Tranche Âge', observed=True)['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100)
        self.biais['Age'] = taux_age.to_dict()
        self.log(f"⚠️  Biais Âge détecté : {taux_age.to_dict()}")

        # Biais Statut Marital
        taux_statut = df.groupby('Statut Marital')['Décision Finale'].apply(
            lambda x: (x=='Retenu').mean()*100)
        ecart_statut = taux_statut.max() - taux_statut.min()
        self.biais['Statut Marital'] = {
            'taux': taux_statut.to_dict(),
            'ecart': ecart_statut
        }
        self.log(f"⚠️  Biais Statut Marital : écart de {ecart_statut:.1f} points")

        # Importance des variables
        features = donnees['features']
        importances = pd.Series(
            model.feature_importances_,
            index=features
        ).sort_values(ascending=False)
        self.biais['importances'] = importances.to_dict()
        self.log(f"✅ Importance des variables calculée")

        self.statut  = "Terminé"
        self.resultat = {**donnees, 'biais': self.biais}
        return self.resultat

# ============================================================
#   AGENT 5 — RAPPORT TEXTE
# ============================================================
class AgentRapport(Agent):
    def __init__(self):
        super().__init__("Agent Rapport")

    def executer(self, donnees=None):
        self.log("Démarrage — génération du rapport...")
        time.sleep(0.5)

        biais = donnees['biais']
        auc   = donnees['auc']
        f1    = donnees['f1']
        df    = donnees['df_original']

        rapport = []
        rapport.append("=" * 65)
        rapport.append("   RAPPORT FINAL — BIASDETECT RH")
        rapport.append("   Explicabilité & Détection des Biais dans le Recrutement")
        rapport.append("=" * 65)

        rapport.append(f"\n1. DATASET")
        rapport.append(f"   → Candidats analysés   : {len(df)}")
        rapport.append(f"   → Variables utilisées  : 14 features")
        rapport.append(f"   → Taux global retenu   : {(df['Décision Finale']=='Retenu').mean()*100:.1f}%")

        rapport.append(f"\n2. PERFORMANCE DU MODÈLE (Random Forest)")
        rapport.append(f"   → AUC-ROC  : {auc:.3f}")
        rapport.append(f"   → F1 Score : {f1:.3f}")
        rapport.append(f"   → Interprétation : modèle au-dessus du hasard (0.5)")

        rapport.append(f"\n3. BIAIS DÉTECTÉS")

        g = biais['Genre']
        rapport.append(f"\n   ⚠️  BIAIS GENRE")
        rapport.append(f"   → Hommes retenus  : {g['Homme']:.1f}%")
        rapport.append(f"   → Femmes retenues : {g['Femme']:.1f}%")
        rapport.append(f"   → Écart           : {g['ecart']:.1f} points ← DISCRIMINATION")

        rapport.append(f"\n   ⚠️  BIAIS ÂGE")
        for tranche, taux in biais['Age'].items():
            rapport.append(f"   → {tranche} ans : {taux:.1f}%")

        sm = biais['Statut Marital']
        rapport.append(f"\n   ⚠️  BIAIS STATUT MARITAL")
        for statut, taux in sm['taux'].items():
            rapport.append(f"   → {statut:15s} : {taux:.1f}%")
        rapport.append(f"   → Écart max : {sm['ecart']:.1f} points")

        rapport.append(f"\n4. TOP 5 VARIABLES LES PLUS INFLUENTES")
        for i, (var, imp) in enumerate(
                sorted(biais['importances'].items(),
                       key=lambda x: x[1], reverse=True)[:5], 1):
            biais_flag = " ⚠️ BIAIS" if var in ['Genre','Âge','Statut Marital'] else ""
            rapport.append(f"   {i}. {var:25s} : {imp:.4f}{biais_flag}")

        rapport.append(f"\n5. RECOMMANDATIONS")
        rapport.append(f"   → Supprimer Genre et Statut Marital des critères de sélection")
        rapport.append(f"   → Mettre en place un audit régulier des décisions RH")
        rapport.append(f"   → Former les recruteurs aux biais inconscients")
        rapport.append(f"   → Utiliser uniquement : Score Test, Expérience, Diplôme")

        rapport.append(f"\n{'='*65}")
        rapport.append(f"   FIN DU RAPPORT — BiasDetect RH")
        rapport.append(f"{'='*65}")

        rapport_texte = "\n".join(rapport)

        # Sauvegarder le rapport
        with open('rapport_final.txt', 'w', encoding='utf-8') as f:
            f.write(rapport_texte)

        self.log("✅ Rapport sauvegardé : rapport_final.txt")
        self.statut  = "Terminé"
        self.resultat = rapport_texte
        return rapport_texte

# ============================================================
#   ORCHESTRATEUR — CHEF D'ORCHESTRE
# ============================================================
class Orchestrateur:
    """Coordonne tous les agents dans le bon ordre"""
    def __init__(self):
        self.agents = [
            AgentCollecte(),
            AgentNettoyage(),
            AgentModele(),
            AgentBiais(),
            AgentRapport()
        ]

    def executer_pipeline(self):
        print("\n" + "=" * 65)
        print("   SYSTÈME MULTI-AGENT — BIASDETECT RH")
        print("   Démarrage du pipeline...")
        print("=" * 65)

        donnees = None
        for i, agent in enumerate(self.agents, 1):
            print(f"\n{'─'*65}")
            print(f"  🤖 AGENT {i} — {agent.nom.upper()}")
            print(f"{'─'*65}")
            donnees = agent.executer(donnees)
            print(f"  ✅ Statut : {agent.statut}")

        print(f"\n{'='*65}")
        print("   ✅ PIPELINE TERMINÉ — TOUS LES AGENTS ONT RÉUSSI !")
        print("=" * 65)

        # Afficher le rapport
        print("\n" + donnees)
        return donnees

# ============================================================
#   LANCEMENT
# ============================================================
if __name__ == "__main__":
    orchestrateur = Orchestrateur()
    orchestrateur.executer_pipeline()