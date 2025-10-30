#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import io

# Nom du fichier initial
file_path = "Mesures V0.1.csv"

# --- Fonction de conversion de durée ---
def convert_duration_to_seconds(duration_str):
    if not isinstance(duration_str, str):
        return None
    try:
        parts = duration_str.split(':')
        if len(parts) == 3: # Format H:M:S
            h, m, s = map(int, parts)
            return (h * 3600) + (m * 60) + s
        elif len(parts) == 2: # Format M:S
            m, s = map(int, parts)
            return (m * 60) + s
        else:
            return None
    except (ValueError, TypeError):
        return None

try:
    # --- Étape 0: Chargement Initial et Correction des Colonnes ---
    print("--- Étape 0: Chargement Initial et Correction des Colonnes ---")
    df = pd.read_csv(file_path, sep=';')
    
    # *** CORRECTION ***: Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    print("Noms de colonnes nettoyés (suppression des espaces blancs).")
    
    rows_start = len(df)
    print(f"Nombre de lignes initial: {rows_start}")

    # --- Étape 1: Suppression des valeurs manquantes ---
    print("\n--- Étape 1: Suppression des valeurs manquantes ---")
    print(f"Nombre de lignes avant: {len(df)}")
    # Utilisation des noms de colonnes nettoyés
    subset_cols = ['Show ID', 'Show', 'Publication Date', 'App/Site Name', 'Device Class']
    df_step1 = df.dropna(subset=subset_cols)
    print(f"Nombre de lignes après: {len(df_step1)}")
    
    file_step1 = "Mesures_V1_no_missing.csv"
    df_step1.to_csv(file_step1, index=False, sep=';')
    print(f"Fichier sauvegardé: {file_step1}")

    # --- Étape 2: Conversion de 'Publication Date' ---
    print("\n--- Étape 2: Conversion de 'Publication Date' ---")
    print(f"Nombre de lignes avant: {len(df_step1)}")
    df_step2 = df_step1.copy()
    df_step2['Publication Date'] = pd.to_datetime(df_step2['Publication Date'], format='%d.%m.%Y', errors='coerce')
    
    print(f"Nombre de lignes après: {len(df_step2)}")
    file_step2 = "Mesures_V2_dates_converted.csv"
    df_step2.to_csv(file_step2, index=False, sep=';')
    print(f"Fichier sauvegardé: {file_step2}")

    # --- Étape 3: Conversion de 'New Visit Rate %' ---
    print("\n--- Étape 3: Conversion de 'New Visit Rate %' ---")
    print(f"Nombre de lignes avant: {len(df_step2)}")
    df_step3 = df_step2.copy()
    
    df_step3['New Visit Rate'] = df_step3['New Visit Rate %'].str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    df_step3['New Visit Rate'] = pd.to_numeric(df_step3['New Visit Rate'], errors='coerce') / 100
    df_step3 = df_step3.drop(columns=['New Visit Rate %'])
    
    print(f"Nombre de lignes après: {len(df_step3)}")
    file_step3 = "Mesures_V3_percent_converted.csv"
    df_step3.to_csv(file_step3, index=False, sep=';')
    print(f"Fichier sauvegardé: {file_step3}")

    # --- Étape 4: Conversion des durées ---
    print("\n--- Étape 4: Conversion des durées ('Avg Play Duration', 'Total Play Duration') ---")
    print(f"Nombre de lignes avant: {len(df_step3)}")
    df_step4 = df_step3.copy()
    
    df_step4['Avg Play Seconds'] = df_step4['Avg Play Duration'].apply(convert_duration_to_seconds)
    df_step4['Total Play Seconds'] = df_step4['Total Play Duration'].apply(convert_duration_to_seconds)
    df_step4 = df_step4.drop(columns=['Avg Play Duration', 'Total Play Duration'])
    
    print(f"Nombre de lignes après: {len(df_step4)}")
    file_step4 = "Mesures_V4_duration_converted.csv"
    df_step4.to_csv(file_step4, index=False, sep=';')
    print(f"Fichier sauvegardé: {file_step4}")
    
    # --- Étape 5: Vérification des colonnes catégorielles ---
    print("\n--- Étape 5: Vérification des colonnes catégorielles ---")
    print(f"Nombre de lignes du fichier final: {len(df_step4)}")
    
    print("\nValeurs uniques pour 'App/Site Name':")
    print(df_step4['App/Site Name'].unique())
    
    print("\nValeurs uniques pour 'Device Class':")
    print(df_step4['Device Class'].unique())

    # --- Nettoyage Terminé: Affichage du résultat final ---
    print("\n--- Nettoyage Terminé ---")
    print(f"Aperçu du fichier final ({file_step4}):")
    print(df_step4.head().to_markdown(index=False, numalign="left", stralign="left"))
    
    print("\nInformations sur le fichier final:")
    buffer = io.StringIO()
    df_step4.info(buf=buffer)
    print(buffer.getvalue())

except Exception as e:
    print(f"Une erreur est survenue lors du nettoyage : {e}")


# In[3]:


import pandas as pd
from datetime import datetime

# --- 1. Définir les noms de fichiers et les clés ---

# Noms de vos fichiers d'entrée
fichier1 = "Mesures_V4_duration_converted.csv"
fichier2 = "fichier_harmonise.csv"

# Clés primaires pour la fusion
cles_de_fusion = ["Show ID", "Segment ID"]

# --- 2. Générer le nom du fichier de sortie ---

# Obtenir la date et l'heure actuelles
maintenant = datetime.now()

# Formater la date et l'heure (ex: 20251027_1305)
timestamp = maintenant.strftime("%Y%m%d_%H%M") 

# Créer le nom de fichier complet
fichier_sortie = f"Tags_Measures_Merged_{timestamp}.csv"

# --- 3. Exécuter le processus de fusion ---

try:
    # Charger les fichiers CSV dans des DataFrames Pandas
    # NOTE : Si vos fichiers d'ENTRÉE utilisent aussi des ';', 
    # vous devrez ajouter sep=';' à pd.read_csv() aussi.
    print(f"Chargement du fichier 1 : {fichier1}...")
    df1 = pd.read_csv(fichier1, sep=';') 
    
    print(f"Chargement du fichier 2 : {fichier2}...")
    df2 = pd.read_csv(fichier2, sep=';')

    print("\nInformations sur les fichiers chargés :")
    print(f"  - Fichier 1 : {len(df1)} lignes")
    print(f"  - Fichier 2 : {len(df2)} lignes")

    # Fusionner (merge) les deux DataFrames
    print(f"\nFusion en cours sur les clés : {cles_de_fusion}...")
    df_fusionne = pd.merge(df1, df2, on=cles_de_fusion)

    # Sauvegarder le DataFrame fusionné dans un nouveau fichier CSV
    print(f"Sauvegarde du résultat dans : {fichier_sortie}...")
    
    # --- MODIFICATION ICI ---
    # Ajout de sep=';' pour que le fichier de sortie utilise 
    # des points-virgules.
    df_fusionne.to_csv(
        fichier_sortie, 
        index=False, 
        encoding='utf-8-sig', 
        sep=';'
    )

    print("\n✅ Opération terminée avec succès !")
    print(f"Le fichier fusionné contient {len(df_fusionne)} lignes.")

except FileNotFoundError:
    print("\n❌ ERREUR : Fichier non trouvé.")
    print("Veuillez vérifier que les fichiers suivants sont dans le même dossier que le script :")
    print(f"  - {fichier1}")
    print(f"  - {fichier2}")
except KeyError as e:
    print(f"\n❌ ERREUR : Colonne clé non trouvée.")
    print(f"La colonne {e} n'a pas été trouvée dans l'un des fichiers.")
    print("Vérifiez que les deux fichiers contiennent bien 'Show ID' et 'Segment ID'.")
except Exception as e:
    print(f"\n❌ Une erreur inattendue est survenue : {e}")


# In[4]:


import pandas as pd

# Charger le DataFrame
df = pd.read_csv('Tags_Measures_Merged_20251028_2307.csv', delimiter=';')

# Liste des colonnes de tags à vérifier
tag_cols = ['tag_societe', 'tag_humour', 'tag_info', 'tag_musique', 'tag_sport']

# 1. Supprimer la colonne "Focus_Tag"
df = df.drop(columns=['Focus_Tag'])

# 2. Créer (ou écraser) la colonne "tag_vide"
# La logique est : "tag_vide" vaut 1 si la somme des autres tags est 0 (aucun tag présent), sinon 0.
df['tag_vide'] = (df[tag_cols].sum(axis=1) == 0).astype(int)

# Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
df.to_csv('Tags_Measures_Modified_20251028.csv', sep=';', index=False)


# In[6]:


import pandas as pd
import numpy as np

# --- 1. Configuration et Heuristiques ---
# Définition des heuristiques de mots-clés
KEYWORD_HEURISTICS = {
    'info': ['journal', 'info', 'météo', 'actualités', 'nouvelles', 'presse', 'politique', 'économie', 'faits divers', 'alerte', 'flash'],
    'sport': ['sport', 'match', 'résultats', 'score', 'foot', 'tennis', 'basket', 'rugby', 'F1', 'championnat', 'coupe', 'JO', 'jeux olympiques', 'athlète', 'équipe'],
    'musique': ['musique', 'album', 'chanson', 'clip', 'concert', 'live', 'artiste', 'chanteur', 'chanteuse', 'groupe', 'festival', 'playlist', 'single'],
    'humour': ['rire', 'comique', 'drôle', 'blague', 'sketch', 'parodie', 'stand-up', 'humoriste', 'spectacle', 'bêtisier'],
    'societe': ['société', 'societe', 'culture', 'débat', 'interview', 'reportage', 'documentaire', 'enquête', 'entretien', 'social', 'environnement', 'histoire', 'éducation', 'podcast']
}

# Nom du fichier d'entrée et de sortie
INPUT_FILE = 'Tags_Measures_Modified_20251028.csv'  # Remplacez par le nom réel de votre fichier
OUTPUT_FILE = 'Tags_Focus_Normalises_with_Fallback_final.csv'
SEPARATOR = ';'

# --- 2. Documentation des Heuristiques (Sortie Console) ---
print("📜 Documentation : Résumé des Heuristiques de Mots-Clés Utilisées")
print("-" * 50)
for theme, keywords in KEYWORD_HEURISTICS.items():
    print(f"**{theme.upper()}** : {', '.join(keywords)}")
print("-" * 50)


# --- 3. Fonction de Classification par Mots-Clés ---
def classify_by_keywords(text):
    """
    Applique la logique de mots-clés sur une chaîne de caractères (insensible à la casse).
    Retourne le thème trouvé ou NaN.
    """
    if pd.isna(text):
        return np.nan
        
    text_lower = str(text).lower()
    
    # Parcourt les thèmes par ordre de définition
    for theme, keywords in KEYWORD_HEURISTICS.items():
        # Vérifie si au moins un mot-clé est présent dans le texte
        if any(keyword.lower() in text_lower for keyword in keywords):
            return theme
    
    # Aucun mot-clé trouvé
    return np.nan


# --- 4. Traitement du Fichier ---
try:
    # Lecture du fichier
    df = pd.read_csv(INPUT_FILE, sep=SEPARATOR)
except FileNotFoundError:
    print(f"\n❌ Erreur : Le fichier d'entrée '{INPUT_FILE}' n'a pas été trouvé.")
    print("Veuillez vérifier le chemin ou le nom du fichier.")
    # Créer un DataFrame minimal pour l'exemple si le fichier n'existe pas
    print("Création d'un DataFrame fictif pour démonstration...")
    data = {
        'Show': ['Le Journal de 20h', 'Match de Foot', 'Spectacle de Stand-up', 'Le Mag Société', 'Flash Info Spécial', 'Musique en Live', 'Sujet Sans Mot-Clé', 'Segment Spécial JO'],
        'Segment': ['Météo du Jour', 'Résultats du Championnat', 'Blagues et Sketches', 'Débat sur l\'Environnement', 'Alerte Politique', 'Album Acoustique', 'Description Générique', 'Athlète en interview'],
        'tag_vide': [1, 0, 0, np.nan, 0, 1, 0, 0] # 0, np.nan = Fallback activé
    }
    df = pd.DataFrame(data)
    # Assurez-vous que les colonnes 'Show' et 'Segment' sont des chaînes de caractères
    df['Show'] = df['Show'].astype(str)
    df['Segment'] = df['Segment'].astype(str)
    print("DataFrame Fictif Créé.")


# Création des nouvelles colonnes
df['theme'] = np.nan
df['theme_source'] = 'unknown'


# --- 5. Logique de Remplissage Hiérarchique (Priorité 1: Keyword Fallback) ---

# Condition pour activer le Fallback : 'tag_vide' est vide (NaN) ou vaut 0
# Convertir 'tag_vide' en numérique pour gérer les comparaisons avec 0 et NaN
# Utiliser .fillna(0) pour s'assurer que np.nan est traité comme 0 dans la condition,
# ce qui active le Fallback pour les lignes vides.
fallback_condition = (df['tag_vide'].fillna(0) == 0)

# Pour éviter l'erreur "Unalignable boolean Series", nous allons:
# 1. Travailler uniquement sur les lignes qui remplissent la condition de fallback.
# 2. Utiliser la méthode .loc pour l'assignation conditionnelle.

# Extraction des lignes nécessitant le fallback
df_fallback = df.loc[fallback_condition].copy()

if not df_fallback.empty:
    # Concaténation des colonnes Show et Segment (insensible à la casse)
    df_fallback['search_text'] = df_fallback['Show'].astype(str) + ' ' + df_fallback['Segment'].astype(str)

    # Application de la fonction de classification
    # Le résultat est une Series qui est ALIGNÉE par index avec df_fallback
    fallback_themes = df_fallback['search_text'].apply(classify_by_keywords)

    # Mise à jour du DataFrame original (df) UNIQUEMENT pour les lignes concernées
    # Nous utilisons l'index de df_fallback pour cibler les lignes dans df
    
    # 5.1. Mettre à jour la colonne 'theme'
    # Utiliser .loc[index, colonne] pour cibler les lignes et colonnes
    # L'assignation est faite uniquement pour les index présents dans fallback_themes.index
    df.loc[fallback_themes.dropna().index, 'theme'] = fallback_themes.dropna()

    # 5.2. Mettre à jour la colonne 'theme_source' pour les thèmes trouvés
    # La source devient 'keyword_fallback' uniquement si un thème a été trouvé (i.e., non NaN)
    df.loc[fallback_themes.dropna().index, 'theme_source'] = 'keyword_fallback'

    print(f"\n✅ Fallback de mots-clés appliqué sur {len(df_fallback)} lignes nécessitant le traitement.")
    print(f"   {fallback_themes.dropna().count()} thèmes trouvés et assignés par mots-clés.")
else:
    print("\nℹ️ Aucune ligne ne nécessite l'application du Fallback (Condition 'tag_vide' == 0 ou NaN).")


# --- 6. Sortie (Sauvegarde du Fichier) ---

# Remplacer les valeurs NaN restantes par une chaîne vide dans 'theme_source'
df['theme_source'] = df['theme_source'].fillna('unknown')

try:
    df.to_csv(OUTPUT_FILE, sep=SEPARATOR, index=False, encoding='utf-8')
    print(f"\n💾 DataFrame modifié sauvegardé avec succès dans '{OUTPUT_FILE}' (Séparateur: '{SEPARATOR}').")
except Exception as e:
    print(f"\n❌ Erreur lors de la sauvegarde du fichier : {e}")

# Afficher les premières lignes du résultat pour vérification (optionnel)
print("\n--- Aperçu du Résultat ---")
print(df.head())


# In[7]:


import pandas as pd
import numpy as np
import re

# --- 1. Documentation: Heuristiques de mots-clés ---
# Définition des mots-clés pour chaque thème, en minuscules pour la correspondance
keyword_heuristics = {
    'info': ['journal', 'info', 'météo', 'actualités', 'nouvelles', 'presse', 'politique', 'économie', 'faits divers', 'alerte', 'flash'],
    'sport': ['sport', 'match', 'résultats', 'score', 'foot', 'tennis', 'basket', 'rugby', 'f1', 'championnat', 'coupe', 'jo', 'jeux olympiques', 'athlète', 'équipe'],
    'musique': ['musique', 'album', 'chanson', 'clip', 'concert', 'live', 'artiste', 'chanteur', 'chanteuse', 'groupe', 'festival', 'playlist', 'single'],
    'humour': ['rire', 'comique', 'drôle', 'blague', 'sketch', 'parodie', 'stand-up', 'humoriste', 'spectacle', 'bêtisier'],
    'societe': ['société', 'societe', 'culture', 'débat', 'interview', 'reportage', 'documentaire', 'enquête', 'entretien', 'social', 'environnement', 'histoire', 'éducation', 'podcast']
}

# Affichage de la documentation dans la console comme demandé
print("--- Documentation des Heuristiques de Mots-clés (Priorité 1) ---")
for theme, keywords in keyword_heuristics.items():
    print(f"Thème '{theme}': {', '.join(keywords)}")
print("--------------------------------------------------------------\n")

# --- 2. Configuration Fichiers ---
input_file = 'Tags_Measures_Modified_20251028 v2.csv'
output_file = 'Tags_Focus_Normalises_with_Fallback_final.csv'

try:
    # --- 3. Chargement des données ---
    print(f"Chargement du fichier : {input_file}")
    # Spécification du séparateur ';'
    df = pd.read_csv(input_file, sep=';')
    
    print("Aperçu des données chargées (5 premières lignes) :")
    print(df.head())
    print("\nInformations sur les colonnes (types de données) :")
    df.info()
    print("\n")

    # --- 4. Initialisation des nouvelles colonnes ---
    print("Initialisation des colonnes 'theme' (à NaN) et 'theme_source' (à 'unknown')...")
    df['theme'] = np.nan
    df['theme_source'] = 'unknown'

    # --- 5. Logique de Fallback (Priorité 1: Mots-clés) ---
    print("Application de la logique de fallback par mots-clés...")

    # Étape 5a : Définir les lignes cibles pour le fallback
    # Condition : 'tag_vide' est 0, vide (NaN), ou '0' (chaîne)
    # Conversion robuste de 'tag_vide' en numérique pour une comparaison fiable
    tag_vide_numeric = pd.to_numeric(df['tag_vide'], errors='coerce')
    condition_fallback = (tag_vide_numeric.isna()) | (tag_vide_numeric == 0)
    
    print(f"Nombre de lignes éligibles au fallback (tag_vide=0 ou vide) : {condition_fallback.sum()}")

    # Étape 5b : Créer la chaîne de recherche (concaténation insensible à la casse)
    # Gère les valeurs NaN potentielles dans Show ou Segment avant de concaténer
    search_string = (
        df['Show'].fillna('').astype(str) + ' ' + 
        df['Segment'].fillna('').astype(str)
    ).str.lower()
    
    # Compteur pour les mises à jour
    total_updated = 0

    # Étape 5c : Appliquer la recherche de mots-clés par ordre de priorité
    # L'ordre du dictionnaire 'keyword_heuristics' détermine la priorité
    # (le premier thème qui correspond gagne)
    
    for theme, keywords in keyword_heuristics.items():
        
        # Création du pattern regex pour le thème (ex: 'journal|info|météo')
        pattern = '|'.join([re.escape(k) for k in keywords])
        
        # Identification des correspondances
        matches = search_string.str.contains(pattern, case=False, regex=True)
        
        # Définition du masque de mise à jour :
        # 1. Doit être une ligne éligible au fallback (condition_fallback)
        # 2. Ne doit pas déjà avoir un thème assigné (par une itération précédente)
        # 3. Doit correspondre au pattern de mot-clé actuel
        
        update_mask = (
            condition_fallback &    # Lignes éligibles
            df['theme'].isna() &    # Pas encore de thème assigné
            matches                 # Le mot-clé pour ce thème a été trouvé
        )
        
        # Étape 5d : Assignation des valeurs
        # On assigne une valeur *scalaire* ('theme' et 'keyword_fallback')
        # aux lignes correspondant au masque.
        # CELA ÉVITE L'ERREUR "Unalignable boolean Series".
        
        if update_mask.any():
            count = update_mask.sum()
            print(f"  -> {count} lignes mises à jour avec le thème '{theme}' via mot-clé.")
            df.loc[update_mask, 'theme'] = theme
            df.loc[update_mask, 'theme_source'] = 'keyword_fallback'
            total_updated += count

    print(f"\nTotal des lignes mises à jour par la logique de fallback : {total_updated}")

    # --- 6. Sauvegarde du fichier ---
    print(f"\nSauvegarde du DataFrame modifié dans : {output_file}")
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')

    print("\n--- Traitement terminé avec succès ---")

except FileNotFoundError:
    print(f"ERREUR : Le fichier d'entrée '{input_file}' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")


# In[9]:


import pandas as pd
import numpy as np
import re

# --- 1. Documentation: Heuristiques de mots-clés ---
# Définition des mots-clés pour chaque thème, en minuscules pour la correspondance
keyword_heuristics = {
    'info': ['journal', 'info', 'météo', 'actualités', 'nouvelles', 'presse', 'politique', 'économie', 'faits divers', 'alerte', 'flash'],
    'sport': ['sport', 'match', 'résultats', 'score', 'foot', 'tennis', 'basket', 'rugby', 'f1', 'championnat', 'coupe', 'jeux olympiques', 'athlète', 'équipe'],
    'musique': ['musique', 'album', 'chanson', 'clip', 'concert', 'live', 'artiste', 'chanteur', 'chanteuse', 'groupe', 'festival', 'playlist', 'single'],
    'humour': ['rire', 'comique', 'drôle', 'blague', 'sketch', 'parodie', 'stand-up', 'humoriste', 'spectacle', 'bêtisier'],
    'societe': ['société', 'societe', 'culture', 'débat', 'interview', 'reportage', 'documentaire', 'enquête', 'entretien', 'social', 'environnement', 'histoire', 'éducation', 'podcast']
}

# Affichage de la documentation dans la console comme demandé
print("--- Documentation des Heuristiques de Mots-clés (Priorité 1) ---")
print("Logique : Appliquée si 'tag_vide' == 1.")
for theme, keywords in keyword_heuristics.items():
    print(f"Thème '{theme}': {', '.join(keywords)}")
print("--------------------------------------------------------------\n")

# --- 2. Configuration Fichiers ---
input_file = 'Tags_Measures_Modified_20251028 v2.csv' 
output_file = 'Tags_Focus_Normalises_with_Fallback_final.csv'

try:
    # --- 3. Chargement des données ---
    print(f"Chargement du fichier : {input_file}")
    df = pd.read_csv(input_file, sep=';')
    
    # Vérification rapide pour s'assurer que les colonnes sont correctes
    print("Vérification des colonnes (les 5 premières) :")
    print(df.head())
    
    # Utilisation de la colonne 'tag_vide' (sans espace)
    target_column = 'tag_vide'
    if target_column not in df.columns:
        raise ValueError(f"La colonne '{target_column}' n'a pas été trouvée.")
    
    print(f"\nUtilisation de la colonne '{target_column}' pour la condition.")

    # --- 4. Initialisation des nouvelles colonnes ---
    print("Initialisation des colonnes 'theme' (à NaN) et 'theme_source' (à 'unknown')...")
    df['theme'] = np.nan
    df['theme_source'] = 'unknown'

    # --- 5. Logique de Fallback (Priorité 1: Mots-clés) ---
    print("Application de la logique de fallback par mots-clés...")

    # Étape 5a : Définir les lignes cibles pour le fallback
    # NOUVELLE CONDITION : 'tag_vide' doit valoir 1
    condition_fallback = (df[target_column] == 1)
    
    print(f"Nombre de lignes éligibles au fallback ({target_column}=1) : {condition_fallback.sum()}")

    # Étape 5b : Créer la chaîne de recherche (concaténation insensible à la casse)
    search_string = (
        df['Show'].fillna('').astype(str) + ' ' + 
        df['Segment'].fillna('').astype(str)
    ).str.lower()
    
    total_updated = 0

    # Étape 5c : Appliquer la recherche de mots-clés par ordre de priorité
    for theme, keywords in keyword_heuristics.items():
        
        pattern = '|'.join([re.escape(k) for k in keywords])
        
        # Identification des correspondances
        matches = search_string.str.contains(pattern, case=False, regex=True)
        
        # Définition du masque de mise à jour :
        # 1. Doit être une ligne éligible au fallback (condition_fallback -> tag_vide == 1)
        # 2. Ne doit pas déjà avoir un thème assigné (par une itération précédente)
        # 3. Doit correspondre au pattern de mot-clé actuel
        
        update_mask = (
            condition_fallback &    # Lignes éligibles (tag_vide == 1)
            df['theme'].isna() &    # Pas encore de thème assigné
            matches                 # Le mot-clé pour ce thème a été trouvé
        )
        
        # Étape 5d : Assignation de valeurs scalaires pour éviter l'erreur "Unalignable"
        if update_mask.any():
            count = update_mask.sum()
            print(f"  -> {count} lignes mises à jour avec le thème '{theme}' via mot-clé.")
            df.loc[update_mask, 'theme'] = theme
            df.loc[update_mask, 'theme_source'] = 'keyword_fallback'
            total_updated += count

    print(f"\nTotal des lignes mises à jour par la logique de fallback : {total_updated}")
    
    # Lignes qui étaient éligibles (tag_vide=1) mais n'ont trouvé aucun mot-clé
    no_match_count = (condition_fallback & df['theme'].isna()).sum()
    print(f"Lignes éligibles ({target_column}=1) restantes sans thème assigné (NaN) : {no_match_count}")


    # --- 6. Sauvegarde du fichier ---
    print(f"\nSauvegarde du DataFrame modifié dans : {output_file}")
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')

    print("\n--- Traitement terminé avec succès ---")

except FileNotFoundError:
    print(f"ERREUR : Le fichier d'entrée '{input_file}' n'a pas été trouvé.")
except ValueError as ve:
    print(f"ERREUR (Configuration) : {ve}")
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")


# In[10]:


import pandas as pd

# Nom du fichier d'entrée
file_name = 'Tags_Focus_Normalises_with_Fallback_final (3).csv'
# Nom du fichier de sortie
output_file = 'Tags_Focus_Normalises_modifie.csv'

print(f"Chargement du fichier : {file_name}")

try:
    # Charger le fichier CSV avec le bon délimiteur
    df = pd.read_csv(file_name, delimiter=';')
    
    print("Fichier chargé avec succès.")
    
    # S'assurer que les colonnes requises existent
    if 'tag_vide' not in df.columns or 'theme' not in df.columns:
        print("Erreur : Les colonnes 'tag_vide' ou 'theme' sont manquantes.")
    else:
        # Obtenir l'ensemble des noms de colonnes pour une vérification rapide
        all_columns = set(df.columns)
        
        # Définir la fonction à appliquer à chaque ligne
        def update_tag_based_on_theme(row):
            try:
                # 1. Vérifier si 'tag_vide' vaut 1
                # On vérifie la valeur entière 1 et la chaîne '1' par sécurité
                if row['tag_vide'] == 1 or str(row['tag_vide']).strip() == '1':
                    
                    # 2. Récupérer la valeur de 'theme'
                    theme_value = row['theme']
                    
                    # 3. Vérifier que 'theme' n'est pas vide ou une valeur non significative
                    if pd.notna(theme_value) and isinstance(theme_value, str):
                        theme_value_clean = theme_value.strip()
                        if theme_value_clean not in ['-', '', 'unknown']:
                            
                            # 4. Construire le nom de la colonne cible
                            target_col = f"tag_{theme_value_clean}"
                            
                            # 5. Vérifier si cette colonne cible existe
                            if target_col in all_columns:
                                # 6. Mettre à jour la valeur de cette colonne à 1
                                row[target_col] = 1
                                
            except Exception as e:
                pass
                
            # Retourner la ligne (modifiée ou non)
            return row

        print("Application de la logique de mise à jour...")
        # Appliquer la fonction sur chaque ligne (axis=1)
        df_modified = df.apply(update_tag_based_on_theme, axis=1)
        
        print("Logique appliquée.")
        
        # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
        df_modified.to_csv(output_file, sep=';', index=False)
        
        print(f"\nScript terminé. Le fichier modifié a été enregistré sous : {output_file}")

except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_name}' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur générale est survenue : {e}")


# In[11]:


import pandas as pd
import numpy as np
import sys

# --- Configuration ---
# Nom du fichier d'entrée (celui que vous avez fourni)
input_file = 'Tags_Focus_Normalises_modifie.csv'
# Nom du fichier de sortie (le nouveau fichier qui sera créé)
output_file = 'Tags_Focus_Normalises_avec_kpi.csv'
# --- Fin de la Configuration ---

try:
    # Étape 1 : Lire le fichier CSV. Nous spécifions le délimiteur.
    # L'option low_memory=False peut aider à éviter les avertissements sur les types de données mixtes.
    df = pd.read_csv(input_file, delimiter=';', low_memory=False)
    
    print(f"Fichier '{input_file}' chargé avec succès. {len(df)} lignes trouvées.")

    # Étape 2 : S'assurer que les colonnes de calcul sont numériques
    # C'est crucial. Si une valeur n'est pas un nombre, elle sera convertie en NaN (Not a Number).
    # Nous utilisons les noms de colonnes exacts trouvés dans votre fichier.
    cols_to_numeric = [
        'Entries', 'Visitors', 'Avg Play Seconds', 
        'Segment Length', 'Media Views', 'Returning Visits'
    ]
    
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Erreur : La colonne requise '{col}' est introuvable.", file=sys.stderr)
            sys.exit(1) # Quitte le script si une colonne manque

    # Remplacer les NaN (créés par errors='coerce') par 0 pour les calculs.
    # Sauf pour 'Segment Length', où un NaN doit rester NaN pour la division.
    cols_to_fill_zero = ['Entries', 'Visitors', 'Avg Play Seconds', 'Media Views', 'Returning Visits']
    df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)


    # Étape 3 : Calculer les nouvelles colonnes
    
    # --- 1. entry_rate = entries / visitors ---
    # np.where(condition, si_vrai, si_faux)
    df['entry_rate'] = np.where(
        df['Visitors'] > 0,                  # Condition : si Visitors est supérieur à 0
        df['Entries'] / df['Visitors'],      # Calcul si vrai
        0                                    # Valeur si faux (pour éviter la division par zéro)
    )

    # --- 2. pct_consumed = avg_play_s / segment_length_s ---
    df['pct_consumed'] = np.where(
        df['Segment Length'] > 0,            # Condition : si Segment Length est supérieur à 0
        df['Avg Play Seconds'] / df['Segment Length'], # Calcul si vrai
        np.nan                               # Valeur si faux (un segment de longueur 0 a un % consommé indéfini)
    )

    # --- 3. views_per_visitor = media_views / visitors (handle div0) ---
    df['views_per_visitor'] = np.where(
        df['Visitors'] > 0,
        df['Media Views'] / df['Visitors'],
        0
    )

    # --- 4. returning_rate = returning_visits / visitors ---
    # Définition : Ce KPI représente la proportion de visiteurs
    # qui sont des "visites récurrentes" (Returning Visits) 
    # par rapport au nombre total de "visiteurs" (Visitors) uniques.
    df['returning_rate'] = np.where(
        df['Visitors'] > 0,
        df['Returning Visits'] / df['Visitors'],
        0
    )

    # Étape 4 : Sauvegarder le nouveau DataFrame dans un fichier CSV
    # Nous utilisons index=False pour ne pas ajouter une colonne d'index inutile.
    # Nous réutilisons le délimiteur ';'
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    
    print(f"\nCalculs terminés avec succès.")
    print(f"Le nouveau fichier a été sauvegardé sous : '{output_file}'")

except FileNotFoundError:
    print(f"Erreur : Le fichier '{input_file}' n'a pas été trouvé.", file=sys.stderr)
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}", file=sys.stderr)


# In[ ]:




