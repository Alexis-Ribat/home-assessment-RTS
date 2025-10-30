#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# --- Configuration ---
# REMPLACEZ par le nom de votre fichier d'entrée
fichier_entree = 'Tags V0.1.csv' 
# REMPLACEZ par le nom souhaité pour le fichier de sortie
fichier_sortie = 'Tags nettoye V0.2.csv' 
# ---------------------
try:
    # --- 1. Lecture du fichier ---
    # Ajout de sep=';' pour spécifier le séparateur point-virgule
    try:
        df = pd.read_csv(fichier_entree, sep=';')
    except UnicodeDecodeError:
        # Essayer avec un encodage différent si le premier échoue
        # Ajout de sep=';' ici aussi
        df = pd.read_csv(fichier_entree, sep=';', encoding='latin1')
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        print("S'il s'agit d'un fichier Excel, utilisez pd.read_excel('votre_fichier.xlsx')")
        exit()

    print(f"Fichier initial chargé : {len(df)} lignes.")

    # S'assurer que les colonnes de tags sont bien des chaînes de caractères
    # .fillna('') remplace les cellules vides (NaN) par une chaîne vide
    df['Assigned Tags'] = df['Assigned Tags'].fillna('').astype(str)

    # --- 2. Identification des lignes à supprimer ---
    # On crée une colonne temporaire.
    # Elle sera 'True' si le tag, après avoir enlevé les espaces au début/fin,
    # se termine par un '-'. C'est le critère de la ligne à *supprimer*.
    df['est_indesirable'] = df['Assigned Tags'].str.strip().str.endswith('-')

    # --- 3. Tri des données ---
    # On trie par les colonnes ID, puis par notre colonne 'est_indesirable'.
    # ascending=True fait que 'False' (à garder) vient AVANT 'True' (à supprimer).
    df_trie = df.sort_values(
        by=['Segment ID', 'Show ID', 'est_indesirable'],
        ascending=[True, True, True]
    )

    # --- 4. Suppression des doublons ---
    # Pour chaque groupe ('Segment ID', 'Show ID'), on garde la première ligne ('first').
    # Grâce au tri, la première ligne est toujours celle qu'on veut garder.
    df_propre = df_trie.drop_duplicates(
        subset=['Segment ID', 'Show ID'],
        keep='first'
    )

    # --- 5. Nettoyage final ---
    # On supprime la colonne temporaire qui n'est plus nécessaire
    df_propre = df_propre.drop(columns=['est_indesirable'])

    # --- 6. Sauvegarde du résultat ---
    # On sauvegarde le fichier nettoyé avec le séparateur point-virgule
    # Adaptez si vous voulez un fichier Excel (df_propre.to_excel(fichier_sortie, index=False))
    df_propre.to_csv(fichier_sortie, index=False, sep=';')

    print(f"Nettoyage terminé. {len(df_propre)} lignes sauvegardées dans '{fichier_sortie}'.")

except FileNotFoundError:
    print(f"ERREUR : Le fichier '{fichier_entree}' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")


# In[2]:


import pandas as pd
from datetime import datetime
import re # Bibliothèque pour les expressions régulières (utilisée pour le comptage)

# 1. Définir la liste des mots-clés
focus_tags = ["societe", "humour", "info", "musique", "sport"]

# 2. Charger le fichier CSV
file_name = "Tags nettoye V0.2.csv"
try:
    df = pd.read_csv(file_name, sep=';')
    print(f"Fichier {file_name} chargé avec succès.")

    # 3. Définir la fonction d'extraction (inchangée)
    def extract_focus_tags(tag_string):
        """
        Extrait les mots-clés (focus_tags) d'une chaîne de tags.
        """
        if not isinstance(tag_string, str) or tag_string == '-':
            return ''
        
        found_tags = []
        tag_string_lower = tag_string.lower()
        
        for tag in focus_tags:
            # Vérifie si le mot-clé est présent
            if tag in tag_string_lower:
                found_tags.append(tag)
                
        # Retourne les tags trouvés, séparés par une virgule
        return ','.join(found_tags)

    # 4. Appliquer la fonction pour créer la nouvelle colonne
    df['Focus_Tag'] = df['Assigned Tags'].apply(extract_focus_tags)
    print("Nouvelle colonne 'Focus_Tag' créée.")

    # 5. Compter les occurrences (Nouveau)
    print("\n--- Comptage des tags ---")
    
    # Préparation des colonnes pour le comptage (gestion des NaN et conversion en str/minuscules)
    df_tags_original_lower = df['Assigned Tags'].astype(str).str.lower()
    df_tags_focus = df['Focus_Tag'].astype(str) # Déjà en minuscules par la fonction

    stats = {}
    for tag in focus_tags:
        # Compte le nombre de fois que le tag (substring) apparaît dans la colonne originale
        # .str.count() compte les occurrences de sous-chaînes (ex: "info" dans "rts_info")
        count_original = df_tags_original_lower.str.count(tag, flags=re.IGNORECASE).sum()
        
        # Compte le nombre de lignes contenant le tag dans la nouvelle colonne
        # .str.contains() vérifie si la ligne contient le tag (ex: "info" ou "info,societe")
        count_focus = df_tags_focus.str.contains(tag, na=False).sum()
        
        stats[tag] = {
            'Occurrences dans "Assigned Tags" (original)': int(count_original),
            'Lignes dans "Focus_Tag" (nouveau)': int(count_focus)
        }
        
    # Afficher les statistiques sous forme de tableau
    stats_df = pd.DataFrame(stats).T
    print("\nTableau récapitulatif des comptages :")
    print(stats_df.to_markdown(numalign="left", stralign="left"))


    # 6. Générer le nom de fichier dynamique (Nouveau)
    now = datetime.now()
    # Format: YYYYMMDD_HHMM (AnnéeMoisJour_HeureMinute)
    timestamp = now.strftime("%Y%m%d_%H%M")
    
    base_name = "fichier_nettoye 20251026 01h07"
    output_file_name = f"{base_name} {timestamp}.csv"

    print(f"\nNom de fichier dynamique généré : {output_file_name}")

    # 7. Sauvegarder le DataFrame modifié
    df.to_csv(output_file_name, sep=';', index=False, encoding='utf-8')
    print(f"Le nouveau fichier a été sauvegardé avec succès.")
    print(f"\nFichier CSV de sortie : {output_file_name}")


except Exception as e:
    print(f"Une erreur est survenue : {e}")


# In[4]:


import pandas as pd

# Définir les noms de fichiers
input_file = "fichier_nettoye 20251026 01h07 20251028_2143.csv"
output_file = "fichier_avec_tags_colonnes.csv"

try:
    # Charger le fichier CSV avec le bon séparateur
    df = pd.read_csv(input_file, sep=';')

    # Liste des tags pour lesquels créer des colonnes
    tags = ['societe', 'humour', 'info', 'musique', 'sport']

    # Créer une colonne binaire (0 ou 1) pour chaque tag
    for tag in tags:
        col_name = f"tag_{tag}"
        # .astype(str) gère les valeurs NaN (null) en les traitant comme la chaîne 'nan'
        # .str.contains(tag, case=False, na=False) recherche le tag sans tenir compte de la casse
        # et traite les NaN restants comme False (pas de correspondance)
        # .astype(int) convertit True/False en 1/0
        df[col_name] = df['Focus_Tag'].astype(str).str.contains(tag, case=False, na=False).astype(int)

    # Créer la colonne 'tag_vide'
    # Elle est à 1 si 'Focus_Tag' est nul (NaN) ou égal à '-' (basé sur l'aperçu du fichier)
    # L'aperçu du fichier montrait 'NaN' pour les valeurs manquantes après lecture,
    # donc isnull() est la vérification la plus fiable.
    df['tag_vide'] = (df['Focus_Tag'].isnull() | (df['Focus_Tag'] == '-')).astype(int)

    # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
    df.to_csv(output_file, sep=';', index=False)

    print(f"Le fichier '{output_file}' a été créé avec succès.")

    # (Aperçu et infos affichés dans la sortie de code)

except FileNotFoundError:
    print(f"Erreur : Le fichier '{input_file}' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")


# In[5]:


import pandas as pd

# Nom du fichier fourni par l'utilisateur
file_name = 'fichier_avec_tags_colonnes.csv'

try:
    # 1. Charger le fichier CSV en spécifiant le délimiteur
    df = pd.read_csv(file_name, delimiter=';')

    # 2. Inspecter les données pour confirmer les noms de colonnes et les types
    print("Aperçu des 5 premières lignes :")
    print(df.head())
    print("\nInformations sur les colonnes et types de données :")
    df.info()

    # 3. Identifier les colonnes de tags
    # Nous supposons que ce sont toutes les colonnes commençant par 'tag_'
    tag_columns = [col for col in df.columns if col.startswith('tag_')]
    
    if not tag_columns:
        print("Erreur : Aucune colonne commençant par 'tag_' n'a été trouvée.")
        print("Colonnes disponibles :", df.columns.tolist())
    else:
        print(f"\nColonnes de tags identifiées : {tag_columns}")

        # 4. S'assurer que les colonnes de tags sont numériques
        # (Dans ce cas, elles l'étaient déjà, mais c'est une bonne pratique)
        for col in tag_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remplacer les valeurs non numériques (devenues NaN) par 0
        df[tag_columns] = df[tag_columns].fillna(0)

        # 5. Grouper par 'Show ID' et calculer les métriques
        print("\nCalcul en cours...")
        
        # Dictionnaire pour les opérations d'agrégation
        agg_operations = {}
        
        # Calculer le taux (moyenne) pour chaque tag
        for col in tag_columns:
            agg_operations[col] = 'mean'
            
        # Compter le nombre total de segments pour ce Show ID
        # Nous utilisons 'Segment ID' pour le comptage
        agg_operations['Segment ID'] = 'count'
        
        # Exécuter l'agrégation
        analysis_df = df.groupby('Show ID').agg(agg_operations)
        
        # Renommer la colonne de comptage pour plus de clarté
        analysis_df = analysis_df.rename(columns={'Segment ID': 'total_segments'})

        # 6. Calculer le taux de segments ayant au moins un tag (sauf 'tag_vide')
        # Identifier les colonnes de tags "positifs" (tous sauf 'tag_vide')
        positive_tag_columns = [col for col in tag_columns if col != 'tag_vide']
        
        # Calculer la somme des tags positifs pour chaque segment
        df['total_positive_tags'] = df[positive_tag_columns].sum(axis=1)
        
        # Créer un indicateur binaire : 1 si au moins un tag positif, 0 sinon
        df['has_any_positive_tag'] = (df['total_positive_tags'] > 0).astype(int)
        
        # Calculer le taux de segments avec au moins un tag, par Show ID
        rate_any_tag = df.groupby('Show ID')['has_any_positive_tag'].mean()
        
        # Ajouter cette information à notre dataframe d'analyse
        analysis_df['rate_any_positive_tag'] = rate_any_tag

        # 7. Trier les résultats pour une meilleure lisibilité (par nombre de segments)
        analysis_sorted = analysis_df.sort_values(by='total_segments', ascending=False)

        # 8. Afficher les résultats
        print("\n--- Analyse du Taux de Tags par Show ID ---")
        print(f"Chaque ligne est un 'Show ID'.")
        print(f"Les colonnes 'tag_...' (sauf tag_vide) montrent le pourcentage de segments de ce show ayant ce tag.")
        print(f"'tag_vide' montre le pourcentage de segments n'ayant AUCUN tag.")
        print(f"'rate_any_positive_tag' montre le pourcentage de segments ayant AU MOINS UN tag (hors 'vide').")
        print(f"'total_segments' est le nombre total de segments analysés pour ce show.")
        print("\nRésultats (triés par nombre de segments) :")
        print(analysis_sorted)

        # 9. Sauvegarder les résultats dans un nouveau fichier CSV
        output_file = 'analyse_taux_tags_par_show_id.csv'
        analysis_sorted.to_csv(output_file, index=True) # index=True pour conserver le Show ID

        print(f"\nAnalyse complétée. Les résultats ont été sauvegardés dans : {output_file}")
        
except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_name}' n'a pas été trouvé.")
except pd.errors.EmptyDataError:
    print(f"Erreur : Le fichier '{file_name}' est vide.")
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")
    print("Veuillez vérifier le format de votre fichier, le délimiteur (doit être ';') et les noms de colonnes.")


# In[6]:


import pandas as pd

# Nom du fichier d'entrée
file_name = 'fichier_avec_tags_colonnes.csv'

try:
    # Charger le fichier CSV avec le bon séparateur
    df = pd.read_csv(file_name, sep=';')
    
    print("Données initiales (5 premières lignes) :")
    print(df.head())
    print("\nInformations sur les colonnes et types de données :")
    df.info()

    # Identifier dynamiquement toutes les colonnes de tags
    # (celles qui commencent par 'tag_')
    tag_columns = [col for col in df.columns if col.startswith('tag_')]
    
    if not tag_columns:
        print("Erreur : Aucune colonne commençant par 'tag_' n'a été trouvée.")
    else:
        print(f"\nColonnes de tags identifiées : {tag_columns}")

        # Définir le seuil (30%)
        threshold = 0.3

        # Créer une copie du DataFrame pour l'harmonisation
        df_harmonized = df.copy()

        # Calculer les proportions en groupant par 'Show ID'
        # .transform('mean') calcule la moyenne pour chaque groupe ('Show ID')
        # et propage cette moyenne à toutes les lignes appartenant à ce groupe.
        tag_proportions = df.groupby('Show ID')[tag_columns].transform('mean')

        # Appliquer la règle :
        # Si la proportion du groupe est >= 0.3, mettre 1
        # Sinon, mettre 0
        harmonized_values = (tag_proportions >= threshold).astype(int)

        # Remplacer les anciennes valeurs de tags par les nouvelles valeurs harmonisées
        df_harmonized[tag_columns] = harmonized_values

        # --- Section de Vérification (optionnelle mais utile) ---
        
        # Trouver un 'Show ID' avec plusieurs entrées pour montrer la différence
        show_id_counts = df['Show ID'].value_counts()
        # Filtrer pour les Show ID qui apparaissent plus d'une fois
        multi_segment_shows = show_id_counts[show_id_counts > 1].index
        
        if not multi_segment_shows.empty:
            sample_show_id = multi_segment_shows[0] # Prendre le premier
            
            print(f"\n--- Exemple de vérification pour Show ID : {sample_show_id} ---")
            
            print("\nAvant Harmonisation :")
            print(df[df['Show ID'] == sample_show_id][['Show ID'] + tag_columns].head())
            
            print("\nAprès Harmonisation :")
            print(df_harmonized[df_harmonized['Show ID'] == sample_show_id][['Show ID'] + tag_columns].head())
        else:
            print("\nPas de 'Show ID' avec plusieurs segments trouvé pour l'exemple de vérification.")
            
        # --- Fin de la vérification ---

        # Sauvegarder le fichier harmonisé
        output_file = 'fichier_harmonise.csv'
        df_harmonized.to_csv(output_file, sep=';', index=False, encoding='utf-8')

        print(f"\nL'harmonisation est terminée.")
        print(f"Le nouveau fichier a été enregistré sous : {output_file}")

except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_name}' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")

