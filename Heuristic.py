import pandas as pd
import numpy as np
import re

# Fonction pour appliquer la logique de mots-clés (inchangée)
def apply_keyword_logic(row):
    """
    Applique une logique de mots-clés sur les colonnes 'Show' et 'Segment'.
    Retourne un thème ('info', 'sport', 'musique', 'humour', 'societe') ou np.nan
    """
    show_text = str(row['Show']).lower() if pd.notna(row['Show']) else ''
    segment_text = str(row['Segment']).lower() if pd.notna(row['Segment']) else ''
    combined_text = show_text + " " + segment_text

    keywords = {
        'info': [r'\bjournal\b', r'\binfo\b', 'météo'],
        'sport': [r'\bsport\b', r'\bmatch\b'],
        'musique': [r'\bmusique\b', r'\balbum\b'],
        'humour': [r'\brire\b', r'\bcomique\b'],
        'societe': [r'\bsoci[eé]t[eé]\b', r'\bculture\b', r'\bd[eé]bat\b', r'\binterview\b', r'\breportage\b']
    }

    for theme, kws in keywords.items():
        if any(re.search(kw, combined_text) for kw in kws):
            return theme
    
    return np.nan

# ----- Chargement des données -----
file_name = "Tags_Focus_Normalises_20251027_231733.csv"
try:
    df = pd.read_csv(file_name, sep=';')

    # ----- Implémentation de la hiérarchie des thèmes -----

    # 1. Initialiser les nouvelles colonnes
    df['theme'] = np.nan
    df['theme_source'] = 'unknown' 

    # 2. Règle 1: "Assigned Tags Focus"
    df['Assigned Tags Focus'] = df['Assigned Tags Focus'].replace('', np.nan)
    condition_tag = df['Assigned Tags Focus'].notna()
    df.loc[condition_tag, 'theme'] = df['Assigned Tags Focus']
    df.loc[condition_tag, 'theme_source'] = 'tag'

    # 3. Règle 2: "Show Mapping" (Espace réservé)
    # ...

    # 4. Règle 3: "Fallback Keywords"
    # Sélectionner les lignes où le thème n'a pas encore été défini
    condition_keyword_fallback = df['theme'].isna()
    
    # Obtenir le sous-ensemble du DataFrame où le fallback s'applique
    df_fallback = df[condition_keyword_fallback]
    
    if not df_fallback.empty:
        # Appliquer la fonction 'apply_keyword_logic' sur ce sous-ensemble
        # keyword_themes aura le MÊME index que df_fallback (un sous-ensemble de l'index de df)
        keyword_themes = df_fallback.apply(apply_keyword_logic, axis=1)
        
        # Trouver les lignes (dans ce sous-ensemble) où un thème a été trouvé
        condition_keyword_found = keyword_themes.notna()
        
        # Obtenir la série de thèmes qui ont été trouvés (non nuls)
        # L'index de 'themes_to_apply' est un sous-ensemble de l'index de keyword_themes
        themes_to_apply = keyword_themes[condition_keyword_found]
        
        # Appliquer ces thèmes et la source au DataFrame original
        # en utilisant l'index de 'themes_to_apply'
        # C'est la correction clé : on utilise themes_to_apply.index
        if not themes_to_apply.empty:
            df.loc[themes_to_apply.index, 'theme'] = themes_to_apply
            df.loc[themes_to_apply.index, 'theme_source'] = 'keyword'
        
        # Les lignes où keyword_themes était np.nan garderont 
        # df['theme'] = np.nan et df['theme_source'] = 'unknown' (défini à l'étape 1)

    # ----- Documentation et Couverture -----
    
    print("------ Documentation des Heuristiques (Fallback Keywords) (Corrigé) ------")
    print("La logique de 'fallback' s'applique si 'Assigned Tags Focus' est vide...")
    print("- 'info': ['journal', 'info', 'météo']")
    print("- 'sport': ['sport', 'match']")
    print("- 'musique': ['musique', 'album']")
    print("- 'humour': ['rire', 'comique']")
    print("- 'societe': ['société', 'culture', 'débat', 'interview', 'reportage'] (avec gestion des accents)")
    print("\nSi aucun de ces mots-clés n'est trouvé, le thème reste NUL ('unknown').")
    print("-----------------------------------------------------------------")
    
    print("\n------ Couverture des Sources de Thèmes ------")
    total_rows = len(df)
    print(f"Nombre total de lignes : {total_rows}\n")
    
    coverage_counts = df['theme_source'].value_counts(dropna=False)
    print("Comptage absolu des sources :")
    print(coverage_counts)
    
    coverage_perc = (df['theme_source'].value_counts(normalize=True, dropna=False) * 100).round(2)
    print("\nCouverture en pourcentage :")
    print(coverage_perc)
    print("-------------------------------------------------")
    
    # ----- Sauvegarde des résultats -----
    output_file = "Tags_Focus_Normalises_with_Fallback_v3_corrige.csv"
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    
    print(f"\nFichier de sortie (v3 corrigé) avec les thèmes et sources enregistré sous : {output_file}")

except FileNotFoundError:
    print(f"Erreur : Le fichier {file_name} n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")
