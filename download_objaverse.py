import objaverse
import os
import shutil

# --- 1. Configuration des chemins et catégories ---
# Votre chemin spécifique
dossier_base = "/home/nathan/Documents/research/HY3DGS/Datasets/objaverse/raw/"

# Sélection Option 3 : Diversité maximale (999 objets environ)
categories_souhaitees = ["chair", "seashell", "antenna"]

# Création du dossier racine s'il n'existe pas
os.makedirs(dossier_base, exist_ok=True)

# --- 2. Récupération des UIDs via LVIS ---
print("Chargement des annotations LVIS...")
lvis_annotations = objaverse.load_lvis_annotations()

uids_par_categorie = {}
uids_a_telecharger = []

for categorie in categories_souhaitees:
    if categorie in lvis_annotations:
        # On récupère tous les objets disponibles pour ces catégories
        uids = lvis_annotations[categorie]
        uids_par_categorie[categorie] = uids
        uids_a_telecharger.extend(uids)
        print(f"Catégorie '{categorie}' : {len(uids)} objets identifiés.")
    else:
        print(f"Erreur : La catégorie '{categorie}' est introuvable.")

# --- 3. Téléchargement des objets ---
print(f"\nLancement du téléchargement de {len(uids_a_telecharger)} objets...")
print("Note : Le téléchargement peut être long selon votre connexion.")

# Téléchargement effectif (utilise le cache par défaut d'objaverse)
objects = objaverse.load_objects(
    uids=uids_a_telecharger,
    download_processes=4  # Vous pouvez augmenter ce chiffre si vous avez un CPU puissant
)

# --- 4. Organisation par dossier de catégorie ---
print("\nOrganisation des fichiers dans les dossiers cibles...")

for categorie, uids in uids_par_categorie.items():
    # Création du sous-dossier de catégorie (ex: .../raw/chair/)
    dossier_cible_cat = os.path.join(dossier_base, categorie)
    os.makedirs(dossier_cible_cat, exist_ok=True)
    
    compteur_deplace = 0
    
    for uid in uids:
        if uid in objects:
            chemin_source = objects[uid]
            # Le nom du fichier sera l'UID.glb
            chemin_destination = os.path.join(dossier_cible_cat, f"{uid}.glb")
            
            # Déplacement du fichier du cache vers votre dossier final
            if os.path.exists(chemin_source):
                try:
                    shutil.move(chemin_source, chemin_destination)
                    compteur_deplace += 1
                except Exception as e:
                    print(f"Erreur lors du déplacement de {uid} : {e}")
    
    print(f"-> {compteur_deplace} fichiers déplacés dans {dossier_cible_cat}")

print(f"\nOpération terminée avec succès !")
print(f"Vos données sont prêtes dans : {dossier_base}")