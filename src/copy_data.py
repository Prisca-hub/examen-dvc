#!/usr/bin/env python3
"""
Script simple pour copier raw.csv vers le serveur Ubuntu
"""

import subprocess
import sys

def copy_file():
    source = "C:\\Users\\Prisc\\Desktop\\raw.csv"
    destination = "ubuntu@ip-172-31-33-103:~/examen-dvc/data/"
    
    try:
        print(f"Copie de {source} vers {destination}...")
        result = subprocess.run(
            ["scp", source, destination], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("Fichier copié avec succès!")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la copie: {e}")
        print(f"Sortie d'erreur: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    copy_file()
