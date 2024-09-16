#importo le librerie
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#per ogni file .csv presente nella cartella /results crea una cartella con lo stesso nome del file .csv
#e al suo interno salva i grafici
import os

# Funzione principale
def crea_cartelle_per_csv(directory):
    # Controlla se la directory esiste
    if not os.path.exists(directory):
        print(f"La directory {directory} non esiste.")
        return
    
    # Scorre tutti i file nella directory
    for file in os.listdir(directory):
        # Controlla se il file ha estensione .csv
        if file.endswith(".csv"):
            # Crea una cartella con lo stesso nome del file (senza estensione)
            nome_cartella = os.path.splitext(file)[0]
            percorso_cartella = os.path.join(directory, nome_cartella)
            
            # Crea la cartella se non esiste già
            if not os.path.exists(percorso_cartella):
                os.makedirs(percorso_cartella)
                print(f"Creata cartella: {percorso_cartella}")
            else:
                print(f"La cartella {percorso_cartella} esiste già.")

# Esempio di utilizzo
directory_input = "percorso/della/tu_directory"
crea_cartelle_per_csv(directory_input)
