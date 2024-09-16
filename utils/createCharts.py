#importo le librerie
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns

#per ogni file .csv presente nella cartella /results crea una cartella con lo stesso nome del file .csv
#e al suo interno salva i grafici
import os
import shutil

# Funzione principale
def crea_cartelle_e_sposta_file(directory):
    # Controlla se la directory esiste
    if not os.path.exists(directory):
        print(f"La directory {directory} non esiste.")
        return
    
    # Scorre tutti i file nella directory
    for file in os.listdir(directory):
        # Controlla se il file è un file .csv
        if file.startswith("result-") and file.endswith(".csv"):
            # Estrai il suffisso <DATA>-<ORA> dal nome del file
            nome_base = file[7:-4]  # Rimuove "result-" all'inizio e ".csv" alla fine
            file_txt = f"description-{nome_base}.txt"
            
            # Controlla se il file .txt corrispondente esiste
            if os.path.exists(os.path.join(directory, file_txt)):
                # Crea una cartella con il nome result-<DATA>-<ORA>
                nome_cartella = f"result-{nome_base}"
                percorso_cartella = os.path.join(directory, nome_cartella)
                
                # Crea la cartella se non esiste già
                if not os.path.exists(percorso_cartella):
                    os.makedirs(percorso_cartella)
                    print(f"Creata cartella: {percorso_cartella}")
                
                # Sposta il file .csv nella cartella
                shutil.move(os.path.join(directory, file), os.path.join(percorso_cartella, file))
                print(f"Spostato {file} in {percorso_cartella}")
                
                # Sposta il file .txt nella cartella
                shutil.move(os.path.join(directory, file_txt), os.path.join(percorso_cartella, file_txt))
                print(f"Spostato {file_txt} in {percorso_cartella}")
            else:
                print(f"Il file {file_txt} non esiste per il file {file}")

# Esempio di utilizzo
directory_input = "./results"
crea_cartelle_e_sposta_file(directory_input)
