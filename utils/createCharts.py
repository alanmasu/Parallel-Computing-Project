# Per ogni file .csv presente nella cartella /results crea una cartella con lo stesso nome del file .csv
# e al suo interno salva i grafici

# Importo le librerie
import os
import shutil
import pythonCharts as charts
import argparse

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

# Cicla sulle cartelle create e crea i grafici
def creaGrafici(directory, force):
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            # Controlla se esistono delle immagini nella cartella
            images = [f for f in os.listdir(os.path.join(directory, folder)) if f.endswith('.png')]
            if images and force == False:
                print(f"Le immagini sono già state create per {folder}.")
                continue
            
            # Controlla se il file .csv esiste nella cartella
            file_csv = os.path.join(directory, folder, f"result-{folder[7:]}.csv")
            if os.path.exists(file_csv):
                # Crea il grafico
                res = charts.createFlopsChart(file_csv, os.path.join(directory, folder, "chart"))
                if res:
                    print(f"Creato grafico per {file_csv}")
            else:
                print(f"Il file {file_csv} non esiste.")
    

## Main
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Create charts from results')
    argparser.add_argument('-f', '--force', action='store_true', help='Force creation of charts even if they already exist')
    argparser.add_argument('-d', '--directory', type=str, help='Directory to create charts from', default='./results')
    args = argparser.parse_args()
    
    force = args.force
    directory_input = args.directory
    
    charts.seabornConfig()
    crea_cartelle_e_sposta_file(directory_input)
    creaGrafici(directory_input, force)