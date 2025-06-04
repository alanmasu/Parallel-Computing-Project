#importo le librerie
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Argparse
import argparse
import os
import sys
import shutil



def seabornConfig():
    # Set the style
    sns.set_style("whitegrid")

def createFlopsChart(filename, img_filename):
    plt.clf()
    
    # Try to read the dataframe from the file, if error, return 0
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading dataframe from file {filename}: {e}")
        return 0
    
    if df.empty:
        print(f"Dataframe is empty")
        return 0
    
    # Print the dataframe
    print(df)

    # Create a scatter plot
    sns.lineplot(x='Size', y='cuBLAS_TFLOPS', data=df, label='cuBlas', color='red')
    sns.lineplot(x='Size', y='TFLOPS', data=df, label='WMMA', color='blue')
    # sns.lineplot(x='Size', y='wall_clock_time_routine2', data=df, label='Vectorized + flag', color='green')

    # Add title and axis names
    plt.title('Performance comparison of cuBLAS and WMMA')
    plt.xlabel('Size')
    plt.ylabel('TFLOPs')
    plt.xscale('log')

    ticks = []
    ticks_labels = []

    for i in range(5, 15):
        ticks.append(2**i)
        ticks_labels.append('$2^{' + str(i) + '}$')
        
    plt.xticks(ticks, ticks_labels)
    plt.legend()
    if img_filename.endswith(".pdf"):
        plt.savefig(img_filename, format="pdf")
    else
        plt.savefig(img_filename + '.png')
    # plt.show()

    plt.yscale('log')
    if img_filename.endswith(".pdf"):
        #remove the .pdf extension to avoid duplication
        img_filename = img_filename[:-4]
        plt.savefig(img_filename + '-log.pdf', format="pdf")
    else:
        plt.savefig(img_filename + '-log.png')
    return 1

# Main
if __name__ == "__main__":
    # Set the style
    seabornConfig()
    
    # Get from parameters using argparse
    argparser = argparse.ArgumentParser(description='Create charts from results')
    argparser.add_argument('-f', '--filename', type=str, help='Path to the result file')
    argparser.add_argument('-o', '--output', type=str, help='Path to the output file')
    args = argparser.parse_args()
    filename = args.filename
    output = args.output

    # Create the chart
    createFlopsChart(filename, output)