import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import seaborn as sns
import pandas as pd

def calculate_mean_bleu(bleu):
    return sum(bleu.values()) / len(bleu)

def calculate_mean_rouge(rouge):
    return sum([rouge['rouge_1']['fmeasure'], rouge['rouge_L']['fmeasure']]) / 2

def calculate_mean_bertscore(bertscore):
    return sum(bertscore.values()) / len(bertscore)

def plot_metrics(data, situation, title):
    # Préparer les données pour le DataFrame
    plot_data = []
    for entry in data:
        llm = entry["parameters"]["llm"]
        splitting_method = entry["parameters"]["splitting method"]
        bleu = calculate_mean_bleu(entry["metrics"][situation]["BLEU"])
        rouge = calculate_mean_rouge(entry["metrics"][situation]["ROUGE"])
        meteor = entry["metrics"][situation]["METEOR"]
        bertscore = calculate_mean_bertscore(entry["metrics"][situation]["BERTScore"])
        bertscore_precision = entry["metrics"][situation]["BERTScore"]["Precision"] 
        bertscore_recall = entry["metrics"][situation]["BERTScore"]["Recall"] 
        bertscore_f1 = entry["metrics"][situation]["BERTScore"]["F1_measure"]
        
        plot_data.append({
            'splitting method': splitting_method,
            'llm': llm,
            'BLEU': bleu,
            'ROUGE': rouge,
            'METEOR': meteor,
            'BERTScore': bertscore,
            'BertScore precision': bertscore_precision,
            'BertScore recall': bertscore_recall,
            'BertScore F1': bertscore_f1
        })
    
    df = pd.DataFrame(plot_data)

    # Utilisation de seaborn pour créer des faceted plots
    g1 = sns.FacetGrid(df, col='splitting method', height=4, aspect=1.5)
    g1.map_dataframe(sns.lineplot, x='llm', y='BLEU', label='BLEU', marker='o', color='blue')
    g1.map_dataframe(sns.lineplot, x='llm', y='ROUGE', label='ROUGE', marker='o', color='red')
    g1.map_dataframe(sns.lineplot, x='llm', y='METEOR', label='METEOR', marker='o', color='green')
    g1.map_dataframe(sns.lineplot, x='llm', y='BERTScore', label='BERTScore', marker='o', color='purple')

    g1.set_axis_labels('Llms', 'Scores')
    g1.set_titles('Splitting : {col_name}')
    
    # Ajouter un titre global
    plt.subplots_adjust(top=0.9)  # Ajuster la position du titre global
    g1.fig.suptitle(title)
    
    # Ajouter des légendes
    for ax in g1.axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:4], ['BLEU', 'ROUGE', 'METEOR', 'BERTScore'], loc='best')

    # Utilisation de seaborn pour créer des faceted plots
    g2 = sns.FacetGrid(df, col='splitting method', height=4, aspect=1.5)
    g2.map_dataframe(sns.lineplot, x='llm', y='BertScore precision', label='BertScore precision', marker='o', color='blue')
    g2.map_dataframe(sns.lineplot, x='llm', y='BertScore recall', label='BertScore recall', marker='o', color='red')
    g2.map_dataframe(sns.lineplot, x='llm', y='BertScore F1', label='BertScore F1', marker='o', color='green')

    g2.set_axis_labels('Llms', 'BERTScores')
    g2.set_titles('Splitting : {col_name}')
    
    # Ajouter un titre global
    plt.subplots_adjust(top=0.9)  # Ajuster la position du titre global
    g2.fig.suptitle(title)
    
    # Ajouter des légendes
    for ax in g2.axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:4], ['BertScore precision', 'BertScore recall', 'BertScore F1'], loc='best')

    return g1, g2

def add_constant_parameters_page(pdf, data):
    constant_parameters = data[0]["parameters"].copy()
    del constant_parameters["llm"]
    del constant_parameters["splitting method"]
    
    text = "Constant Parameters:\n\n"
    for key, value in constant_parameters.items():
        text += f"{key}: {value}\n"

    if 'queries' in data[0]:
        text += "\nQueries:"
        for i in range(len(data[0]['queries'])):
            text += f"\n{data[0]['queries'][i]}\n"
    else:
        text += f"\nQuery: {data[0]['query']}"

    lines = text.split('\n')
    max_lines_per_page = 35  
    pages = []
    page = []
    for line in lines:
        if len(page) < max_lines_per_page:
            page.append(line)
        else:
            pages.append(page)
            page = [line]
    if page:
        pages.append(page)

    for page in pages:
        plt.figure(figsize=(10, 10))
        plt.text(0.1, 0.5, '\n'.join(page), fontsize=12, va='center', wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    situations = ["no additional prompt / no context", "additional prompt / no context", "no additional prompt / context", "additional prompt / context"]
    titles = [
        "No Additional Prompt / No Context",
        "Additional Prompt / No Context",
        "No Additional Prompt / Context",
        "Additional Prompt / Context"
    ]    
    
    json_file = f"../json_result_files/llm/results_llm_config_query.json"

    if os.path.isfile(os.path.join("../json_result_files", json_file)):
    
        with open(json_file, 'r') as f:
            data = json.load(f)

        pdf_filename = "pdf_files/llm/llm_graphs.pdf"

        with PdfPages(pdf_filename) as pdf:
            # Add the constant parameters page
            add_constant_parameters_page(pdf, data)

            for situation, title in zip(situations, titles):
                plt.figure(figsize=(15, 10))
                g1, g2 = plot_metrics(data, situation, title)
                pdf.savefig(g1.fig)
                pdf.savefig(g2.fig)
                plt.close()

        print(f"Graphs saved to {pdf_filename}")

    json_file = "../json_result_files/llm/results_llm_config_queries.json"

    if os.path.isfile(os.path.join("../json_result_files", json_file)):
        with open(json_file, 'r') as f:
            data = json.load(f)

        pdf_filename = "pdf_files/llm/llm_graphs_queries.pdf"

        with PdfPages(pdf_filename) as pdf:
            # Add the constant parameters page
            add_constant_parameters_page(pdf, data)

            for situation, title in zip(situations, titles):
                plt.figure(figsize=(15, 10))
                g1, g2 = plot_metrics(data, situation, title)
                pdf.savefig(g1.fig)
                pdf.savefig(g2.fig)
                plt.close()

        print(f"Graphs saved to {pdf_filename}")