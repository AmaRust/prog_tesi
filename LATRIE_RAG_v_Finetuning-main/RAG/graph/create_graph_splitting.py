import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def calculate_mean_bleu(bleu):
    return sum(bleu.values()) / len(bleu)

def calculate_mean_rouge(rouge):
    return sum([rouge['rouge_1']['fmeasure'], rouge['rouge_L']['fmeasure']]) / 2

def calculate_mean_bertscore(bertscore):
    return sum(bertscore.values()) / len(bertscore)

def plot_metrics(data, situation, title):
    splitting_method_names = {
        "Settings": "Settings",
        "From scratch with Faiss vector store and SYNTACTIC splitter": "Syntactic splitter from scratch",
        "From scratch with Faiss vector store and SEMANTIC splitter": "Semantic splitter from scratch",
    }

    param_values = [splitting_method_names[entry["parameters"]["splitting method"]] for entry in data]

    bleu_scores = [calculate_mean_bleu(entry["metrics"][situation]["BLEU"]) for entry in data]
    rouge_scores = [calculate_mean_rouge(entry["metrics"][situation]["ROUGE"]) for entry in data]
    meteor_scores = [entry["metrics"][situation]["METEOR"] for entry in data]
    bertscore_scores = [calculate_mean_bertscore(entry["metrics"][situation]["BERTScore"]) for entry in data]

    bertscore_precision = [entry["metrics"][situation]["BERTScore"]["Precision"] for entry in data]
    bertscore_recall = [entry["metrics"][situation]["BERTScore"]["Recall"] for entry in data]
    bertscore_f1 = [entry["metrics"][situation]["BERTScore"]["F1_measure"] for entry in data]

    plt.figure(figsize=(14, 6))

    # First subplot
    plt.subplot(1, 2, 1)
    plt.plot(param_values, bleu_scores, label='BLEU', marker='o')
    plt.plot(param_values, rouge_scores, label='ROUGE', marker='o')
    plt.plot(param_values, meteor_scores, label='METEOR', marker='o')
    plt.plot(param_values, bertscore_scores, label='BERTScore', marker='o')
    plt.xlabel("Splitting method")
    plt.ylabel('Scores')
    plt.title(title + " - General Scores")
    plt.legend()
    plt.grid(True)
    
    # Second subplot
    plt.subplot(1, 2, 2)
    plt.plot(param_values, bertscore_precision, label='BERTScore Precision', marker='o')
    plt.plot(param_values, bertscore_recall, label='BERTScore Recall', marker='o')
    plt.plot(param_values, bertscore_f1, label='BERTScore F1', marker='o')
    plt.xlabel("Splitting method")
    plt.ylabel('Scores')
    plt.title(title + " - BERTScore Components")
    plt.legend()
    plt.grid(True)


def add_constant_parameters_page(pdf, data):
    constant_parameters = data[0]["parameters"].copy()
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

    json_file = "../json_result_files/splitting_method/results_splitting_method_config_query.json" 

    if os.path.isfile(os.path.join("../json_result_files", json_file)):
    
        with open(json_file, 'r') as f:
            data = json.load(f)

        pdf_filename = "pdf_files/splitting_method/splitting_method_graphs.pdf"

        with PdfPages(pdf_filename) as pdf:
            # Add the constant parameters page
            add_constant_parameters_page(pdf, data)

            for situation, title in zip(situations, titles):
                plt.figure(figsize=(10, 6))
                plot_metrics(data, situation, title)
                pdf.savefig()
                plt.close()

        print(f"Graphs saved to {pdf_filename}")

    json_file = "../json_result_files/splitting_method/results_splitting_method_config_queries.json"

    if os.path.isfile(os.path.join("../json_result_files", json_file)):
        with open(json_file, 'r') as f:
            data = json.load(f)

        pdf_filename = "pdf_files/splitting_method/splitting_method_graphs_queries.pdf"

        with PdfPages(pdf_filename) as pdf:
            # Add the constant parameters page
            add_constant_parameters_page(pdf, data)

            for situation, title in zip(situations, titles):
                plt.figure(figsize=(10, 6))
                plot_metrics(data, situation, title)
                pdf.savefig()
                plt.close()

        print(f"Graphs saved to {pdf_filename}")