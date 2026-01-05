import glob 
import re
from collections import defaultdict
import numpy as np

import glob 
import re
from collections import defaultdict

import numpy as np

experiment_dir = "experiments/*/*.txt"

file_paths = sorted(list(glob.glob(experiment_dir)))

# print(file_paths)

model2dataset2nc2scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def extract_best_test_score(text):
    pattern = r"best_test_accuracy:\s*([\d\.]+)"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]  # Return the last one found
    return None


def get_best_test_score(fpath):
    with open(fpath, "r") as f:
        whole_text = f.read()
    best_test_score = extract_best_test_score(whole_text)
    return float(best_test_score) if best_test_score is not None else None
    

for fpath in file_paths:
    base_dir = fpath.split("/")[-1]
    parts = base_dir.split("_")
    model_name = parts[0]
    dataset_name = parts[1]
    client_num = parts[3]
    best_test_score = get_best_test_score(fpath)
    model2dataset2nc2scores[model_name][dataset_name][client_num].append(best_test_score)

# print(model2dataset2nc2scores)
# ... (keep your existing data collection logic here) ...
# Assuming model2dataset2nc2scores is populated exactly as you had it.

dataset_names = ["MUTAG", "BZR", "COX2", "ENZYMES", "DD", "PROTEINS", "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]
# Map long names to the short versions in your target table
display_datasets = ["MUTAG", "BZR", "COX2", "ENZYMES", "DD", "PROTEINS", "COLLAB", "BINARY", "MULTI"]
model_names = ["FedAvg", "FedProx", "Scaffold", "FedProto", "FedALA", "FedCALA"]
client_scales = ["5", "10", "20"]

print("\\begin{table*}[!htp]")
print("\\caption{Graph-FL test accuracy (\\%) for different client scales.}")
print("\\label{tab:graph-fl-baselines}")
print("\\centering")
print("\\renewcommand{\\arraystretch}{1.1}")
print("\\resizebox{\\textwidth}{!}{%")
print("\\begin{tabular}{ll" + "c" * len(dataset_names) + "}")
print("\\hline")
print("\\textbf{Model} & \\textbf{Clients} & " + " & ".join([f"\\textbf{{{d}}}" for d in display_datasets]) + " \\\\")
print("\\hline")

for model in model_names:
    # We use multirow for the first column. 
    # The first row of the model block gets the multirow command.
    for i, client in enumerate(client_scales):
        row_cells = []
        
        # Handle Model Name Column
        if i == 0:
            # Check if FedCALA to bold it as per your requirement
            m_display = f"\\textbf{{{model}}}" if model == "FedCALA" else model
            row_cells.append(f"\\multirow{{{len(client_scales)}}}{{*}}{{{m_display}}}")
        else:
            row_cells.append("")
            
        # Handle Client Scale Column
        row_cells.append(client)
        
        # Handle Dataset Score Columns
        for dataset in dataset_names:
            # Note: Ensure the key matches your file path structure (lower case etc.)
            key = model.lower() 
            scores = model2dataset2nc2scores[key][dataset][client]
            # # print(model2dataset2nc2scores.keys())
            # print(scores)
            # print(dataset, model, client)
            # input()
            if len(scores) > 1:
                m = np.mean(scores)
                s = np.std(scores)
                # Formatting as 0.74 \pm 0.09
                val = f"${m:.2f} \\pm {s:.2f}$"
                row_cells.append(val)
            elif len(scores) == 1:
                row_cells.append(f"{scores[0]:.2f}")
            else:
                row_cells.append("-")
        
        print(" & ".join(row_cells) + " \\\\")
    
    # Add a horizontal line after each model group
    print("\\hline")

print("\\end{tabular}")
print("}")
print("\\end{table*}")