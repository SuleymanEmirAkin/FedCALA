import glob 
import re
from collections import defaultdict

import numpy as np

experiment_dir = "experiments/*/*.txt"

file_paths = sorted(list(glob.glob(experiment_dir)))


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


model2dataset2nc2mean = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
model2dataset2nc2std = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))


for model in model2dataset2nc2scores:
    for dataset in model2dataset2nc2scores[model]:
        for nc in model2dataset2nc2scores[model][dataset]:
            scores = model2dataset2nc2scores[model][dataset][nc]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        model2dataset2nc2mean[model][dataset][nc] = mean_score
        model2dataset2nc2std[model][dataset][nc] = std_score


dataset_names = ["MUTAG", "BZR", "COX2", "ENZYMES", "DD", "PROTEINS", "COLLAB", "BINARY", "MULTI"]
model_names = ["FedAvg", "FedProx", "Scaffold", "Scaffold", "FedProto", "FedALA", "FedCALA"]

# print like a table to copy to latex

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{l" + "c" * len(dataset_names) + "}")
print("\\hline")
print("Model & " + " & ".join(dataset_names) + " \\\\")
print("\\hline")

for model in model_names:
    row_str = [model]
    for dataset in dataset_names:
        print(model.lower(), dataset.lower())
        print(model2dataset2nc2scores[model.lower()][dataset].keys())
        biggest_nc = max(model2dataset2nc2scores[model.lower()][dataset].keys(), key=lambda x: int(x))
        scores = model2dataset2nc2scores[model.lower()][dataset.lower()][str(biggest_nc)]
        
        if len(scores) > 1:
            # We have multiple runs: Show Mean ± Std
            m = np.mean(scores)
            s = np.std(scores)
            row_str.append(f"${m:.2f} \\pm {s:.2f}$")
        elif len(scores) == 1:
            # Only one run found: Show just the value
            row_str.append(f"{scores[0]:.2f}")
        else:
            row_str.append("-")
            
    print(" & ".join(row_str) + " \\\\")

print("\\hline")
print("\\end{tabular}")
print("\\end{table}")