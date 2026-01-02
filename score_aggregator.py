import glob 
import re
from collections import defaultdict

import numpy as np

experiment_dir = "experiments/*/*.txt"

file_paths = sorted(list(glob.glob(experiment_dir)))


model2dataset2scores = defaultdict(lambda: defaultdict(list))

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
    model2dataset2scores[model_name][dataset_name+"_nc"+client_num].append(best_test_score)



dataset_names = set()
for model in model2dataset2scores:
    for dataset in model2dataset2scores[model]:
        dataset_names.add(dataset)
dataset_names = sorted(list(dataset_names))

model_names = set()
for model in model2dataset2scores:
    model_names.add(model)
model_names = sorted(list(model_names))


model2dataset2mean = defaultdict(lambda: defaultdict(float))
model2dataset2std = defaultdict(lambda: defaultdict(float))


for model in model2dataset2scores:
    for dataset in model2dataset2scores[model]:
        scores = model2dataset2scores[model][dataset]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        model2dataset2mean[model][dataset] = mean_score
        model2dataset2std[model][dataset] = std_score



# print like a table to copy to latex

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{l" + "c" * len(dataset_names) + "}")
print("\\hline")
print("Model & " + " & ".join(dataset_names) + " \\\\")
print("\\hline")

for model in sorted(model2dataset2scores.keys()):
    row_str = [model]
    for dataset in dataset_names:
        scores = model2dataset2scores[model].get(dataset, [])
        
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