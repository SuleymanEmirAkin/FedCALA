#!/bin/bash
echo "Starting"
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate openfgl

# Define variables
dataset_name="COLLAB"
model_name="fedcala"
num_clients=10
num_rounds=100
use_cuda=1
results_dir="experiments/${model_name}_${dataset_name}_nc_${num_clients}_results"
# Create the directory if it doesn't exist
mkdir -p "$results_dir"

# Run the loop 5 times
for i in {1..1}
do
   echo "Starting iteration $i..."
   
   # Define the output file path inside the new directory
   output_file="${results_dir}/${model_name}_${dataset_name}_nc_${num_clients}_run_${i}.txt"
   
   # Run python script:
   # 2>&1 merges Errors (stderr) into Output (stdout)
   # | tee "$output_file" saves everything to the file AND prints it to your screen
   python ./train.py "$dataset_name" "$model_name" "$num_clients" "$num_rounds" "$use_cuda" 2>&1 | tee "$output_file"
 
   echo "Iteration $i complete. Saved to $output_file"
done
echo "---------------------------------------"
echo "All 5 runs finished. Results are in: $results_dir"