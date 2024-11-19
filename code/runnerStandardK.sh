#!/bin/bash

#Define ground truth folder
GT_folder="K31standardM13S6_T7dT1.5_thr0.2_Noise_None"
# Define output file and folder for the additional code
output_folder="K31standardM13S6_T7dT1.5_thr0.2_Noise_None"
output_file="$output_folder/outputNoiseNone.log"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

##Run data_generator.py, Avg_runs.py, and validation.py with additional code and capture output
echo "Running No Noise case"
python3 -u data_generator.py --threshold 0.2 | tee -a $output_file
python3 -u Avg_runs.py --data_path $output_folder --device 'cuda:1' --kernel_size 31 --lr 1e-5 --batch_size 1 | tee -a $output_file 
python3 -u Validation.py --folder_path $output_folder --GT_path $GT_folder --kernel_size 31 | tee -a $output_file

echo "Script execution completed. Check $output_file  for details."