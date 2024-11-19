output_file="output_Ice$(date +'%Y-%m-%d_%H-%M-%S').log"

python3 -u main_meta.py | tee -a "$output_file"