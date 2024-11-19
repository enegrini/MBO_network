# python3 -u data_generator.py --threshold 0.5 --kernel_kind hard2 --noise_kind blur --deltat 0.2
# python3 -u data_generator.py --threshold 0.2 --kernel_kind standard --noise_kind None --deltat 1.5 --dataset_size 1

python3 -u data_gen_metaLearning.py --kernel_kinds "['standard', 'hard', 'hard2', 'circle', 'MNIST']" \
--thresholds '0.2,0.3,0.5,0.6' --runs 5 --dataset_size 30 --noise_kinds "['None','blur', 'SP']" --deltat 0.8