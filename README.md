# Neural Networks for Threshold Dynamics Reconstruction
Code for paper "Neural Networks for Threshold Dynamics Reconstruction" by E. Negrini, A. Gao, A. Bowering, W. Zhu, L. Capogna.

**Abstract:** We introduce two convolutional neural network (CNN) architectures, inspired by the Merriman-Bence-Osher (MBO) algorithm and by  cellular automatons, to model and learn threshold-based dynamics for front evolution from video data. The first model, termed the (single-dynamics) MBO network, learns a specific kernel and threshold for each input video without adapting to new dynamics, while the second, a meta-learning MBO network, generalizes across diverse thresholding dynamics by adapting its parameters per input. Both models are evaluated on synthetic and real-world videos (ice melting and fire propagation), with performance metrics indicating effective reconstruction and extrapolation of evolving boundaries, even under noisy conditions. Empirical results highlight the robustness of both networks across varied synthetic and real-world dynamics.

**Data Generation Code:** <br /> 
"data_generator.py" generates data for MBO network<br />
"data_gen_metaLearning.py": generates data for metalearning MBO network<br />
"data_gen.sh": script to run data generators<br />
<br />
**MBO network training and testing**<br />
"Avg_runs.py": training of MBO network<br />
"Validation.py": testing of MBO network<br />
"runnerStandardK.sh": script to run training and testing of MBO network<br />
<br /> 
**Metalearning MBO network training and testing**<br />
"metalearning.py": defines training and testing loops for metalearning network<br />
"main_meta.py": runs training and testing on user defined data<br />
"run_meta.sh": script to run training and testing<br />
<br />
**Misc**<br />
"utils.py"






