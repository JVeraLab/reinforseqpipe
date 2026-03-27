# This repository contains the pipeline for mRNA sequence optimization using Deep Q-Learning (DQN).

# Prerequisites:

1. Python 3.8+

2. Conda or Mamba (Required for ViennaRNA/import RNA)

# Clone repository: 

git clone https://gitlab.jveralab.net/mirasbbr/reinforseqpipe
cd reinforseqpipe

# Create the environment
conda create -n venv python=3.9 -y
conda activate venv

# Install Python requirements
pip install -r requirements.txt

# Verify installation:

python -c "import torch; import RNA; import Bio; print('Setup Complete')"

# Run the pipeline (example):

python src/main.py -w weights.json -f input.fasta -l 4 -p 4 -e 50 


# Guideline for plots and data

# ReinforSeqPipe Data Analysis Overview

The analysis code and associated visualizations are located in: `data_analysis/analysis.ipynb`

---

## 1. Episode Convergence Tests
These files identify the optimal number of training episodes for the model.

* **Folder:** `reinforseqpipe/data_analysis/episodes_convergence_tests_with_stop`
Contains data on the efficient number of episodes when using a parameter (**sigma**) that stops the optimization if no score improvement is observed for 5 consecutive training episodes.

* **Folder:** `reinforseqpipe/data_analysis/episodes_convergence_tests_no_stop`
Contains results for the efficient number of episodes when no stopping parameter (**sigma**) is applied. In this case, optimization continues until the maximum number of episodes is reached.

---

## 2. Optimized Sequences for Experiments

* **Folder:** `data_analysis/results_2025-05-11_10_episodes_experimental_dataset`
Contains 9 dataframes with three dataframes for each **$lambda$** parameter (1, 4, and 10). Each sequence was run three times per **$lambda$** to obtain statistically meaningful data. After sorting based on the **$MFECAI_{lambda}$** parameter, the three best sequences were selected for experiments.
**IMPORTANT:** This set was optimized with 10 episodes because the optimal episode simulations were completed after the physical constructs were already purchased and delivered.

* **Folder:** `data_analysis/results_2025-08-09_20_episodes`
Contains results for three different **$lambda$** parameters (1, 4, and 10) under ideal conditions (20 episodes of optimization). Each sequence was optimized three times, following the same methodology as the 10-episode dataset.

---

## 3. Hyperparameter Optimization Results

* **Folder:** `data_analysis/stresults_30_05`
This folder contains sensitivity test results for eight different parameters: **batch_size**, **decay**, **initial_epsilon** (or start epsilon), **final_epsilon**, **gamma_parameter**, **hidden_size**, **learning_rate**, and **update_rate**.
    * **Note on hidden_size:** Both the LSTM (mRNA sequence analysis) and the FNN (DQN decision making) used the same number of hidden neurons. While they were intended to be tested separately, they were kept identical due to time constraints.
    * **Note on initial_epsilon:** Tested in August; results confirmed that changing start epsilon was not a decisive hyperparameter.
    * **Summary:** Results for each parameter are provided in separate dataframes.

* **Folder:** `data_analysis/results_grid_search_2025-06-20`
Contains results of the grid search performed on the most significant hyperparameters. After comparing the best average and median performances, the optimal combination was selected for further use.

---

## 4. Time Efficiency and Codon Weights

* **Folder:** `data_analysis/time_experiments`
Contains results regarding time efficiency. Optimization was performed for **$lambda = 4$** only, as this was the primary parameter. The goal was to observe how mRNA length affects computation time.

* **Folder:** `data_analysis/tissue_results`
Contains results regarding the effect of codon weights. Several codon weight dictionaries were used to determine how the solution space is affected by different weighting values.

---

**Variable Definitions:**
* $lambda$: The weighting parameter used to balance objectives in the optimization score.
* $gamma\_parameter$: The discount factor used in the reinforcement learning update rule.
* $MFECAI_{lambda}$: The composite metric (Minimum Free Energy and Codon Adaptation Index) used to rank sequences.

## 5. Visualizations and Experimental Plots

All plots and images generated during the analysis are located in the `data_analysis/images/` folder.

* **Convergence and Optimization Performance:**
    * `Testing the most optimal number of episodes.tiff` & `Testing the most optimal number of episodes part 2.tiff`: These plots show the results of the simulations to find the best training duration for the model.
    * `Optimization process trajectories.tiff`: Visualizes how the sequences evolve and improve over the training iterations for three major parameteres (MFE, CAI and MFECAI$lambda$).
    * `Iterative method finds better solution.tiff`: A comparison demonstrating the advantage of the iterative approach (at least three times run) outperforms the single run.

* **Sequence Parameter Analysis:**
    * `Comparison of distributions of initial and final states.tiff`: Shows the statistical shift in MFECAI$lambda$ scores before and after optimization.
    * `The correlation between initial and final states.tiff`: Analyzes if the starting sequence MFECAI$lambda$ scores influence the final MFECAI$lambda$ scores.
    * `The change of major mRNA parameters.tiff`: Tracks the changes in mRNA metrics (MFECAI$lambda$, MFE, CAI, Effective number of codons, GC content, GC3 content).

* **Biological and Statistical Results:**
    * `Degradation times comparison.tiff`: Comparison of the stability and degradation rates of the optimized mRNA sequences using RT-qPCR.
    * `The activation of T cells comparison.tiff`: Experimental data showing the biological impact and efficiency of the optimized constructs in T cell activation using Flow Cytometry method.
    * `Paired T-test.tiff`: Statistical validation (Paired T-test) of the results to ensure the observed improvements are significant.

* **RT-qPCR Degradation Analysis:**
    * `The degradation curve Init1 Run 1.tiff` through `Run 4`: Shows RT-qPCR results for the first initial sequence across four experimental runs.
    * `The degradation curve Init2 Run 1.tiff` through `Run 4`: Shows RT-qPCR results for the second initial sequence across four experimental runs.
    * `The degradation curve Optim1 Run 1.tiff` through `Run 4`: Shows RT-qPCR results for the optimized sequence, demonstrating how the RL process affected stability across four runs for Init1 construct.
    * `data_analysis\images\The degradation curves.tiff`: A summary file that combines all RT-qPCR data to provide a clear comparison of degradation kinetics between initial and optimized constructs.

* **Comparative Benchmarking and Model Performance:**
    * `The LinearDesign vs ReinforSeqPipe lambda 1.tiff`, `lambda 4.tiff`, and `lambda 10.tiff`: Comparative analysis between ReinforSeqPipe and LinearDesign across different lambda values.
    * `The role of lambda in the MFE-CAI trade-off.tiff`: Visualizes how the lambda parameter shifts the balance between folding energy and codon usage bias.
    * `The iterative method of ReifnorSeqPipe vs LienarDesign determinism.tiff`: Highlights the performance difference between the iterative reinforcement learning approach and deterministic design methods.
    * `The influence of codon dicitonary on a final result.tiff`: Shows how different tissue-specific codon weights affect the resulting optimized solution.
    * `Time dependency and bottleneck in viennarna.tiff`: Analysis of computation time versus sequence length, identifying bottlenecks within the ViennaRNA package.


