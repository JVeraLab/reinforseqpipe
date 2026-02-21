# This repository contains the pipeline for mRNA sequence optimization using Deep Q-Learning (DQN).

# Prerequisites

1. Python 3.8+

2. Conda or Mamba (Required for ViennaRNA/import RNA)

# Clone repository: 
Run followign commands in Bash:
git clone 
cd 

# Create the environment
conda create -n mrna_env python=3.9 -y
conda activate mrna_env

# Install ViennaRNA via Bioconda
conda install -c bioconda viennarna -y

# Install Python requirements
pip install -r requirements.txt

# Verify installation:

python -c "import torch; import RNA; import Bio; import codon_bias; print('Setup Complete')"

# Run the pipeline

python src/main.py -w weights.json -f input.fasta -l 4 -p 4 -e 50 