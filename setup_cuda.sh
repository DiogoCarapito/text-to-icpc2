#!/bin/bash

# Create a .venv in python 3.11
python3.11 -m venv .venv
echo "Created .venv"

# Activate the virtual environment
source .venv/bin/activate
echo "Activated .venv"

# install requirements with make
make all

# Uninstall torch libraries
pip uninstall -y torch torchvision torchaudio

# Check which GPU is available
#nvcc --version

# Install torch for the available GPU
#pip install torch==2.1.1+cu121 torchvision==0.15.2+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# upgrade blinker
#pip install --upgrade --ignore-installed blinker
#pip show blinker

# Final check if torch is working properly
python -c "import torch; print(torch.__version__)"

# Check NVIDIA processes
nvidia-smi