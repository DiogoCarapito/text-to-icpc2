#!/bin/bash


# Create a virtual environment (python3.11)
python3 -m venv .venv
source .venv/bin/activate

# install requirements with make
make all

# Uninstall torch libraries
pip uninstall -y torch torchvision torchaudio

# Check which GPU is available
nvcc --version

# Install torch for the available GPU
pip install torch==1.13.0+cu117 torchvision==0.15.2+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Final check if torch is working properly
python -c "import torch; print(torch.__version__)"

# Check NVIDIA processes
nvidia-smi