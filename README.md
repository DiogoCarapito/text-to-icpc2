# text-to-ICPC2

[![Github Actions Workflow](https://github.com/DiogoCarapito/text-to-icpc2/actions/workflows/main.yaml/badge.svg)](https://github.com/DiogoCarapito/text-to-icpc2/actions/workflows/main.yaml)

## Project description

NLP project to transform clinical diagnosis into ICPC-2 codes for Portuguese Primary Care

## Plan

1. Simple prototipe with a model for ICD-10 from Huggingface
2. ETL
3. Select a pretrained model
4. Train a model with specific data
5. Model evaluation
6. Model deployment with Streamlit
7. Model improvement with usage data

## Hugging Face

### Dataset

[https://huggingface.co/datasets/diogocarapito/text-to-icpc2](https://huggingface.co/datasets/diogocarapito/text-to-icpc2)

### Model

[https://huggingface.co/diogocarapito/text-to-icpc2](https://huggingface.co/diogocarapito/text-to-icpc2)

## Demo

Demo available att [https://text-to-icpc2demo.streamlit.app](https://text-to-icpc2demo.streamlit.app)

## Project scructure

- **data/** - starter datasets
- **etl/** - creation of training dataset, from original tables to generated labels
- **augmentation/** - data augmentation scrips and streamlit interface
- **train/** - training algorithms
- **inference/** - interaction with the model
- **validation/** - validation algorithms for acessing model performance
- **utils/** - supporting scripts

## cheat sheet

### venv

create and activate .venv

```bash
python3.12 -m venv .venv &&
source .venv/bin/activate
```

### MLFlow

activate server

```bash
mlflow server --host 127.0.0.1 --port 8080
```

### Docker

build docker image

```bash
docker build -t main:latest .
```

### Paperspace setup with cuda

Remove all files

```bash
rm -rf *
```

Clone git repo

```bash
git clone https://github.com/DiogoCarapito/text-to-icpc2 && cd text-to-icpc2
```

Create a virtual environment (python3.11)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

install torch compatible with available gpu before make all

```bash
pip uninstall -y torch torchvision torchaudio && pip install torch==2.1.1+cu121 torchaudio==2.1.1+cu121 torchvision==0.16.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

manually upgrade blinker (it prevents errors)

```bash
pip install --upgrade --ignore-installed blinker && pip show blinker
```

install the rest of requirements.txt

```bash
make all
```

.sh file for setup automation (not working right now)

```bash
chmod +x train/setup_cuda.sh
source setup_cuda.sh
```

### CUDA

Check torch instalation

```bash
python -c "import torch; print(torch.__version__)"
```

NVIDIA monitor

```bash
nvidia-smi
```

Continuous NVIDIA monitor

```bash
chmod +x train/monitor_gpu.sh
./monitor_gpu.sh
```

### paperspace single command

```bash
rm -rf * && \
git clone https://github.com/DiogoCarapito/text-to-icpc2 && \
cd text-to-icpc2 && \
python3.11 -m venv .venv && \
source .venv/bin/activate && \
pip uninstall -y torch torchvision torchaudio && \
pip install torch==2.1.1+cu121 torchaudio==2.1.1+cu121 torchvision==0.16.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html && \
pip install --upgrade --ignore-installed blinker && \
pip show blinker && \
make all && \
python -c "import torch; print(torch.__version__)"
```
