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

dataset avalilabe at [https://huggingface.co/datasets/diogocarapito/text-to-icpc2](https://huggingface.co/datasets/diogocarapito/text-to-icpc2)

## Project scructure

- **etl.py** - Extration, Transform and Load data script
- **st-etl.py** - streamlit based dataset exploration
- **train.py** - training script
- **inference.py** - cli inference to test predictions
- **st-inference.py** - streamlit based inferece api
- **hf_cli.py** - hugging face interface

## cheat sheet

### venv

create venv

```bash
python3 -m venv .venv
```

activate venv

```bash
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

### paperspace setup with cuda

How to start:

- remove all files

```bash
rm -rf *
```

- clone git repo

```bash
git clone https://github.com/DiogoCarapito/text-to-icpc2
cd text-to-icpc2
````

- create a venv (python3.11)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- install blinker manualy if it gives error

```bash
pip install --upgrade blinker
pip show blinker
```

- uninstall torch libraries

```bash
pip uninstall torch torchvision torchaudio
```

- check which gpu is available

```bash
nvcc --version
```

- install torch for the available gpu

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
````

- final check if torch is working properly

```bash
python
```

```bash
python -c "import torch; print(torch.__version__)"
```

### CUDA

check NVidea processes

```bash
nvidia-smi
```