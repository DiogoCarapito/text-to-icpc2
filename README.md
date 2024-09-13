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

### Paperspace setup with cuda

1. Remove all files

    ```bash
    rm -rf *
    ```

2. Clone git repo

    ```bash
    git clone https://github.com/DiogoCarapito/text-to-icpc2
    cd text-to-icpc2
    ```

3. Create a virtual environment (python3.11)

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4. execute the setup_cuda.sh

    ```bash
    chmod +x setup_cuda.sh
    ./setup_cuda.sh
    ```

install blinker manualy if it gives error

```bash
pip install --upgrade --ignore-installed blinker
pip show blinker
```

### CUDA

NVIDEA monitor

```bash
nvidia-smi
```

```bash
chmod +x monitor_gpu.sh
./monitor_gpu.sh
```
