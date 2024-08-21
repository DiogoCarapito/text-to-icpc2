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

- **etl.py** - Extration, Transform and Load data
- **st-etl.py** - streamlit based dataset exploration
- **train.py** - train script
- **inference.py** - cli inference
- **st-inference.py** - streamlit based inferece api

[https://www.youtube.com/watch?v=H-Cgag672nU](https://www.youtube.com/watch?v=H-Cgag672nU)

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

### Docker

build docker image

```bash
docker build -t main:latest .
```
