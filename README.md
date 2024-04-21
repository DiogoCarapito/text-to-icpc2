[![Github Actions Workflow](https://github.com/DiogoCarapito/text-to-icpc2/actions/workflows/main.yaml/badge.svg)](https://github.com/DiogoCarapito/text-to-icpc2/actions/workflows/main.yaml)

# text-to-ICPC2

## Project description
NLP project to transform clinical diagnosis into ICPC-2 codes for Portuguese Primary Care

## Plan
1. Simple prototipe wiht a model for ICD-10 from Huggingface
2. ETL
3. Selectin a pretrained model
4. Traingin model with specific data
5. Model evaluation
6. Model deployment wiht Streamlit
7. Model improvement wiht usage data


## Project scructure
- **app.py** - front end
- **etl.py** - Extration, Transform and Load data
- **basic_model_test.py** - ICD-10 prototipe



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

