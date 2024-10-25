install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	pytest --cov=etl --cov=train --cov=hf --cov=inference --cov=validation --cov=augmentation --cov=utils tests/*.py

format:
	black . */*.py
# utils/*.py tests/*.py

lint:
	pylint --disable=R,C etl/*.py train/*.py hf/*.py inference/*.py validation/*.py augmentation/*.py utils/*.py tests/*.py
# pylint --disable=R,C,E1120,W0621,W0404,W0613,W0201,W1203 *.py utils/*.py tests/*.py
# pylint --disable=R,C,W0613 etl/*.py train/*.py hf/*.py inference/*.py validation/*.py augmentation/*.py utils/*.py tests/*.py
# W0613 - git commit -atrain.py:204:35: W0613: Unused argument 'context' (unused-argument)
    
#container-lint:
#	docker run -rm -i hadolint/hadolint < Dockerfile

refactor:
	format lint

deploy:
	echo "deploy not implemented"

all: install lint test format 
