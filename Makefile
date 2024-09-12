install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	pytest --cov=app --cov=etl --cov=utils --cov=pages tests/*.py

format:
	black . *.py utils/*.py tests/*.py

lint:
# pylint --disable=R,C,E1120,W0621,W0404,W0613,W0201,W1203 *.py utils/*.py tests/*.py
	pylint --disable=R,C,W0613 *.py utils/*.py tests/*.py
# W0613 - train.py:204:35: W0613: Unused argument 'context' (unused-argument)
    
#container-lint:
#	docker run -rm -i hadolint/hadolint < Dockerfile

refactor:
	format lint

deploy:
	echo "deploy not implemented"

all: install lint test format 
