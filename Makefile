install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	pytest --cov=app --cov=etl --cov=utils tests/*.py

format:
	black . *.py utils/*.py tests/*.py

lint:
	pylint --disable=R,C *.py
    
#container-lint:
#	docker run -rm -i hadolint/hadolint < Dockerfile

refactor:
	format lint

deploy:
	echo "deploy not implemented"

all: install lint test format 
