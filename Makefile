create-env:
	virtualenv -p python3 ./env

install-dependencies:
	pip install -r ./requirements.txt