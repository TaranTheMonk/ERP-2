create-env:
	virtualenv -p python3 ./env

install-dependencies:
	pip install -r ./requirements.txt

process-raw-data:
	python ./scripts/process_raw_data.py

run:
	python ./src/main.py
