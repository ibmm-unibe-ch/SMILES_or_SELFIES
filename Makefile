.ONESHELL:

SHELL := /bin/bash

CONDA_ENV_NAME=SoS
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

build-conda-from-req: ## Build the conda environment
	conda create -n $(CONDA_ENV_NAME) --copy -y python=$(PY_VERSION)
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	python -s -m pip install -r requirements.txt

build-conda-from-env:
	conda env create -n $(CONDA_ENV_NAME) -f environment.yml

download_10m:
	mkdir download_10m
	cd download_10m
	wget -O pubchem_10m.txt.zip https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pubchem_10m.txt.zip
	unzip -o pubchem_10m.txt.zip
	rm pubchem_10m.txt.zip
	split -l 1000000 pubchem-10m.txt pubchem_10m-split-
	rm pubchem-10m.txt

new-env:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	conda env export > environment.yml

new-req:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	pip freeze > requirements.txt

format:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	@echo -n "==> Checking that imports are properly sorted with isort..."
	@echo -n ""
	@isort .
	@echo -n "==> Checking that code is autoformatted with black..."
	@echo -n ""
	@black .
