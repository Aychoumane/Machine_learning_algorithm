#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sentim_analyz
PYTHON_VERSION = 3.12.3
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Clear processed data files (.csv, .npz)
.PHONY: clear_processed
clear_processed:
	rm -f data/processed/*.csv
	rm -f data/processed/*.npz
	@echo ">>> Fichiers .csv et .npz supprimés dans data/processed"

## Remove the virtual environment
.PHONY: remove_environment
remove_environment:
	rm -rf .venv
	@echo ">>> Environnement virtuel supprimé."


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 reconnaissance_demotion
	isort --check --diff reconnaissance_demotion
	black --check reconnaissance_demotion

## Format source code with black
.PHONY: format
format:
	isort reconnaissance_demotion
	black reconnaissance_demotion

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> Virtualenv créé dans .venv. Activez-le avec :"
	@echo "source .venv/bin/activate"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) reconnaissance_demotion/dataset.py

## Generate features
.PHONY: features
features: data
	$(PYTHON_INTERPRETER) reconnaissance_demotion/features.py

## Train models
.PHONY: train
train: features
	$(PYTHON_INTERPRETER) reconnaissance_demotion/modeling/train.py

## Predict on test set
.PHONY: predict
predict: train
	$(PYTHON_INTERPRETER) reconnaissance_demotion/modeling/predict.py

## Launch GUI
.PHONY: gui
gui: train
	$(PYTHON_INTERPRETER) main.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
