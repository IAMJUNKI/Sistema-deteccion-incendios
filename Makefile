SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: help deploy deploy-data deploy-runtime deploy-dataset deploy-all deploy-dry-run \
	build-operational-dataset train-egif-comparable train-egif-expanded train-egif-50-control \
	evaluate-model-families promote-egif-48

PYTHON ?= python
PYTHONPATH_ENV := PYTHONPATH=.
OPERATIONAL_DATASET := data/processed/tabular/egif_operational

help:
	@printf '%s\n' \
	  'Atajos de despliegue:' \
	  '  make deploy-dry-run  Validar configuración y mostrar la operación' \
	  '  make deploy          Publicar una release de código' \
	  '  make deploy-data     Sincronizar dataset, modelos y rejilla' \
	  '  make deploy-runtime  Sincronizar sólo modelos y rejilla operativos' \
	  '  make deploy-dataset  Sincronizar sólo el datacubo/datasets EGIF' \
	  '  make deploy-all      Sincronizar artefactos y publicar el código' \
	  '  make build-operational-dataset  Crear Parquet alineado con memoria T-1' \
	  '  make train-egif-comparable      Entrenar familia 48 comparable' \
	  '  make train-egif-expanded        Entrenar familia 48 ampliada' \
	  '  make train-egif-50-control      Entrenar control alineado de 50' \
	  '  make evaluate-model-families    Comparar familias y FWI' \
	  '' \
	  'Configuración: copiar deploy/deploy.env.example a deploy/deploy.env.'

deploy:
	@bash scripts/deploy.sh

deploy-data:
	@bash scripts/sync_server_artifacts.sh

deploy-runtime:
	@bash scripts/sync_server_artifacts.sh --no-dataset

deploy-dataset:
	@bash scripts/sync_server_artifacts.sh --no-models --no-grid

deploy-all:
	@bash scripts/sync_server_artifacts.sh
	@bash scripts/deploy.sh

deploy-dry-run:
	@bash scripts/deploy.sh --dry-run

build-operational-dataset:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/build_operational_benchmark.py \
	  --source-dir data/processed/tabular/egif \
	  --output-dir $(OPERATIONAL_DATASET) \
	  --years 2016-2023 \
	  --static-layer-quality provisional_import

train-egif-comparable:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/train_egif_operational.py \
	  --dataset-dir $(OPERATIONAL_DATASET) --output-dir data/models \
	  --feature-contract-version egif-2d-48-v1 --train-years 2019-2020 \
	  --calibration-year 2021 --validation-year 2022 --test-years 2023 \
	  --experiment-name comparable

train-egif-expanded:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/train_egif_operational.py \
	  --dataset-dir $(OPERATIONAL_DATASET) --output-dir data/models \
	  --feature-contract-version egif-2d-48-v1 --train-years 2016-2020 \
	  --calibration-year 2021 --validation-year 2022 --test-years 2023 \
	  --experiment-name expanded

train-egif-50-control:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/train_egif_operational.py \
	  --dataset-dir $(OPERATIONAL_DATASET) --output-dir data/models \
	  --feature-contract-version egif-2d-v1 --aligned-50-control \
	  --train-years 2019-2020 --calibration-year 2021 \
	  --validation-year 2022 --test-years 2023 --experiment-name aligned50

evaluate-model-families:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/evaluate_model_families.py \
	  --models-root data/models --dataset-dir $(OPERATIONAL_DATASET) \
	  --legacy-dataset-dir data/processed/tabular/egif --test-years 2023

promote-egif-48:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/promote_model_family.py \
	  --source-dir data/models/experiments/expanded \
	  --destination-dir data/models --dataset-dir $(OPERATIONAL_DATASET) \
	  --prefix forecast_risk_egif_48 --dry-run
