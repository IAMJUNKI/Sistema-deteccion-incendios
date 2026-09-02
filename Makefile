SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: help deploy deploy-data deploy-all deploy-dry-run

help:
	@printf '%s\n' \
	  'Atajos de despliegue:' \
	  '  make deploy-dry-run  Validar configuración y mostrar la operación' \
	  '  make deploy          Publicar una release de código' \
	  '  make deploy-data     Sincronizar dataset, modelos y rejilla' \
	  '  make deploy-all      Sincronizar artefactos y publicar el código' \
	  '' \
	  'Configuración: copiar deploy/deploy.env.example a deploy/deploy.env.'

deploy:
	@bash scripts/deploy.sh

deploy-data:
	@bash scripts/sync_server_artifacts.sh

deploy-all:
	@bash scripts/sync_server_artifacts.sh
	@bash scripts/deploy.sh

deploy-dry-run:
	@bash scripts/deploy.sh --dry-run
