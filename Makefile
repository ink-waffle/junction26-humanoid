SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
DC ?= docker compose -f compose.verda.yaml

.PHONY: help validate build up down logs ps aalto host-setup check discover

help:
	@printf '%s\n' \
	  'make validate         Validate scripts, Compose, and the Aalto Dockerfile' \
	  'make build            Build the Aalto scene image on the GPU host' \
	  'make up               Build and start the scene (run with sudo)' \
	  'make down             Stop the scene (run with sudo)' \
	  'make logs             Follow scene logs (run with sudo)' \
	  'make ps               Show container status (run with sudo)' \
	  'make aalto ARGS=check Run the repository scene checks in the container' \
	  'make host-setup       Prepare data directories and SSH firewall (run with sudo)' \
	  'make check            Check Xorg, VNC, and container status' \
	  'make discover TYPE=t  List matching Verda availability and Ubuntu images'

validate:
	bash -n aaltonomous/cloud/verda/*.sh
	$(DC) --env-file .env.example config --quiet
	docker buildx build --check --file aaltonomous/cloud/Dockerfile .

build:
	$(DC) build aalto-sim

up:
	bash aaltonomous/cloud/verda/start-instance.sh

down:
	$(DC) down

logs:
	$(DC) logs -f aalto-sim

ps:
	$(DC) ps

aalto:
	$(DC) exec aalto-sim aalto $(ARGS)

host-setup:
	bash aaltonomous/cloud/verda/setup-instance.sh

check:
	bash aaltonomous/cloud/verda/check-instance.sh

discover:
	bash aaltonomous/cloud/verda/discover.sh $(TYPE)
