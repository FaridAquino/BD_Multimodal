.PHONY: up down logs db-only ingest bench test lint fmt

up:            ## Levanta Postgres + app (Opción B)
	docker compose up --build

db-only:       ## Solo Postgres (Opción A: corre Python en local)
	docker compose up -d postgres

down:          ## Apaga y conserva datos
	docker compose down

reset-db:      ## Apaga y BORRA el volumen (re-ejecuta los .sql de init)
	docker compose down -v

logs:
	docker compose logs -f

ingest:        ## Carga datasets en la BD
	python -m scripts.ingest

bench:         ## Corre la batería experimental (Fase 4)
	bash scripts/run_experiments.sh

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests
