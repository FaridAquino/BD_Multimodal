# Sistema Multimodal de Recuperación y Búsqueda

Proyecto 2 — Base de Datos 2 (UTEC, 2026-1). Motor de búsqueda unificado sobre
**texto, imágenes y audio** que aplica el mismo paradigma a las tres modalidades:

```
split  →  extractor  →  codebook  →  índice invertido
```

y compara esa implementación propia (**Lado A**) contra las técnicas nativas de
PostgreSQL (**Lado B**: GIN/GiST para texto, pgvector para imagen y audio).

Aplicaciones implementadas: **App 1 – Búsqueda Visual E-commerce** y
**App 2 – Búsqueda Musical Inteligente**.

## Arquitectura

El corazón es `src/core/`, que define el **contrato común** (interfaces
abstractas). Cada modalidad lo implementa de forma independiente:

| Etapa            | Interfaz (`core`)  | Texto              | Imagen            | Audio              |
|------------------|--------------------|--------------------|-------------------|--------------------|
| Split            | `Splitter`         | Párrafos           | Patches           | Ventanas 100–200ms |
| Extractor        | `Extractor`        | TF-IDF             | SIFT              | MFCC               |
| Codebook         | `CodebookBuilder`  | Top-k palabras     | K-Means (visual)  | K-Means (acústico) |
| Índice (Lado A)  | `InvertedIndex`    | **SPIMI**          | Histogramas       | Histogramas        |
| Baseline (Lado B)| —                  | GIN/GiST           | pgvector          | pgvector           |

`core/pipeline.py` orquesta las cuatro etapas sin conocer la modalidad concreta.

## Puesta en marcha

Requisitos: Docker + Docker Compose. (Para desarrollar en local: Python 3.11.)

```bash
cp .env.example .env          # ajusta credenciales si quieres

# Opción A — desarrollar en local (Python en tu máquina, Postgres en Docker)
make db-only                  # levanta solo PostgreSQL + pgvector
pip install -r requirements.txt
uvicorn src.backend.main:app --reload

# Opción B — todo en Docker (recomendado para la entrega/demo)
make up
```

La diferencia entre A y B es solo `DB_HOST` en el `.env` (`localhost` vs
`postgres`); el código no cambia. Comprobar que todo está arriba:

```bash
curl http://localhost:8000/health     # {"status":"ok","pgvector":true}
```

Los scripts `docker/postgres/init/*.sql` crean extensiones, tablas e índices
automáticamente la primera vez que arranca el volumen. Para reiniciar la BD desde
cero: `make reset-db`.

## Estructura

```
src/core/        Interfaces y pipeline (arquitectura unificada)
src/text|image|audio/   Implementación por modalidad
src/db/          Conexión y repositorios (tablas compartidas)
src/baselines/   Comparativas nativas: GIN/GiST y pgvector
src/backend/     API FastAPI + las 2 aplicaciones
src/eval/        Benchmarks, métricas y gráficos (Fase 4)
docker/          Compose, Dockerfile y SQL de inicialización
```

## Equipo y forma de trabajo

| Rol                         | Área principal                          |
|-----------------------------|-----------------------------------------|
| Tech Lead — Arquitectura/DevOps | `core/`, `db/`, `docker/`, integración, review |
| Ing. Texto                  | `text/`, `baselines/gin_gist.py`        |
| Ing. Imágenes               | `image/`, parte de `baselines/pgvector.py` |
| Ing. Audio                  | `audio/`, parte de `baselines/pgvector.py` |
| Ing. Backend / Evaluación   | `backend/`, `eval/`                     |

El flujo de ramas, PRs y reviews está en [`CONTRIBUTING.md`](CONTRIBUTING.md).
La planificación y los hitos se llevan en **GitHub Projects**.
