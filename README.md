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

## Procesamiento de imágenes

La modalidad **imagen** aplica el paradigma común con la técnica clásica de
*Bag of Visual Words*:

```
imagen → patches (rejilla rows×cols) → SIFT (128-d) → codebook K-Means
       → histograma de visual words → índice invertido (Lado A) / pgvector (Lado B)
```

Módulos en `src/image/` (cada uno implementa una interfaz de `core`):

| Archivo                  | Clase                                   | Rol                                                        |
|--------------------------|-----------------------------------------|------------------------------------------------------------|
| `splitter.py`            | `PatchSplitter`                         | Parte la imagen en una rejilla `rows×cols`; un patch = un chunk. |
| `extractor.py`           | `SiftExtractor`                         | Descriptores SIFT (128-d) por patch.                       |
| `codebook.py`            | `KMeansVisualBuilder` / `VisualCodebook`| Entrena las *visual words* (K-Means) y codifica histogramas. |
| `index.py`               | `VisualInvertedIndex`                   | Índice invertido propio (Lado A), búsqueda por coseno.     |

**Persistencia.** Los datos se guardan en PostgreSQL (`sources`, `chunks`,
`codebooks`, `histograms` y `embeddings_image` con `vector(256)` = `k`) y los
modelos entrenados en `models/image/` (`codebook_image.npy` y el índice
`index_image.pkl`).

**Lado A vs Lado B.** El Lado A es el índice invertido propio sobre histogramas
(`index.py`); el Lado B usa pgvector (operador `<=>`, distancia coseno) sobre
`embeddings_image`. `scripts/probe_query.py` ejecuta ambos y compara rankings y
tiempos.

Ejecución del pipeline (con PostgreSQL levantado y desde la raíz):

```bash
python -m scripts.ingest --modality image --images data/samples/image --limit 112 --truncate
python -m scripts.build_index --modality image     # índice Lado A → models/image/index_image.pkl
python -m scripts.probe_query_image --image data/samples/image/10000.jpg --top 5
```

Pruebas de validación por etapa (sin tocar la BD) en `scripts/probes/image/`:
`probe_splitter.py`, `probe_extractor.py` y `probe_codebook.py`.

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

## Probar el demo paso a paso (las 3 modalidades)

El repo incluye **muestras** en `data/samples/` (≈1000 chunks por modalidad) para
reproducir todo sin descargar los datasets completos. Los **modelos NO se versionan**:
cada quien genera los suyos (`models/`) al ingestar. Prerequisitos: **Docker** y
**Python 3.11**.

```bash
# 1) Entorno Python
python -m venv .venv
source .venv/Scripts/activate        # Windows; en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2) Levantar PostgreSQL (pgvector). Crea el esquema vacío automáticamente.
docker compose up -d postgres

# 3) IMAGEN — la ingesta entrena el codebook K-Means y lo guarda en models/image/
python -m scripts.ingest --modality image --images data/samples/image --limit 112 --truncate
python -m scripts.build_index --modality image

# 4) TEXTO — la ingesta entrena el vocabulario top-k en models/text/
python -m scripts.ingest --modality text --csv data/samples/text/songs_sample.csv
python -m scripts.build_index --modality text

# 5) AUDIO — primero entrena TU propio codebook K-Means sobre las muestras
python -m scripts.train_kmeans                       # → models/audio/kmeans_256_fma.joblib
python -m scripts.ingest --modality audio --audio data/samples/audio --limit 25
python -m scripts.build_index --modality audio

# 6) Verificar conteos (≈1000 chunks por modalidad)
docker exec -e PGPASSWORD=postgres multimodal_db psql -U postgres -d multimodal \
  -c "SELECT modality, count(DISTINCT source_id) AS fuentes, count(*) AS chunks \
      FROM chunks GROUP BY modality ORDER BY modality;"

# 7) Probar las búsquedas — Lado A (índice propio) vs Lado B (nativo de Postgres)
python -m scripts.probe_query_image --image data/samples/image/10000.jpg --top 5
python -m scripts.probe_query_text  --q "love" --top 5
python -m scripts.probe_query_audio --query data/samples/audio/000002.mp3 --top 5
```

> Para empezar de cero (BD vacía): `docker compose down -v && docker compose up -d postgres`
> y repite desde el paso 3.

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
