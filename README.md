# Sistema Unificado de Recuperacion y Busqueda Multimodal

**Informe Tecnico**

Proyecto 2 -- Base de Datos 2
Universidad de Ingenieria y Tecnologia (UTEC), 2026-1

**Equipo:**
- Farid Aquino -- Tech Lead, Arquitectura e Infraestructura
- [Ing. Texto] -- Modulo de Texto
- [Vasco2510] -- Modulo de Imagenes
- [ChRi5-PT] -- Modulo de Audio
- [J-D-Rosales] -- Backend y Evaluacion

---

## Tabla de Contenido

1.  Descripcion del sistema y arquitectura
2.  Dataset utilizado y caracteristicas
3.  Detalles de implementacion por modulo
4.  Resultados experimentales
5.  Analisis de trade-offs y conclusiones
6.  Instrucciones de instalacion y uso

---
## PARTE B: DOCUMENTACION OPERATIVA

### B.1 Puesta en marcha

Requisitos: Docker + Docker Compose. (Para desarrollar en local: Python 3.11.)

```bash
cp .env.example .env          # ajusta credenciales si quieres

# Opcion A: desarrollar en local (Python en tu maquina, Postgres en Docker)
make db-only                  # levanta solo PostgreSQL + pgvector
pip install -r requirements.txt
uvicorn src.backend.main:app --reload

# Opcion B: todo en Docker (recomendado para la entrega/demo)
make up
```

La diferencia entre A y B es solo `DB_HOST` en el `.env` (`localhost` vs
`postgres`); el codigo no cambia. Comprobar que todo esta arriba:

```bash
curl http://localhost:8000/health     # {"status":"ok","pgvector":true}
```

Los scripts `docker/postgres/init/*.sql` crean extensiones, tablas e indices
automaticamente la primera vez que arranca el volumen. Para reiniciar la BD desde
cero: `make reset-db`.

### B.2 Probar el demo paso a paso (las 3 modalidades)

El repo incluye **muestras** en `data/samples/` (aproximadamente 1000 chunks por
modalidad) para reproducir todo sin descargar los datasets completos. Los
**modelos no se versionan**: cada quien genera los suyos (`models/`) al ingestar.
Prerrequisitos: **Docker** y **Python 3.11**.

```bash
# 1) Entorno Python
python -m venv .venv
source .venv/Scripts/activate        # Windows; en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2) Levantar PostgreSQL (pgvector). Crea el esquema vacio automaticamente.
docker compose up -d postgres

# 3) IMAGEN: la ingesta entrena el codebook K-Means y lo guarda en models/image/
python -m scripts.ingest --modality image --images data/samples/image --limit 112 --truncate
python -m scripts.build_index --modality image

# 4) TEXTO: la ingesta entrena el vocabulario top-k en models/text/
python -m scripts.ingest --modality text --csv data/samples/text/songs_sample.csv
python -m scripts.build_index --modality text

# 5) AUDIO: primero entrena el codebook K-Means sobre las muestras
python -m scripts.train_kmeans                       # -> models/audio/kmeans_256_fma.joblib
python -m scripts.ingest --modality audio --audio data/samples/audio --limit 25
python -m scripts.build_index --modality audio

# 6) Verificar conteos (aproximadamente 1000 chunks por modalidad)
docker exec -e PGPASSWORD=postgres multimodal_db psql -U postgres -d multimodal \
  -c "SELECT modality, count(DISTINCT source_id) AS fuentes, count(*) AS chunks \
      FROM chunks GROUP BY modality ORDER BY modality;"

# 7) Probar las busquedas: Lado A (indice propio) vs Lado B (nativo de Postgres)
python -m scripts.probe_query_image --image data/samples/image/10000.jpg --top 5
python -m scripts.probe_query_text  --q "love" --top 5
python -m scripts.probe_query_audio --query data/samples/audio/000002.mp3 --top 5
```

> Para empezar de cero (BD vacia): `docker compose down -v && docker compose up -d postgres`
> y repetir desde el paso 3.

### B.3 Estructura del repositorio

```
src/core/        Interfaces y pipeline (arquitectura unificada)
src/text|image|audio/   Implementacion por modalidad
src/db/          Conexion y repositorios (tablas compartidas)
src/baselines/   Comparativas nativas: GIN/GiST y pgvector
src/backend/     API FastAPI + las 2 aplicaciones
src/eval/        Benchmarks, metricas y graficos
docker/          Compose, Dockerfile y SQL de inicializacion
```

### B.4 Equipo y forma de trabajo

| Rol                         | Area principal                          |
|-----------------------------|-----------------------------------------|
| Tech Lead / Arquitectura    | `core/`, `db/`, `docker/`, integracion  |
| Ing. Texto                  | `text/`, `baselines/gin_gist.py`        |
| Ing. Imagenes               | `image/`, parte de `baselines/pgvector` |
| Ing. Audio                  | `audio/`, parte de `baselines/pgvector` |
| Ing. Backend / Evaluacion   | `backend/`, `eval/`                     |

El flujo de ramas, PRs y revisiones esta descrito en [`CONTRIBUTING.md`](CONTRIBUTING.md).
