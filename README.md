# Sistema Unificado de Recuperación y Búsqueda Multimodal

Proyecto 2 — Base de Datos 2 (UTEC, 2026-1). 
Motor de búsqueda unificado sobre
**texto, imágenes y audio** que aplica el mismo paradigma a las tres modalidades:
1. Split 
2. Extractor
3. codebook
4. Índice invertido


## 0. Levantar proyecto 
( poner para crear el venv para windows y para mac)
(poner el isntalar los requirements)
(poner el comando apra levantar el streamlet)

## 1. Introducción y Resumen del Proyecto
* Breve descripción del sistema multimodal.
* Justificación del paradigma unificado: ¿Por qué procesar Texto, Imágenes y Audio bajo el mismo flujo conceptual (Split + Extracción + Codebook + Índice)?

## 2. Casos de Uso / Aplicaciones Implementadas
### App 1: Búsqueda Visual E-commerce (Modo Imágenes)
* **Dataset utilizado:** Fashion Product Images Dataset (Kaggle).
* **Descripción:** El usuario carga una imagen de una prenda y el sistema recupera los artículos visualmente más similares.
### App 2: Búsqueda Musical Inteligente (Modo Texto + Audio)
* **Dataset utilizado:** Spotify Songs / FMA Dataset.
* **Descripción:** Búsqueda por fragmentos de letras de canciones o similitud acústica de ritmos.

## 3. Arquitectura del Sistema e Infraestructura
* **Diagrama de Arquitectura:** Flujo del pipeline unificado (Lado A vs Lado B).
* **Infraestructura de Datos:** Repositorio en Docker con PostgreSQL + extensión `pgvector`.
* **Doble Rol de PostgreSQL:** Explicación de cómo actúa como almacén pasivo de chunks (Lado A) y como motor de búsqueda vectorial activo (Lado B).

## 4. Detalles de Implementación por Modalidad

### 4.1. Módulo de Imágenes (¡Tu sección, Yuri!)
* **Splitter (`PatchSplitter`):** Explicación de la estrategia de división por grillas regulares de parches ($N \times M$) y el uso de técnicas de superposición (*overlap*).
* **Extractor (`SiftExtractor`):** Por qué se seleccionó el algoritmo SIFT, invarianza a la escala/rotación y la estructura de los descriptores locales de 128 dimensiones.
* **Codebook (`VisualCodebook` y `KMeansVisualBuilder`):** Proceso de clustering con `MiniBatchKMeans` para agrupar millones de vectores SIFT en $k$ palabras visuales (*visual words*).
* **Índice Invertido (`VisualInvertedIndex`):** Cómo se construye la bolsa de palabras visuales (*Bag of Visual Words*), generación de histogramas de frecuencias por imagen y cálculo de la similitud del coseno.
* **Baseline Nativo (`pgvector`):** Configuración de la extensión en Postgres, almacenamiento de embeddings densos y uso de índices HNSW/IVF para la comparación.

### 4.2. Módulo de Texto
* Tokenización, remoción de stopwords, stemming.
* Lado A: Implementación obligatoria de **SPIMI** para el índice invertido de texto.
* Lado B: Uso de búsquedas Full-Text nativas de PostgreSQL con índices `GIN/GiST` (`to_tsvector` y `@@`).

### 4.3. Módulo de Audio
* Ventanas de tiempo (100-200ms) y extracción de coeficientes MFCC.
* Generación del Codebook acústico mediante K-Means y su índice invertido.
* Baseline en `pgvector` para vectores acústicos.

## 5. Resultados Experimentales y Análisis de Rendimiento
*(Sección crítica coordinada con José)*
* **Configuración del Entorno:** Características de la máquina donde se corrieron las pruebas (CPU, RAM, Disco).
* **Tablas Comparativas:** Carga de datos escalonada (1K, 10K, 100K chunks) evaluando:
  * Tiempo de construcción del índice.
  * Latencia de consultas (búsqueda en Python vs. Postgres nativo).
  * Consumo de memoria/espacio en disco.
  * Calidad de recuperación (*Recall* / precisión).
* **Gráficos:** Curvas de latencia vs. cantidad de registros para cada modalidad.

## 6. Análisis de Trade-offs y Conclusiones
* Ventajas y desventajas de procesar en memoria con Python puro (Lado A) frente a delegar la lógica de vecindad e índices estructurados a la base de datos (Lado B).
* Lecciones aprendidas sobre la unificación de estructuras de datos multimodales.

## 7. Instrucciones de Instalación y Uso
### Requisitos Previos
* Docker y Docker Compose instalados.
* Python 3.10+ con dependencias virtuales (`pip install -r requirements.txt`).
### Despliegue Rápido
1. Levantar la base de datos: `docker-compose up -d`
2. Poblar el sistema y entrenar los codebooks: `python src/eval/populate.py`
3. Lanzar el backend API: `uvicorn src.backend.main:app --reload`

## Levantar frontend 


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
