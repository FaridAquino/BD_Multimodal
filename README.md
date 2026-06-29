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

## 1. Descripcion del sistema y arquitectura

### 1.1 El paradigma unificado multimodal

El sistema implementa un motor de busqueda que opera sobre tres modalidades de
datos (texto, imagen y audio) aplicando un mismo flujo conceptual de cuatro
etapas. La premisa fundamental del proyecto es demostrar que contenidos de
naturaleza radicalmente distinta pueden ser tratados con una arquitectura comun,
donde solo cambian las estrategias de segmentacion, extraccion y codificacion
propias de cada dominio.

Las cuatro etapas del pipeline son las siguientes:

1.  **Split**: el contenido se divide en unidades atomicas de procesamiento
    denominadas chunks. En texto corresponde a parrafos, en imagen a parches
    extraidos de una rejilla regular, y en audio a ventanas temporales
    deslizantes.
2.  **Extractor**: cada chunk se transforma en un vector numerico de
    caracteristicas. El texto se representa mediante pesos TF-IDF sobre tokens
    preprocesados; la imagen mediante descriptores locales SIFT de 128
    dimensiones; el audio mediante coeficientes MFCC.
3.  **Codebook**: los vectores extraidos se agrupan mediante un algoritmo de
    clustering (K-Means en sus variantes MiniBatch) para construir un
    diccionario finito de patrones representativos. Para texto, el codebook es
    simplemente el conjunto de los k terminos mas frecuentes del corpus. Para
    imagen y audio, son k palabras visuales o acusticas respectivamente.
4.  **Indice invertido**: cada fuente se codifica como un histograma de
    frecuencias sobre el codebook. Sobre estos histogramas se construye un
    indice invertido propio (Lado A) que permite recuperar las fuentes mas
    similares mediante similitud coseno o ponderacion TF-IDF.

A este flujo se le denomina Lado A (implementacion propia). Como contraparte,
el Lado B delega la logica de indizacion y busqueda a PostgreSQL, utilizando
su busqueda de texto completo con indices GIN para la modalidad de texto y la
extension pgvector con indices HNSW para las modalidades de imagen y audio.
Ambos lados parten exactamente de los mismos datos y codebooks, lo que permite
una comparacion directa de rendimiento y calidad de recuperacion.

El nucleo del sistema reside en `src/core/`, que define las interfaces
abstractas (`Splitter`, `Extractor`, `CodebookBuilder`, `InvertedIndex`) y el
orquestador `ModalityPipeline` que coordina las cuatro etapas sin conocer la
modalidad concreta. Cada modalidad implementa estas interfaces de forma
independiente en los modulos `src/text/`, `src/image/` y `src/audio/`.

### 1.2 Correspondencia entre etapas y modalidades

La tabla siguiente presenta, para cada etapa del pipeline, la interfaz comun
definida en `core` y la implementacion concreta que adopta cada modalidad.

| Etapa           | Interfaz (`core`)    | Texto              | Imagen               | Audio                |
|-----------------|----------------------|--------------------|----------------------|----------------------|
| Split           | `Splitter`           | `ParagraphSplitter`| `PatchSplitter`      | `SlidingWindowSplitter` |
| Extractor       | `Extractor`          | `TfidfExtractor`   | `SiftExtractor`      | `MfccExtractor`      |
| Codebook        | `CodebookBuilder`    | `TopKCodebookBuilder` | `KMeansVisualBuilder` | `KMeansAcousticBuilder` |
| Indice (Lado A) | `InvertedIndex`      | `SpimiIndex`       | `VisualInvertedIndex`  | `AcousticInvertedIndex` |
| Indice (Lado B) | --                   | GIN / GiST         | pgvector (HNSW)      | pgvector (HNSW)      |

Todas las modalidades comparten la misma estructura de cuatro etapas, pero
difieren en la complejidad computacional de cada una. La extraccion SIFT para
imagenes opera sobre descriptores de 128 dimensiones, mientras que los MFCC de
audio se limitan a 20 coeficientes. El codebook de texto es determinista (top-k
por frecuencia de termino), en tanto que los de imagen y audio requieren
entrenamiento no supervisado mediante MiniBatchKMeans. En el Lado B, la
modalidad de texto utiliza indices GIN sobre vectores de documento
(`to_tsvector`), mientras que imagen y audio emplean la extension pgvector con
indices HNSW para busqueda por similitud coseno.

### 1.3 Doble rol de PostgreSQL y flujo Lado A versus Lado B

PostgreSQL desempena dos funciones diferenciadas dentro del sistema. En
primer lugar, actua como almacen pasivo de los datos procesados: las fuentes
originales, los chunks resultantes de la segmentacion, los codebooks
entrenados y los histogramas de frecuencias se persisten en tablas
relacionales. Este almacenamiento permite reconstruir el estado del sistema
desde cero sin perder los resultados de la ingesta.

En segundo lugar, y de forma simultanea, PostgreSQL opera como motor de
busqueda activo en el Lado B. Sobre las mismas tablas que almacenan los
datos del Lado A, se construyen indices especializados: indices GIN sobre
vectores de documento TSVECTOR para la busqueda full-text en texto, e
indices HNSW sobre columnas de tipo `vector(n)` para la busqueda por
similitud coseno en imagen y audio. De esta forma, la base de datos no solo
conserva los datos, sino que los indexa con mecanismos nativos de PostgreSQL.

El flujo comparativo entre ambos lados es el siguiente:

```
                     Datos originales
                           |
                     Split + Extraccion
                           |
                     Codebook (comun)
                          / \
                         /   \
                        /     \
                       /       \
        Lado A (propio)         Lado B (PostgreSQL nativo)
              |                           |
    Histogramas de frecuencias    Embeddings densos (vector(n))
              |                           |
    Indice invertido en Python    Indice HNSW / GIN en PostgreSQL
              |                           |
    Busqueda por similitud        Busqueda por operador nativo
    coseno (manual)               (<=> para pgvector, @@ para texto)
```

Ambos lados parten de los mismos chunks y del mismo codebook entrenado. La
diferencia reside exclusivamente en la representacion intermedia (histogramas
discretos frente a embeddings continuos) y en el motor de indizacion y
recuperacion (Python en memoria frente a SQL con indices nativos).

---

## PARTE B: DOCUMENTACION OPERATIVA

## PARTE B: DOCUMENTACION OPERATIVA

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
