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

## 2. Dataset utilizado y caracteristicas

### 2.1 Dataset de imagenes: Fashion Product Images

El modulo de imagenes utiliza el dataset Fashion Product Images Small
disponible en Kaggle. Este conjunto contiene aproximadamente 44 000 imagenes
de productos de moda categorizados por tipo de prenda, genero y uso. Para el
presente proyecto se trabajo con una muestra representativa de 112 imagenes
almacenadas en `data/samples/image/`, cada una en formato JPEG con resolucion
variable.

Las imagenes se someten a un preprocesamiento que incluye conversion a escala
de grises y redimensionamiento uniforme antes de la etapa de division en
parches. Este paso garantiza que los descriptores SIFT se computen sobre
entradas de dimension homogenea.

### 2.2 Dataset de texto: Spotify Million Song Dataset

Para la modalidad de texto se emplea el Spotify Million Song Dataset
(Kaggle), que contiene letras de aproximadamente 1.2 millones de canciones
con metadatos asociados (artista, album, genero). La muestra utilizada en
este proyecto comprende aproximadamente 1000 chunks de letras almacenados en
`data/samples/text/songs_sample.csv`.

Cada cancion se segmenta en parrafos que constituyen los chunks individuales.
Sobre cada chunk se aplica tokenizacion, eliminacion de stopwords y stemming
Porter antes de calcular los pesos TF-IDF. El codebook de texto se construye
seleccionando los 5000 terminos mas frecuentes del corpus.

### 2.3 Dataset de audio: Free Music Archive (FMA)

El modulo de audio utiliza el dataset Free Music Archive (FMA), una coleccion
de pistas musicales de dominio publico organizadas por genero. La muestra del
proyecto incluye 25 pistas en formato MP3 almacenadas en `data/samples/audio/`.

Cada pista se segmenta en ventanas deslizantes de 150 milisegundos con un
solapamiento de 75 milisegundos (hop length). De cada ventana se extraen 20
coeficientes MFCC que constituyen el descriptor numerico. El codebook
acustico se entrena con 256 centroides mediante MiniBatchKMeans.

### 2.4 Resumen de datasets

La tabla siguiente consolida las caracteristicas principales de cada dataset
utilizado.

| Modalidad | Dataset                | Fuente                       | Muestras | Chunks generados | Tipo de chunk        |
|-----------|------------------------|------------------------------|----------|------------------|----------------------|
| Imagen    | Fashion Product Images | Kaggle                       | 112      | 1008             | Parche 3x3           |
| Texto     | Spotify Million Songs  | Kaggle                       | ~200     | ~1000            | Parrafo de letra     |
| Audio     | Free Music Archive     | github.com/mdeff/fma         | 25       | ~1000            | Ventana 150 ms       |

La cantidad de chunks por modalidad se mantiene deliberadamente equilibrada
en el orden de 1000 unidades para permitir comparaciones directas entre
modalidades en las pruebas de rendimiento. La dimensionalidad de los
descriptores varia significativamente: 128 para SIFT (imagen), 20 para MFCC
(audio) y variable para TF-IDF segun el vocabulario (texto, hasta 5000
terminos).

---

## 3. Detalles de implementacion por modulo

### 3.1 Modulo de imagenes

El modulo de imagenes reside en `src/image/` y materializa el paradigma
unificado bajo la tecnica clasica de Bag of Visual Words. Cada archivo
implementa una de las interfaces definidas en `src/core/`.

**Splitter (`PatchSplitter`).** La imagen se divide en una rejilla regular de
3 filas por 3 columnas, generando 9 parches por imagen. Cada parche constituye
un chunk independiente. No se aplica solapamiento entre parches adyacentes,
aunque la implementacion soporta parametros de overlap configurable para
futuros experimentos.

**Extractor (`SiftExtractor`).** Sobre cada parche se ejecuta el algoritmo
SIFT (Scale-Invariant Feature Transform) implementado por OpenCV. SIFT
produce un conjunto de descriptores locales de 128 dimensiones cada uno,
invariantes a cambios de escala, rotacion e iluminacion. La cantidad de
descriptores por parche depende del contenido textual de la imagen.

**Codebook (`KMeansVisualBuilder` y `VisualCodebook`).** Los descriptores
SIFT de todas las imagenes de entrenamiento se agrupan en k palabras visuales
mediante MiniBatchKMeans. El valor de k se fija en 256. Una vez entrenado el
modelo, `VisualCodebook` asigna cada descriptor SIFT a la palabra visual mas
cercana y produce un histograma de frecuencias de dimension k por imagen.

**Indice invertido (`VisualInvertedIndex`).** El histograma de cada imagen se
almacena en una estructura de indice invertido en memoria. La busqueda recibe
el histograma de la imagen consulta y recupera las imagenes mas similares
mediante similitud coseno entre histogramas. El indice se persiste en
`models/image/index_image.pkl`.

**Lado B (pgvector).** En paralelo, cada histograma (tratado como vector
denso de 256 dimensiones) se almacena en la tabla `embeddings_image` de
PostgreSQL con tipo `vector(256)`. Sobre esta columna se construye un indice
HNSW con distancia coseno. La busqueda en Lado B se ejecuta mediante el
operador `<=>` de pgvector.

### 3.2 Modulo de texto

El modulo de texto reside en `src/text/` y sigue el mismo esquema de cuatro
etapas, con particularidades propias del dominio linguistico.

**Splitter (`ParagraphSplitter`).** Cada cancion se segmenta en parrafos
utilizando saltos de linea dobles como delimitadores. Cada parrafo se
convierte en un chunk de texto.

**Extractor (`TfidfExtractor`).** El texto se somete a un pipeline de
preprocesamiento que incluye tokenizacion, eliminacion de stopwords (usando
el corpus de NLTK) y stemming Porter. Sobre los tokens resultantes se calcula
el peso TF-IDF, produciendo un vector disperso donde cada dimension
corresponde a un termino del vocabulario global.

**Codebook (`TopKCodebookBuilder` y `LinguisticCodebook`).** A diferencia de
las modalidades sensoriales, el codebook de texto no requiere entrenamiento
de clustering. Se seleccionan los k terminos con mayor frecuencia en el
corpus (k = 5000). `LinguisticCodebook` codifica cada chunk como un vector de
pesos TF-IDF limitado a estos k terminos.

**Indice invertido Lado A (`SpimiIndex`).** La implementacion obligatoria
para texto es SPIMI (Single-Pass In-Memory Indexing). El indice se construye
en una sola pasada sobre los chunks: cada termino apunta a una lista
ordenada de pares (id_chunk, peso TF-IDF). SPIMI maneja la construccion en
bloques cuando el volumen de datos excede la memoria disponible. Como
alternativa, el repositorio incluye `BsbiIndex` (Blocked Sort-Based Indexing)
con fines comparativos.

**Lado B (GIN / GiST).** PostgreSQL implementa busqueda full-text nativa
mediante los operadores `to_tsvector` y `@@`. Sobre la columna `tsv` de tipo
TSVECTOR en la tabla `chunks` se construye un indice GIN. La busqueda recibe
una cadena de texto, la convierte a `tsquery` y recupera los chunks que
satisfacen la condicion de coincidencia, ordenados por relevancia.

### 3.3 Modulo de audio

El modulo de audio reside en `src/audio/` y procesa senales musicales
siguiendo el flujo unificado.

**Splitter (`SlidingWindowSplitter`).** Cada pista de audio se segmenta en
ventanas deslizantes de 150 milisegundos de duracion con un solapamiento de
75 milisegundos (50% de hop length). Cada ventana constituye un chunk. Este
solapamiento garantiza que no se pierdan eventos acusticos que ocurren en los
limites entre ventanas contiguas.

**Extractor (`MfccExtractor`).** De cada ventana se extraen 20 coeficientes
MFCC (Mel-Frequency Cepstral Coefficients) utilizando la biblioteca librosa.
Los MFCC capturan las caracteristicas espectrales de la senal de forma
compacta y son el descriptor estandar en tareas de recuperacion de informacion
musical.

**Codebook (`KMeansAcousticBuilder` y `AcousticCodebook`).** Los vectores
MFCC de todas las pistas de entrenamiento se agrupan en k palabras acusticas
(k = 256) mediante MiniBatchKMeans. El codebook entrenado se persiste en
`models/audio/kmeans_256_fma.joblib`.

**Indice invertido (`AcousticInvertedIndex`).** Cada pista se codifica como un
histograma de frecuencias de palabras acusticas (bag-of-audio-words), con
ponderacion TF-IDF para atenuar las palabras acusticas muy frecuentes. La
busqueda recupera las pistas con mayor similitud coseno respecto al histograma
de la consulta.

**Lado B (pgvector).** De forma analoga a imagen, los histogramas se
almacenan como vectores densos de 256 dimensiones en la tabla
`embeddings_audio` con tipo `vector(256)` e indice HNSW para busqueda por
similitud coseno.

### 3.4 Infraestructura comun

La infraestructura del sistema se apoya en tres componentes principales.

**PostgreSQL 16 con pgvector.** La base de datos se ejecuta en un contenedor
Docker basado en la imagen `pgvector/pgvector:pg16`. La inicializacion
automatica ejecuta tres scripts SQL: `01_extensions.sql` (activa las
extensiones `vector` y `pg_trgm`), `02_schema.sql` (crea las tablas
`sources`, `chunks`, `codebooks`, `histograms`, `codewords_text`,
`embeddings_image` y `embeddings_audio`) y `03_indexes.sql` (construye
indices GIN sobre TSVECTOR y HNSW sobre columnas vectoriales). El esquema
compartido permite que el Lado A y el Lado B operen sobre los mismos datos
persistidos.

**FastAPI (backend).** La capa de servicios expone tres endpoint principales:
`GET /health` para verificar el estado del sistema, `POST /visual/search`
para busqueda de imagenes, y endpoints en `/music/` para busqueda por letras
(`GET /music/search`) y por similitud acustica (`POST /music/search_audio`).
Cada endpoint implementa internamente la logica de ambos lados (A y B) y
retorna los resultados de forma independiente para su comparacion.

**Streamlit (frontend).** La interfaz de usuario se implementa con Streamlit
en `frontend/app.py`, proporcionando una visualizacion web interactiva de las
dos aplicaciones del proyecto.

**Docker Compose.** El archivo `docker-compose.yml` orquesta dos servicios:
`postgres` (con la imagen pgvector y los scripts de inicializacion montados
como volumen) y `app` (construida a partir del Dockerfile en `docker/app/`,
que ejecuta uvicorn con FastAPI). La aplicacion puede ejecutarse tambien en
modalidad hibrida (Python local, PostgreSQL en Docker) mediante `make db-only`.

---

## 4. Resultados experimentales

### 4.1 Configuracion del entorno

Las pruebas experimentales se ejecutaron sobre la siguiente configuracion de
hardware y software.

[PENDIENTE: completar con las especificaciones de la maquina donde se
corrieron las pruebas: modelo de CPU, frecuencia, cantidad de nucleos, RAM
total, tipo de disco, sistema operativo y version de Python]

### 4.2 Tiempos de construccion del indice

La tabla siguiente presenta los tiempos de construccion del indice invertido
(Lado A) y del indice nativo de PostgreSQL (Lado B) para cada modalidad,
medidos en segundos sobre cargas progresivas de 1K, 10K y 100K chunks.

| Modalidad | Lado   | 1K chunks | 10K chunks | 100K chunks |
|-----------|--------|-----------|------------|-------------|
| Texto     | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Texto     | B (GIN)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Imagen    | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Imagen    | B (HNSW)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Audio     | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Audio     | B (HNSW)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

[PENDIENTE: completar los valores tras ejecutar la bateria experimental.
Se espera que el Lado A (indice en memoria) sea mas rapido en construccion
para volumenes pequenos, pero que el Lado B escale mejor al delegar la
indexacion a estructuras de datos optimizadas en C.]

### 4.3 Latencia de consultas

La tabla recoge la latencia promedio de las consultas top-5 para cada
modalidad y cada lado, medida en milisegundos sobre cargas de 1K, 10K y
100K chunks.

| Modalidad | Lado   | 1K chunks | 10K chunks | 100K chunks |
|-----------|--------|-----------|------------|-------------|
| Texto     | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Texto     | B (GIN)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Imagen    | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Imagen    | B (HNSW)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Audio     | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Audio     | B (HNSW)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

[PENDIENTE: completar tras ejecutar la bateria. Se anticipa que el Lado A
presente menor latencia en conjuntos pequenos por evitar la sobrecarga de la
conexion a base de datos, mientras que el Lado B sea mas estable al crecer
el volumen gracias a los indices HNSW optimizados.]

### 4.4 Consumo de recursos y calidad de recuperacion

**Uso de memoria y espacio en disco.** La siguiente tabla resume el consumo
de recursos para cada modalidad en la carga de 10K chunks.

[PENDIENTE: incluir tabla con columnas: Modalidad, Lado, Memoria RAM (MB),
Espacio en disco (MB)]

**Recall@k.** La calidad de recuperacion se evalua mediante la metrica
recall@k, que mide la proporcion de resultados relevantes recuperados entre
los k primeros puestos.

| Modalidad | Lado   | Recall@1 | Recall@5 | Recall@10 |
|-----------|--------|----------|----------|-----------|
| Texto     | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Texto     | B (GIN)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Imagen    | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Imagen    | B (HNSW)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Audio     | A      | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Audio     | B (HNSW)| [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

[PENDIENTE: completar tras definir el conjunto de consultas de prueba y
ejecutar la evaluacion. La metrica ground truth se establece mediante
inspeccion manual o mediante el consenso entre ambos lados para las
coincidencias en los primeros puestos.]

**Curvas de latencia.** A continuacion se presentan las curvas de latencia
promedio en funcion de la cantidad de registros para cada modalidad, con
escala logaritmica en el eje de registros.

[PENDIENTE: insertar grafico generado por `src/eval/plots.py` con tres
subgraficos (uno por modalidad) comparando Lado A vs Lado B.]

---

## 5. Analisis de trade-offs y conclusiones

### 5.1 Lado A (indice propio en memoria) versus Lado B (PostgreSQL nativo)

La comparacion entre ambos lados revela un conjunto de ventajas y desventajas
que dependen del volumen de datos, la modalidad y los requisitos operativos.

**Ventajas del Lado A.** La implementacion en memoria elimina la latencia de
red y la sobrecarga del planificador de consultas SQL. Esto se traduce en
menores tiempos de respuesta para volumenes de datos pequenos y medianos.
Ademas, el control total sobre la estructura de datos permite implementar
variantes algoritmicas (como SPIMI) que no estan disponibles en PostgreSQL.
La portabilidad es otra ventaja: el indice invertido puede serializarse y
trasladarse entre entornos sin depender de una base de datos.

**Desventajas del Lado A.** La principal limitacion es la escalabilidad: el
indice reside en memoria RAM y su tamaño maximo esta acotado por la memoria
disponible. Para volumenes del orden de 100K chunks o superiores, la
construccion y la busqueda pueden degradarse. Ademas, la ausencia de
mecanismos nativos de concurrencia y persistencia transaccional requiere
gestion explicita de bloqueos y puntos de recuperacion.

**Ventajas del Lado B.** PostgreSQL ofrece indices optimizados en C (GIN,
HNSW) que escalan a volumenes considerablemente mayores sin comprometer la
memoria del proceso Python. La base de datos gestiona automaticamente la
concurrencia, la integridad transaccional y la recuperacion ante fallos.
El indice HNSW, en particular, ofrece tiempos de busqueda logaritmicos
respecto al tamaño del conjunto de datos.

**Desventajas del Lado B.** La dependencia de una base de datos externa
introduce latencia de red y costos de conexion. Para conjuntos pequenos
(1K chunks), la sobrecarga de establecer una conexion y ejecutar un plan
de consulta puede superar el tiempo neto de busqueda. Ademas, la
flexibilidad algoritmica es menor: solo se dispone de los operadores y
tipos de indice que PostgreSQL ofrece.

### 5.2 Lecciones aprendidas sobre la unificacion multimodal

La arquitectura unificada demostro ser viable para las tres modalidades
estudiadas. La abstraccion en interfaces (Splitter, Extractor,
CodebookBuilder, InvertedIndex) permitio que cada modulo se desarrollara de
forma independiente y que el orquestador `ModalityPipeline` permaneciera
agnostico al tipo de dato.

No obstante, se identificaron diferencias cualitativas que limitan la
simetria perfecta del paradigma. El codebook de texto (top-k terminos) es
cualitativamente distinto al de imagen y audio (centroides K-Means): el
primero es determinista y linguisticamente interpretable, mientras que los
segundos son probabilisticos y opacos. La busqueda full-text de PostgreSQL
(GIN) opera sobre tokens linguisticos, no sobre vectores densos, lo que
impide una comparacion directa con pgvector dentro de la misma modalidad.

### 5.3 Limitaciones y trabajo futuro

[PENDIENTE: completar tras la fase de evaluacion. Incluir limitaciones
identificadas como la ausencia de overlap en PatchSplitter, la dependencia
de un unico valor de k para los codebooks, y la necesidad de conjuntos de
prueba mas grandes para validar las curvas de escalabilidad.]

---

## 6. Instrucciones de instalacion y uso

### 6.1 Requisitos previos

- Docker y Docker Compose instalados.
- Python 3.11 o superior.
- Git para clonar el repositorio.

### 6.2 Pasos de instalacion

1.  Clonar el repositorio y crear el entorno virtual:

    ```bash
    git clone <url-del-repositorio>
    cd <directorio-del-proyecto>
    python -m venv .venv
    source .venv/Scripts/activate    # Windows
    # source .venv/bin/activate      # Linux / Mac
    pip install -r requirements.txt
    ```

2.  Copiar el archivo de configuracion de entorno:

    ```bash
    cp .env.example .env
    ```

3.  Levantar PostgreSQL con pgvector:

    ```bash
    docker compose up -d postgres
    ```

4.  Verificar que el sistema esta operativo:

    ```bash
    curl http://localhost:8000/health
    ```

### 6.3 Ingesta de datos y construccion de indices

Para cada modalidad, ejecutar la ingesta y la posterior construccion del
indice invertido. Los comandos detallados para cada modalidad se encuentran
en la seccion B.2 de este documento.

### 6.4 Ejecucion de consultas

Las busquedas pueden realizarse a traves de la API REST (FastAPI), de la
interfaz web (Streamlit) o mediante los scripts de prueba incluidos en
`scripts/`. Los endpoints disponibles son:

- `GET /health`: verifica el estado del sistema.
- `POST /visual/search`: busqueda de imagenes por similitud visual.
- `GET /music/search`: busqueda de canciones por letra.
- `POST /music/search_audio`: busqueda de canciones por similitud acustica.

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
