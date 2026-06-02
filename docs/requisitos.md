# Requisitos — Sistema Multimodal de Recuperación y Búsqueda

> Documento de requisitos del Proyecto 2 (BD2, UTEC 2026-1).
> Resumen de alto nivel en el [README principal](../README.md);

## 1. Objetivo del sistema

Construir un motor de búsqueda unificado que recupere contenido de múltiples
modalidades (texto, imagen y audio) bajo una misma arquitectura
(`split → extractor → codebook → índice invertido`), y comparar la
implementación propia (Lado A) contra las técnicas nativas de PostgreSQL
(Lado B: GIN/GiST y pgvector).

## 2. Aplicaciones a implementar

### App 1 — Búsqueda Visual E-commerce
- **Modalidad:** imagen.
- **Entrada del usuario:** una imagen de un producto.
- **Salida:** los productos visualmente más similares (top-k).
- **Pipeline:** patches → SIFT → K-Means (visual words) → histogramas.

### App 2 — Búsqueda Musical Inteligente
- **Modalidad:** texto (letras) + audio.
- **Entrada del usuario:** texto (fragmento de letra) y/o un clip de audio.
- **Salida:** canciones que coinciden por letra (full-text) y/o por similitud
  acústica (top-k).
- **Pipeline texto:** párrafos → TF-IDF → top-k → SPIMI.
- **Pipeline audio:** ventanas → MFCC → K-Means (acoustic words) → histogramas.

> Cubrir estas dos apps ejercita las tres modalidades, lo que habilita la
> comparativa completa de la Fase 3.

## 3. Requisitos funcionales

| ID  | Requisito                                                                 |
|-----|---------------------------------------------------------------------------|
| RF1 | Indexar colecciones de texto, imagen y audio con el pipeline común.       |
| RF2 | Construir el índice invertido propio por modalidad (texto vía SPIMI).     |
| RF3 | Resolver consultas top-k con la implementación propia (Lado A).           |
| RF4 | Resolver las mismas consultas con GIN/GiST (texto) y pgvector (img/audio).|
| RF5 | Exponer las 2 aplicaciones vía la API (endpoints de búsqueda).            |
| RF6 | Persistir codebooks, histogramas y metadatos en PostgreSQL.               |
| RF7 | Soportar las cargas experimentales de 1K, 10K y 100K registros.           |

## 4. Requisitos no funcionales

| ID   | Requisito                                                                |
|------|--------------------------------------------------------------------------|
| RNF1 | Todo el entorno debe levantar con un comando (`docker compose up`).      |
| RNF2 | Ambos lados (A y B) deben partir de IDÉNTICA entrada → comparación justa.|
| RNF3 | Medir latencia, recall, uso de memoria y accesos a disco en cada carga.  |
| RNF4 | Código documentado; CI (ruff + pytest) en verde en cada PR.              |
| RNF5 | Reproducibilidad: cualquier integrante reproduce el entorno desde cero.  |

## 5. Alcance

**Incluido**
- Las dos aplicaciones (App 1 y App 2) y sus pipelines.
- Implementación propia + baselines nativos de PostgreSQL.
- Evaluación experimental en las tres cargas.


## 6. Decisiones pendientes del equipo

- [ ] Confirmar datasets concretos por modalidad (ver `data/README.md`).
- [ ] Definir `k` del codebook por modalidad (afecta `vector(k)` en el esquema SQL).
- [ ] Definir el conjunto de consultas de prueba para la evaluación.
