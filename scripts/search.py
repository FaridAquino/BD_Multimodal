import os
import numpy as np
from src.audio.splitter import SlidingWindowSplitter
from src.audio.extractor import MfccExtractor
from src.audio.codebook import KMeansAcousticBuilder
from src.db.connection import get_conn, get_pool

MODEL_PATH = "data/models/kmeans_256_fma.joblib"
QUERY_AUDIO = "data/raw/fma_small/000/000005.mp3" 

def calcular_similitud_coseno(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def main():
    print("=== 🎧 MOTOR DE BÚSQUEDA HÍBRIDA (LADO A + LADO B) ===")
    print(f"🔍 Analizando: {QUERY_AUDIO}\n")

    # 1. Pipeline de Extracción de la Consulta
    builder = KMeansAcousticBuilder()
    codebook = builder.load_from_file(MODEL_PATH)
    splitter = SlidingWindowSplitter()
    extractor = MfccExtractor()

    chunks = splitter.split(QUERY_AUDIO, "query_track")
    descriptors = extractor.extract(chunks)

    # Construimos el Histograma Global de la Consulta (Lado A)
    # E Inicializamos el Vector Promedio (Lado B)
    query_histogram = np.zeros(256)
    vectores_query = []

    for desc in descriptors:
        hist = codebook.encode(desc)
        # Sumamos la palabra activada al histograma global
        for word_id, count in hist.counts.items():
            query_histogram[int(word_id)] += count
        vectores_query.append(desc.vector)

    vector_query_promedio = np.mean(vectores_query, axis=0).tolist()

    # 2. FASE 1: RECOVERY (Traer top 100 usando Lado B + HNSW)
    print("🚀 Fase 1: Recuperando top 100 candidatos vía HNSW...")
    
    sql_recovery = """
        SELECT source_id 
        FROM embeddings_audio
        GROUP BY source_id
        ORDER BY MIN(embedding <=> %s::vector) ASC
        LIMIT 100;
    """
    
    with get_conn() as conn:
        vector_str = "[" + ",".join(map(str, vector_query_promedio)) + "]"
        cursor = conn.execute(sql_recovery, (vector_str,))
        candidatos = [row[0] for row in cursor.fetchall()]

    # 3. FASE 2: RE-RANKING (Comparar Histogramas Globales del Lado A)
    print("📊 Fase 2: Re-clasificando candidatos con Histogramas Lado A...")
    
    # Traemos todos los fragmentos de histogramas de nuestros 100 candidatos
    sql_histograms = """
        SELECT source_id, counts 
        FROM histograms 
        WHERE source_id = ANY(%s);
    """
    
    with get_conn() as conn:
        cursor = conn.execute(sql_histograms, (candidatos,))
        rows = cursor.fetchall()

    # Agrupamos los micro-histogramas de la BD en Histogramas Globales por Canción
    canciones_db_histograms = {source_id: np.zeros(256) for source_id in candidatos}
    
    for source_id, counts_dict in rows:
        # PostgreSQL nos devuelve el jsonb como un diccionario de Python
        if counts_dict:
            for word_id, count in counts_dict.items():
                canciones_db_histograms[source_id][int(word_id)] += count

    # 4. Calcular Similitud Real entre Histogramas
    ranking_final = []
    for source_id, hist_vector in canciones_db_histograms.items():
        similitud = calcular_similitud_coseno(query_histogram, hist_vector)
        ranking_final.append((source_id, similitud))

    # Ordenamos de mayor a menor similitud
    ranking_final.sort(key=lambda x: x[1], reverse=True)

    # 5. Mostrar Resultados Definitivos
    print("\n=== 🏆 TOP 5 RECOMENDACIONES HÍBRIDAS ===")
    for i, (source_id, similitud) in enumerate(ranking_final[:5], 1):
        porcentaje = similitud * 100
        print(f"{i}. ID: {source_id} | Similitud de Histograma: {porcentaje:.2f}%")

    get_pool().close()

if __name__ == "__main__":
    main()