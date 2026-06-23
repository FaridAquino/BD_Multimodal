import os
from src.audio.splitter import SlidingWindowSplitter
from src.audio.extractor import MfccExtractor
from src.audio.codebook import KMeansAcousticBuilder
from src.audio.index import AcousticInvertedIndex
from src.db.connection import get_conn, get_pool
from src.db.repositories import insert_source, insert_chunks, insert_histograms

def main():
    print("=== Iniciando Pipeline Completo de Audio ===")
    
    # 1. Configuración inicial
    # Asegúrate de que esta ruta apunte a un MP3 válido en tu proyecto
    file_path = "data/raw/fma_small/000/000005.mp3" 
    source_id = "test_track_000005"
    codebook_id = 1  # Asumimos 1 para este test, el Tech Lead manejará esto luego
    
    if not os.path.exists(file_path):
        print(f"Error: No se encuentra el archivo {file_path}")
        return

    # 2. Partición del Audio (Splitter)
    print("\n1. Cortando el audio en ventanas...")
    splitter = SlidingWindowSplitter()
    chunks = splitter.split(file_path, source_id)
    print(f"   -> {len(chunks)} chunks generados.")

    # =========================================================
    # INSERCIÓN EN BD: Fase 1 (Source y Chunks)
    # =========================================================
    print("\n2. Guardando metadatos en PostgreSQL...")
    insert_source(source_id=source_id, modality="audio", uri=file_path, metadata={"title": "Test FMA"})
    insert_chunks(chunks)
    print("   -> Source y Chunks guardados. ¡IDs recuperados de la BD!")
    # Nota: Aquí cada 'chunk' dentro de la lista ya tiene su chunk.chunk_id real de Postgres.

    # 3. Extracción de Características (Extractor)
    print("\n3. Extrayendo características MFCC...")
    extractor = MfccExtractor()
    descriptors = extractor.extract(chunks)
    print(f"   -> {len(descriptors)} descriptores extraídos.")

    # 4. Creación del Diccionario de Sonidos (Codebook)
    print("\n4. Entrenando modelo K-Means (Codebook)...")
    builder = KMeansAcousticBuilder()

    ruta_modelo = "data/models/kmeans_256_fma.joblib"

    if not os.path.exists(ruta_modelo):
        print(f"\n❌ ERROR: No se encontró el modelo en {ruta_modelo}")
        print("   Debes ejecutar 'train_kmeans.py' primero para crear el vocabulario global.")
        return
        
    codebook = builder.load_from_file(ruta_modelo)
    print(f"   -> Codebook global cargado con {codebook.size} centroides.")

    # 5. Cuantización (Generar Histogramas)
    print("\n5. Generando histogramas (Bag of Audio Words)...")
    histograms = []
    for desc in descriptors:
        hist = codebook.encode(desc)
        # Asegurarnos de que el codebook_id esté asignado para la BD
        hist.codebook_id = codebook_id
        histograms.append(hist)
    print(f"   -> {len(histograms)} histogramas generados.")

    # =========================================================
    # INSERCIÓN EN BD: Fase 2 (Histogramas y Vectores)
    # =========================================================
    print("\n6. Guardando histogramas (Lado A) y vectores densos (Lado B) en BD...")
    
    with get_conn() as conn:
      
        conn.execute("""
            INSERT INTO codebooks (id, modality, k) 
            VALUES (1, 'audio', 256) 
            ON CONFLICT (id) DO NOTHING;
        """)
    
    
    insert_histograms(histograms)
    print("   -> ¡Inserción exitosa en tablas histograms y embeddings_audio!")

    # 6. Prueba del Índice Invertido Propio (Lado A en Memoria)
    print("\n7. Construyendo Índice Invertido local...")
    index = AcousticInvertedIndex()
    index.build(histograms)
    
    # Probamos buscar usando la misma canción como query
    resultados = index.search(histograms[0], k=3)
    print(f"   -> Búsqueda de prueba completada. Top 1 ID: {resultados[0].source_id if resultados else 'N/A'}")

    print("\n=== ¡Validación Exitosa! Todo el pipeline está conectado ===")

    get_pool().close()  # Cerrar el pool de conexiones al finalizar

if __name__ == "__main__":
    main()