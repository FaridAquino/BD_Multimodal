import os
import time
from pathlib import Path

# Importamos tu pipeline
from src.audio.splitter import SlidingWindowSplitter
from src.audio.extractor import MfccExtractor
from src.audio.codebook import KMeansAcousticBuilder
from src.db.connection import get_conn, get_pool
from src.db.repositories import insert_source, insert_chunks, insert_histograms

# ==========================================
# CONFIGURACIÓN
# ==========================================
FMA_DIR = "data/raw/fma_small" 
MODEL_PATH = "data/models/kmeans_256_fma.joblib"
CODEBOOK_ID = 1

def main():
    print("=== INICIANDO INGESTA MASIVA FMA SMALL ===")
    
    # 1. Cargar el modelo universal (¡Solo se hace UNA vez!)
    print(f"🔍 Cargando modelo K-Means global desde {MODEL_PATH}...")
    builder = KMeansAcousticBuilder()
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: No se encontró el modelo global. Ejecuta train_kmeans.py primero.")
        return
    codebook = builder.load_from_file(MODEL_PATH)
    
    # Asegurarnos de que el codebook existe en la BD
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO codebooks (id, modality, k) 
            VALUES (%s, 'audio', %s) 
            ON CONFLICT (id) DO NOTHING;
        """, (CODEBOOK_ID, codebook.size))

    # 2. Inicializar herramientas del pipeline
    splitter = SlidingWindowSplitter()
    extractor = MfccExtractor()

    # 3. Buscar todos los archivos MP3
    mp3_files = list(Path(FMA_DIR).rglob("*.mp3"))
    total_files = len(mp3_files)
    print(f"🎵 Se encontraron {total_files} canciones listas para procesar.\n")

    exitos = 0
    errores = 0
    start_time = time.time()

    # ==========================================
    # EL GRAN BUCLE
    # ==========================================
    for i, file_path in enumerate(mp3_files, 1):
        # Usamos el nombre del archivo sin extensión como ID (ej. "000005")
        track_id = file_path.stem 
        source_id = f"fma_track_{track_id}"
        
        print(f"[{i}/{total_files}] Procesando {track_id}...", end=" ", flush=True)

        try:
            # Fase 1: Cortar
            chunks = splitter.split(str(file_path), source_id)
            if not chunks:
                raise ValueError("No se generaron chunks (audio muy corto o vacío).")

            # Fase 2: Metadatos a BD
            insert_source(source_id=source_id, modality="audio", uri=str(file_path), metadata={"fma_id": track_id})
            insert_chunks(chunks)

            # Fase 3: Extraer y Traducir
            descriptors = extractor.extract(chunks)
            histograms = []
            for desc in descriptors:
                hist = codebook.encode(desc)
                hist.codebook_id = CODEBOOK_ID
                histograms.append(hist)

            # Fase 4: Histogramas y Vectores a BD
            insert_histograms(histograms)
            
            exitos += 1
            print("✅ OK")

        except Exception as e:
            errores += 1
            print(f"❌ ERROR: {e}")
            # El script continúa con la siguiente canción gracias a este bloque

    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    elapsed_time = (time.time() - start_time) / 60
    print("\n=== REPORTE DE INGESTA ===")
    print(f"⏱️ Tiempo total: {elapsed_time:.2f} minutos")
    print(f"✅ Canciones procesadas con éxito: {exitos}")
    print(f"❌ Canciones ignoradas por errores: {errores}")

    get_pool().close()

if __name__ == "__main__":
    main()