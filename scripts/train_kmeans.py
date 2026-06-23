import os
import random
import librosa
import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

# ==========================================
# CONFIGURACIÓN (Ajusta estas rutas)
# ==========================================
# La ruta a la carpeta donde está el FMA Small extraído
FMA_AUDIO_DIR = "data/raw/fma_small" 

# Dónde quieres guardar el modelo entrenado
MODEL_OUTPUT_PATH = "data/models/kmeans_256_fma.joblib" 
SAMPLE_SIZE = 500       # Canciones aleatorias a usar (500 es ideal para empezar)
K_CLUSTERS = 256        # Tu vocabulario acústico
N_MFCC = 20             # Las dimensiones de tu Lado B
BATCH_SIZE = 10000      # Número de vectores procesados por iteración de RAM

def train_global_vocabulary():
    print("🔍 1. Buscando archivos MP3...")
    all_mp3_files = list(Path(FMA_AUDIO_DIR).rglob("*.mp3"))
    
    if not all_mp3_files:
        print("❌ No se encontraron archivos .mp3 en la ruta especificada.")
        return

    print(f"🎵 Se encontraron {len(all_mp3_files)} pistas en total.")
    
    # Tomamos una muestra aleatoria para no quemar la computadora
    sample_size = min(SAMPLE_SIZE, len(all_mp3_files))
    sampled_files = random.sample(all_mp3_files, sample_size)
    print(f"🎲 Seleccionadas {sample_size} pistas aleatorias para entrenar.")

    # Inicializamos el modelo eficiente para grandes volúmenes de datos
    kmeans = MiniBatchKMeans(n_clusters=K_CLUSTERS, batch_size=BATCH_SIZE, random_state=42)
    
    # Acumulador temporal para ir procesando por lotes
    features_buffer = []
    processed_count = 0
    error_count = 0

    print("\n⚙️ 2. Extrayendo características y entrenando por lotes...")
    
    for i, file_path in enumerate(sampled_files, 1):
        try:
            # sr=None conserva el sample rate original, duration=30 asegura no leer de más
            y, sr = librosa.load(file_path, sr=None, duration=30.0)
            
            # Extraemos los MFCCs idénticos a los de tu pipeline
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            
            # Transponemos: (20 dimensiones, N frames) -> (N frames, 20 dimensiones)
            mfccs = mfccs.T 
            features_buffer.append(mfccs)
            processed_count += 1
            
            # Cada 50 canciones, entrenamos parcialmente el modelo y vaciamos la RAM
            if len(features_buffer) >= 50:
                X_batch = np.vstack(features_buffer)
                kmeans.partial_fit(X_batch)
                features_buffer = [] # Liberamos memoria
                print(f"   -> Progreso: {i}/{sample_size} canciones procesadas...")
                
        except Exception as e:
            # ¡CRÍTICO para FMA! Ignoramos los MP3s corruptos sin detener el script
            error_count += 1
            print(f"   ⚠️ Error leyendo {file_path.name} (se ignorará)")

    # Entrenamos con lo que haya quedado en el buffer final
    if features_buffer:
        X_batch = np.vstack(features_buffer)
        kmeans.partial_fit(X_batch)

    print("\n✅ 3. Entrenamiento completado.")
    print(f"   -> Pistas exitosas: {processed_count}")
    print(f"   -> Pistas corruptas/ignoradas: {error_count}")
    
    # Creamos el directorio si no existe y guardamos el modelo
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(kmeans, MODEL_OUTPUT_PATH)
    
    print(f"\n💾 ¡Modelo guardado con éxito en: {MODEL_OUTPUT_PATH}!")
    print(f"   Ahora tu pipeline puede cargar este archivo para clasificar audios universales.")

if __name__ == "__main__":
    train_global_vocabulary()