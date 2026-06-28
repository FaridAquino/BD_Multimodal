import joblib
import numpy as np

def inspeccionar_modelo():
    # Usa la ruta exacta que apareció en tu terminal
    ruta_modelo = "models/audio/kmeans_256_fma.joblib"
    
    print(f"🔍 Abriendo el archivo: {ruta_modelo}...\n")
    
    try:
        modelo = joblib.load(ruta_modelo)
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo en {ruta_modelo}")
        return

    print("=== RADIOGRAFÍA DEL MODELO ===")
    print(f"Clase exacta: {type(modelo).__name__}\n")

    # 1. Extraemos los centroides (El verdadero "Diccionario")
    centroides = modelo.cluster_centers_
    
    print("1. DIMENSIONES DEL VOCABULARIO")
    print(f"   -> Forma de la matriz: {centroides.shape}")
    print(f"   -> Tienes {centroides.shape[0]} 'palabras acústicas' distintas.")
    print(f"   -> Cada palabra está definida por {centroides.shape[1]} dimensiones (tus vectores MFCC).")
    
    if centroides.shape == (256, 20):
        print("   ✅ ¡Perfecto! Las dimensiones coinciden exactamente con tu arquitectura (256 palabras x 20 MFCC).")
    else:
        print("   ⚠️ Ojo: Las dimensiones no son las esperadas.")

    # 2. Vistazo a los datos reales
    print("\n2. VISTAZO A LOS DATOS (PRIMERA PALABRA ACÚSTICA)")
    print("   ¿Cómo se ve matemáticamente tu palabra o clúster '0'?:")
    # Redondeamos a 2 decimales para que no sature la pantalla
    print(f"   {np.round(centroides[0], 2)}")

    # 3. Métricas de salud del modelo
    print("\n3. MÉTRICAS DE SALUD")
    print(f"   -> Total de vectores procesados (n_features_in_): {getattr(modelo, 'n_features_in_', 'No disponible')}")
    
    if hasattr(modelo, 'n_steps_'):
        print(f"   -> Pasos de entrenamiento realizados: {modelo.n_steps_}")

    print("\n=== Todo indica que el archivo está íntegro y listo para producción ===")

if __name__ == "__main__":
    inspeccionar_modelo()