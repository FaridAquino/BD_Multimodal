import json
from src.core import Histogram
from src.audio.index import AcousticInvertedIndex
from src.db.connection import get_conn



def test_busqueda_lado_a():
    print("1. Descargando histogramas de la base de datos...")
    histograms_db = []
    
    with get_conn() as conn:
        # Traemos TODOS los histogramas para poder calcular los pesos TF-IDF globales
        rows = conn.execute("SELECT chunk_id, source_id, counts FROM histograms;").fetchall()
        
        for row in rows:
            # PostgreSQL a veces devuelve un string JSON y a veces un dict nativo
            counts_dict = row[2] if isinstance(row[2], dict) else json.loads(row[2])
            
            # Reconstruimos los objetos Python
            histograms_db.append(Histogram(
                chunk_id=row[0],
                source_id=row[1],
                counts=counts_dict,
                raw_embedding=[] 
            ))

    if not histograms_db:
        print("❌ La base de datos está vacía.")
        return

    print(f"   -> ¡{len(histograms_db)} fragmentos cargados!")

    print("\n2. Entrenando el Índice Invertido en memoria...")
    index = AcousticInvertedIndex()
    index.build(histograms_db)
    print(f"   -> Índice construido. Palabras únicas en el vocabulario: {len(index.idf)}")

    print("\n3. Preparando el fragmento de búsqueda (Chunk 4183)...")
    # Buscamos nuestro chunk de prueba (de la canción 5) dentro de los que ya descargamos
    query_hist = next((h for h in histograms_db if str(h.chunk_id) == '4183'), None)
    
    if not query_hist:
        print("❌ No se encontró el chunk 4183.")
        return
        
    print(f"   -> Query listo: Canción {query_hist.source_id} | Palabras: {query_hist.counts}")

    print("\n4. ¡Disparando la búsqueda híbrida TF-IDF!")
    resultados = index.search(query=query_hist, k=5)

    print("\n🏆 RESULTADOS DEL LADO A:")
    if not resultados:
        print("No se encontraron coincidencias válidas.")
    else:
        for idx, res in enumerate(resultados, 1):
            print(f"{idx}. Canción: {res.source_id} | Similitud de Documento: {res.score:.4f}")

if __name__ == "__main__":
    test_busqueda_lado_a()