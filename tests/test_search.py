import sys
# Aseguramos que Python pueda ver la carpeta 'src'
sys.path.append(".")

from src.db.repositories import search_similar_audio, get_conn

def test_vector_search():
    print("=== 🔍 Probando Búsqueda por Similitud Vectorial (Lado B) ===")
    
    # 1. Extraemos un embedding real de la base de datos para usarlo como consulta
    with get_conn() as conn:
        cursor = conn.execute("SELECT chunk_id, embedding FROM embeddings_audio LIMIT 1;")
        row = cursor.fetchone()
        
        if not row:
            print("❌ No se encontraron embeddings en la base de datos. Recuerda correr el pipeline primero.")
            return
            
        target_chunk_id, embedding_raw = row
        
        # pgvector puede devolver el vector como string '[0.1,0.2,...]' o como lista dependiendo del driver.
        # Aquí manejamos ambos casos para evitar errores:
        if isinstance(embedding_raw, str):
            query_embedding = [float(x) for x in embedding_raw.strip("[]").split(",")]
        else:
            query_embedding = list(embedding_raw)

    print(f"\nSelected Chunk ID: {target_chunk_id} como 'Audio Objetivo' para la consulta.")
    print(f"Dimensiones del vector: {len(query_embedding)}")
    print("Enviando consulta a PostgreSQL (Operador de distancia de coseno `<=>`)...")
    
    # 2. Ejecutamos la función de búsqueda que escribimos en el repositorio
    results = search_similar_audio(query_embedding, limit=5)
    
    # 3. Desplegamos los resultados en la pantalla
    print("\n================ TOP 5 RESULTADOS ENCONTRADOS ================")
    print(f"{'Chunk ID':<10} | {'Similitud del Coseno':<22} | {'Estado':<20}")
    print("-" * 65)
    
    for chunk_id, similarity in results:
        # Si el ID coincide con el que usamos de query, es un clon perfecto
        if chunk_id == target_chunk_id:
            status = "✨ ¡Clon Exacto! (100%)"
        else:
            status = "🎵 Sonido Similar"
            
        print(f"{chunk_id:<10} | {similarity:<22.6f} | {status}")
    print("==============================================================")

if __name__ == "__main__":
    test_vector_search()