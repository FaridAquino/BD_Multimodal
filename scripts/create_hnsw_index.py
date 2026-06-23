from src.db.connection import get_conn, get_pool

def crear_indice_vectorial():
    print("=== Optimizando Base de Datos Vectorial ===")
    print("⚙️ Creando índice HNSW en la tabla embeddings_audio...")
    
    # Asumimos que tu columna vectorial se llama 'embedding' 
    # y usamos la distancia coseno (vector_cosine_ops)
    query_sql = """
        CREATE INDEX IF NOT EXISTS embeddings_audio_hnsw_idx 
        ON embeddings_audio 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
    """
    
    try:
        with get_conn() as conn:
            conn.execute(query_sql)
        print("✅ ¡Índice HNSW creado con éxito!")
        print("🚀 Tu base de datos ahora puede buscar entre cientos de miles de vectores en milisegundos.")
    except Exception as e:
        print(f"❌ Ocurrió un error al crear el índice:\n{e}")
    finally:
        get_pool().close()

if __name__ == "__main__":
    crear_indice_vectorial()