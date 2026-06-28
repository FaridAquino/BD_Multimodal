import streamlit as st
import requests
import pandas as pd
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Sistema Multimodal", layout="wide")

st.title("Sistema Multimodal de Recuperación y Búsqueda")

tab1, tab2 = st.tabs(["🛒 Buscador de Ropa", "🎵 Buscador de Letras Musicales"])

with tab1:
    st.header("Buscador Visual E-commerce")
    uploaded_file = st.file_uploader("Sube una imagen del producto", type=["jpg", "jpeg", "png"])
    
    col_img, col_ctrl = st.columns([1, 2])
    
    with col_img:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen de consulta", width=250)
            
    with col_ctrl:
        side_visual = st.radio("Motor de Búsqueda (Visual):", ["Lado A: Algoritmo Propio", "Lado B: Postgres Nativo (pgvector)"], key="visual_side")
        side_v_param = "A" if "Lado A" in side_visual else "B"
        k_visual = st.slider("Resultados (k):", min_value=1, max_value=50, value=10, key="visual_k")
        
        if st.button("Buscar Producto") and uploaded_file is not None:
            with st.spinner("Buscando productos similares..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {"side": side_v_param, "k": k_visual}
                    response = requests.post(f"{API_URL}/visual/search", params=params, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        if results:
                            st.subheader("Resultados:")
                            cols = st.columns(4)
                            for idx, res in enumerate(results):
                                with cols[idx % 4]:
                                    st.markdown(f"**{res.get('product_name', 'Desconocido')}**")
                                    img_url = res.get('image_url')
                                    if img_url:
                                        try:
                                            st.image(img_url, use_container_width=True)
                                        except:
                                            st.write("*(Imagen no disponible)*")
                                    st.write(f"💵 Precio: {res.get('price', 'N/A')}")
                                    st.write(f"📊 Score: {res.get('score', 0)}")
                        else:
                            st.warning("No se encontraron resultados.")
                    else:
                        st.error(f"Error en la API: {response.text}")
                except Exception as e:
                    st.error(f"Error de conexión con el Backend: {e}")

with tab2:
    st.header("Buscador de Letras Musicales")
    query_text = st.text_input("Ingresa un extracto de la letra de la canción:")
    
    side_music = st.radio("Motor de Búsqueda (Música):", ["Lado A: SPIMI Propio", "Lado B: GIN Nativo"], key="music_side")
    side_m_param = "A" if "Lado A" in side_music else "B"
    k_music = st.slider("Resultados (k):", min_value=1, max_value=50, value=10, key="music_k")
    
    if st.button("Buscar Canción") and query_text:
        with st.spinner("Buscando canciones..."):
            try:
                params = {"q": query_text, "side": side_m_param, "k": k_music}
                response = requests.get(f"{API_URL}/music/search", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        st.subheader("Resultados:")
                        df = pd.DataFrame(results)
                        if not df.empty:
                            df = df.rename(columns={"song": "Canción", "artist": "Artista", "score": "Score", "source_id": "ID"})
                            cols_order = ["Canción", "Artista", "Score", "ID"]
                            cols_to_show = [c for c in cols_order if c in df.columns]
                            st.dataframe(df[cols_to_show], use_container_width=True)
                    else:
                        st.warning("No se encontraron resultados.")
                else:
                    st.error(f"Error en la API: {response.text}")
            except Exception as e:
                st.error(f"Error de conexión con el Backend: {e}")
