import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Sistema Multimodal", layout="wide")

st.title("Sistema Multimodal de Recuperación y Búsqueda")

tab1, tab2 = st.tabs(["Buscador de Ropa", "Buscador de Letras Musicales"])

with tab1:
    st.header("Buscador Visual E-commerce")
    
    with st.container():
        col_img, col_ctrl = st.columns([1, 2])
        
        with col_img:
            uploaded_file = st.file_uploader("Sube una imagen del producto", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen de consulta", width=250)
                
        with col_ctrl:
            st.write("### Opciones de Búsqueda")
            side_visual = st.radio("Motor de Búsqueda (Visual):", ["Lado A: Algoritmo Propio", "Lado B: Postgres Nativo (pgvector)"], key="visual_side")
            side_v_param = "A" if "Lado A" in side_visual else "B"
            k_visual = st.slider("Resultados (k):", min_value=1, max_value=50, value=10, key="visual_k")
            search_button = st.button("Buscar Producto", type="primary", use_container_width=True)

    if search_button:
        if uploaded_file is None:
            st.warning("⚠️ Por favor, sube una imagen antes de buscar.")
        else:
            st.divider()
            with st.spinner("🔍 Analizando imagen y buscando productos similares..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {"side": side_v_param, "k": k_visual}
                    response = requests.post(f"{API_URL}/visual/search", params=params, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        if results:
                            st.success(f"¡Se encontraron {len(results)} resultados!")
                            st.subheader("Resultados:")
                            cols = st.columns(4)
                            for idx, res in enumerate(results):
                                with cols[idx % 4]:
                                    st.markdown(f"**{res.get('product_name', 'Desconocido')}**")
                                    img_url = res.get('image_url')
                                    
                                    # --- SOLUCIÓN DE REPLIEGUE PARA IMÁGENES LOCALES ---
                                    if img_url:
                                        try:
                                            # Primero intenta renderizarlo como una URL normal
                                            st.image(img_url, use_container_width=True)
                                        except:
                                            # Si falla (porque es una ruta de archivo local), intenta leerlo directamente del disco
                                            if os.path.exists(img_url):
                                                try:
                                                    local_img = Image.open(img_url)
                                                    st.image(local_img, use_container_width=True)
                                                except:
                                                    st.error("*(Error al cargar archivo local)*")
                                            else:
                                                st.warning("*(Imagen no encontrada)*")
                                    else:
                                        st.info("*(Sin ruta de imagen)*")
                                    # ----------------------------------------------------
                                    
                                    metric_cols = st.columns(2)
                                    with metric_cols[0]:
                                        st.metric("Precio", f"${res.get('price', 'N/A')}")
                                    with metric_cols[1]:
                                        st.metric("Score", f"{res.get('score', 0):.4f}")
                                    st.divider()
                        else:
                            st.info("No se encontraron resultados similares.")
                    else:
                        st.error(f" Error en la API ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Error crítico de conexión con el Backend: {e}")

with tab2:
    st.header("Buscador Musical")
    
    with st.container():
        search_type = st.radio("Tipo de Búsqueda:", ["Letra (Texto)", "Fragmento (Audio)"], horizontal=True)
        
        if search_type == "Letra (Texto)":
            query_text = st.text_input("Ingresa un extracto de la letra de la canción:", placeholder="Ej. Is this the real life? Is this just fantasy?")
            audio_file = None
        else:
            query_text = ""
            audio_file = st.file_uploader("Sube un fragmento de la canción (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
            
        col1, col2 = st.columns(2)
        with col1:
            side_music = st.radio("Motor de Búsqueda (Música):", ["Lado A: Algoritmo Propio", "Lado B: Postgres Nativo"], key="music_side")
            side_m_param = "A" if "Lado A" in side_music else "B"
        with col2:
            k_music = st.slider("Resultados (k):", min_value=1, max_value=50, value=10, key="music_k")
        search_music_btn = st.button("Buscar Canción", type="primary", use_container_width=True)
    
    if search_music_btn:
        if search_type == "Letra (Texto)" and not query_text.strip():
            st.warning(" Por favor, ingresa al menos una palabra para buscar.")
        elif search_type == "Fragmento (Audio)" and audio_file is None:
            st.warning("⚠️ Por favor, sube un archivo de audio para buscar.")
        else:
            st.divider()
            with st.spinner("🎶 Buscando coincidencias en la base de datos..."):
                try:
                    params = {"side": side_m_param, "k": k_music}
                    
                    if search_type == "Letra (Texto)":
                        params["q"] = query_text
                        response = requests.get(f"{API_URL}/music/search", params=params)
                    else:
                        audio_file.seek(0)
                        files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                        response = requests.post(f"{API_URL}/music/search_audio", params=params, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        if results:
                            st.success(f"¡Se encontraron {len(results)} canciones!")
                            st.subheader("Resultados Musicales:")
                            df = pd.DataFrame(results)
                            if not df.empty:
                                df = df.rename(columns={"song": "Canción", "artist": "Artista", "score": "Score", "source_id": "ID"})
                                cols_order = ["Canción", "Artista", "Score", "ID"]
                                cols_to_show = [c for c in cols_order if c in df.columns]
                                st.dataframe(df[cols_to_show], use_container_width=True)
                        else:
                            st.info("No se encontraron resultados para esa búsqueda.")
                    else:
                        st.error(f"❌ Error en la API ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"🚨 Error crítico de conexión con el Backend: {e}")