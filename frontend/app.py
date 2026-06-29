import os
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 15
DEFAULT_K = 10
MAX_K = 50

VISUAL_SIDE_OPTIONS = {
    "Lado A: Algoritmo Propio": "A",
    "Lado B: Postgres Nativo (pgvector)": "B",
}

MUSIC_SIDE_OPTIONS = {
    "Lado A: Algoritmo Propio": "A",
    "Lado B: Postgres Nativo": "B",
}

st.set_page_config(page_title="Sistema Multimodal", layout="wide")

st.title("Sistema Multimodal de Recuperación y Búsqueda")
st.markdown(
    "En este panel puedes buscar ropa usando una imagen o encontrar canciones por letra y audio. "
    "Selecciona las opciones y presiona buscar para consultar el backend local."
)


def call_api(method, path, params=None, files=None):
    url = f"{API_URL}{path}"
    try:
        if method == "GET":
            return requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        return requests.post(url, params=params, files=files, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        st.error(f"Error de conexión con el backend: {exc}")
        return None


def normalize_side(selection, options):
    return options.get(selection, "A")


def load_image_from_url_or_path(image_ref):
    if not image_ref:
        return None

    parsed = urlparse(image_ref)
    if parsed.scheme in ("http", "https"):
        return image_ref

    if os.path.exists(image_ref):
        try:
            return Image.open(image_ref)
        except Exception:
            return None

    return None


def display_visual_results(results):
    if not results:
        st.info("No se encontraron resultados similares.")
        return

    st.success(f"¡Se encontraron {len(results)} resultados!")
    st.subheader("Resultados visuales")

    cols = st.columns(4)
    for idx, result in enumerate(results):
        with cols[idx % 4]:
            st.markdown(f"**{result.get('product_name', 'Desconocido')}**")
            image_ref = load_image_from_url_or_path(result.get("image_url"))
            if image_ref:
                st.image(image_ref, use_container_width=True)
            else:
                st.info("Imagen no disponible")

            price = result.get("price")
            score = result.get("score")
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Precio", f"${price}" if price is not None else "N/A")
            with metric_cols[1]:
                st.metric("Score", f"{score:.4f}" if isinstance(score, (int, float)) else "N/A")
            st.divider()


def display_music_results(results):
    if not results:
        st.info("No se encontraron resultados para esa búsqueda.")
        return

    st.success(f"¡Se encontraron {len(results)} canciones!")
    st.subheader("Resultados musicales")

    df = pd.DataFrame(results)
    if df.empty:
        st.write("No hay datos para mostrar.")
        return

    df = df.rename(columns={
        "song": "Canción",
        "artist": "Artista",
        "score": "Score",
        "source_id": "ID",
    })
    selected_columns = [col for col in ["Canción", "Artista", "Score", "ID"] if col in df.columns]
    st.dataframe(df[selected_columns], use_container_width=True)


with st.tabs(["Buscador de Ropa", "Buscador de Letras Musicales"]) as tabs:
    tab1, tab2 = tabs

with tab1:
    st.header("Buscador Visual E-commerce")
    st.write("Sube una imagen y elige el motor visual para encontrar productos similares.")

    col_img, col_ctrl = st.columns([1, 2])
    with col_img:
        uploaded_file = st.file_uploader(
            "Sube una imagen del producto", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen de consulta", width=260)

    with col_ctrl:
        side_visual = st.radio(
            "Motor de Búsqueda (Visual):",
            list(VISUAL_SIDE_OPTIONS.keys()),
            key="visual_side",
        )
        k_visual = st.slider(
            "Resultados (k):",
            min_value=1,
            max_value=MAX_K,
            value=DEFAULT_K,
            key="visual_k",
        )
        search_button = st.button(
            "Buscar Producto", type="primary", use_container_width=True
        )

    if search_button:
        if uploaded_file is None:
            st.warning("⚠️ Por favor, sube una imagen antes de buscar.")
        else:
            st.divider()
            with st.spinner("🔍 Analizando imagen y buscando productos similares..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {
                    "side": normalize_side(side_visual, VISUAL_SIDE_OPTIONS),
                    "k": k_visual,
                }
                response = call_api("POST", "/visual/search", params=params, files=files)
                if response is None:
                    return

                if response.status_code == 200:
                    display_visual_results(response.json().get("results", []))
                else:
                    st.error(f"Error en la API ({response.status_code}): {response.text}")

with tab2:
    st.header("Buscador Musical")
    st.write("Busca canciones por letra o por fragmento de audio.")

    search_type = st.radio(
        "Tipo de Búsqueda:", ["Letra (Texto)", "Fragmento (Audio)"], horizontal=True
    )

    query_text = ""
    audio_file = None
    if search_type == "Letra (Texto)":
        query_text = st.text_input(
            "Ingresa un extracto de la letra de la canción:",
            placeholder="Ej. Is this the real life? Is this just fantasy?",
        )
    else:
        audio_file = st.file_uploader(
            "Sube un fragmento de la canción (.wav, .mp3, .ogg)",
            type=["wav", "mp3", "ogg"],
        )

    col1, col2 = st.columns(2)
    with col1:
        side_music = st.radio(
            "Motor de Búsqueda (Música):",
            list(MUSIC_SIDE_OPTIONS.keys()),
            key="music_side",
        )
    with col2:
        k_music = st.slider(
            "Resultados (k):",
            min_value=1,
            max_value=MAX_K,
            value=DEFAULT_K,
            key="music_k",
        )

    search_music_btn = st.button(
        "Buscar Canción", type="primary", use_container_width=True
    )

    if search_music_btn:
        if search_type == "Letra (Texto)" and not query_text.strip():
            st.warning("Por favor, ingresa al menos una palabra para buscar.")
        elif search_type == "Fragmento (Audio)" and audio_file is None:
            st.warning("Por favor, sube un archivo de audio para buscar.")
        else:
            st.divider()
            with st.spinner("🎶 Buscando coincidencias en la base de datos..."):
                params = {
                    "side": normalize_side(side_music, MUSIC_SIDE_OPTIONS),
                    "k": k_music,
                }
                if search_type == "Letra (Texto)":
                    params["q"] = query_text
                    response = call_api("GET", "/music/search", params=params)
                else:
                    audio_file.seek(0)
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    response = call_api("POST", "/music/search_audio", params=params, files=files)

                if response is None:
                    return

                if response.status_code == 200:
                    display_music_results(response.json().get("results", []))
                else:
                    st.error(f"Error en la API ({response.status_code}): {response.text}")
