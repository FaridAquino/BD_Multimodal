"""Frontend Streamlit — Búsqueda Multimodal (paleta de colores fríos pastel)."""
import os
from urllib.parse import urlparse

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"
# La canción completa tarda más en procesarse (ventaneo + MFCC de todo el mp3).
REQUEST_TIMEOUT = 120
DEFAULT_K = 10
MAX_K = 50

# La canción se envía COMPLETA y sin re-muestrear (igual que scripts.probe_query_audio):
# cualquier recorte o cambio de sample rate altera los MFCC y degrada el ranking.

VISUAL_SIDE_OPTIONS = {
    "Lado A: Índice invertido propio": "A",
    "Lado B: Postgres (pgvector)": "B",
}

MUSIC_SIDE_OPTIONS = {
    "Lado A: Índice invertido propio": "A",
    "Lado B: Postgres nativo": "B",
}

# ── Paleta fría pastel ───────────────────────────────────────────────────────
VERDE = "#CBE7AD"
AGUA = "#A9E0D4"
CELESTE = "#B3DCF5"
AZUL = "#8FA9DC"
LAVANDA = "#C0B0E4"
MORADO = "#AC96D4"
TINTA = "#33415E"
FONDO = "#F4F9FC"

st.set_page_config(page_title="Búsqueda Multimodal", page_icon="🗄️", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background: {FONDO}; }}

    /* Encabezado con degradado de fríos pastel */
    .hero {{
        background: linear-gradient(90deg, {VERDE} 0%, {AGUA} 25%, {CELESTE} 50%,
                                    {AZUL} 75%, {LAVANDA} 100%);
        border-radius: 18px;
        padding: 26px 34px;
        margin-bottom: 6px;
    }}
    .hero h1 {{ color: {TINTA}; margin: 0; font-size: 1.9rem; }}
    .hero p  {{ color: {TINTA}; margin: 6px 0 0 0; opacity: .85; }}

    /* Tarjetas (contenedores con borde) */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #FFFFFF;
        border: 1px solid {CELESTE};
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(143, 169, 220, .12);
    }}

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: #FFFFFF;
        border: 1px solid {CELESTE};
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        color: {TINTA};
    }}
    .stTabs [aria-selected="true"] {{
        background: {CELESTE};
        font-weight: 600;
    }}

    /* Botón primario */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, {AZUL}, {LAVANDA});
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(90deg, {LAVANDA}, {MORADO});
        color: #FFFFFF;
    }}

    /* Barras de progreso (score) */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {AGUA}, {AZUL});
    }}

    /* Chips de categoría / género */
    .chip {{
        display: inline-block;
        padding: 2px 12px;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 600;
        color: {TINTA};
    }}
    .chip-verde   {{ background: {VERDE}; }}
    .chip-agua    {{ background: {AGUA}; }}
    .chip-celeste {{ background: {CELESTE}; }}
    .chip-lavanda {{ background: {LAVANDA}; }}
    </style>

    <div class="hero">
      <h1>Búsqueda Multimodal</h1>
      <p>Imagen, letra y audio — índice invertido propio vs. Postgres, lado a lado.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def score_fraction(score):
    """Normaliza un score a [0, 1] para la barra de progreso."""
    if not isinstance(score, (int, float)):
        return 0.0
    return min(max(float(score), 0.0), 1.0)


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


def cargar_audio_resultado(uri):
    """Devuelve algo reproducible por st.audio a partir del uri del resultado.

    - URL http(s): se devuelve tal cual (Streamlit la reproduce en streaming).
    - Ruta local (absoluta o relativa a la raíz del proyecto): se leen los bytes.
    Devuelve None si el uri no existe o no es un archivo de audio.
    """
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        return uri
    candidatos = [uri]
    if not os.path.isabs(uri):
        # frontend/app.py -> raíz del proyecto (por si el cwd no es la raíz)
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidatos.append(os.path.join(raiz, uri))
    for ruta in candidatos:
        if os.path.isfile(ruta):
            try:
                with open(ruta, "rb") as fh:
                    return fh.read()
            except OSError:
                return None
    return None


# ── Renderizado de resultados ────────────────────────────────────────────────

def display_visual_results(results):
    if not results:
        st.info("No se encontraron resultados similares.")
        return

    st.success(f"Se encontraron {len(results)} resultados.")
    cols = st.columns(4)
    for idx, result in enumerate(results):
        with cols[idx % 4]:
            with st.container(border=True):
                st.markdown(f"**#{idx + 1} · {result.get('product_name', 'Desconocido')}**")
                categoria = result.get("category")
                if categoria:
                    st.markdown(
                        f'<span class="chip chip-celeste">{categoria}</span>',
                        unsafe_allow_html=True,
                    )
                image_ref = load_image_from_url_or_path(result.get("image_url"))
                if image_ref:
                    st.image(image_ref, use_container_width=True)
                else:
                    st.info("Imagen no disponible")
                score = result.get("score")
                st.progress(score_fraction(score))
                st.caption(
                    f"Score: {score:.4f}" if isinstance(score, (int, float)) else "Score: N/A"
                )


def display_music_results(results, playable_audio=False):
    if not results:
        st.info("No se encontraron resultados para esa búsqueda.")
        return

    st.success(f"Se encontraron {len(results)} canciones.")
    for idx, result in enumerate(results, start=1):
        with st.container(border=True):
            head, score_col = st.columns([4, 1])
            with head:
                song = result.get("song") or "Desconocida"
                artist = result.get("artist") or "Desconocido"
                genre = result.get("genre")
                st.markdown(f"**#{idx} · {song}**")
                if genre:
                    st.markdown(
                        f'<span class="chip chip-lavanda">{genre}</span>'
                        f'&nbsp;<span style="color:{TINTA};opacity:.7;">{artist}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(artist)
            with score_col:
                score = result.get("score")
                st.metric(
                    "Score",
                    f"{score:.4f}" if isinstance(score, (int, float)) else "N/A",
                )
            st.progress(score_fraction(result.get("score")))
            if playable_audio:
                audio = cargar_audio_resultado(result.get("uri"))
                if audio is not None:
                    st.audio(audio)
                else:
                    st.caption("Audio no disponible para reproducir.")
            lyrics = result.get("lyrics")
            if lyrics:
                with st.expander("Ver letra"):
                    st.text(lyrics)
                    uri = result.get("uri")
                    if uri and uri.startswith(("http://", "https://")):
                        st.markdown(f"[Video de la musica]({uri})")


# ── Pestañas ─────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Buscador de Ropa", "Buscador Musical"])

with tab1:
    st.subheader("Buscador visual e-commerce")
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
            "Motor de búsqueda (visual):",
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
            "Buscar producto", type="primary", use_container_width=True
        )

    if search_button:
        if uploaded_file is None:
            st.warning("Por favor, sube una imagen antes de buscar.")
        else:
            st.divider()
            with st.spinner("Analizando imagen y buscando productos similares..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {
                    "side": normalize_side(side_visual, VISUAL_SIDE_OPTIONS),
                    "k": k_visual,
                }
                response = call_api("POST", "/visual/search", params=params, files=files)
                if response is None:
                    st.stop()

                if response.status_code == 200:
                    display_visual_results(response.json().get("results", []))
                else:
                    st.error(f"Error en la API ({response.status_code}): {response.text}")

with tab2:
    st.subheader("Buscador musical")
    st.write("Busca canciones por letra o identifica una canción con un fragmento de audio.")

    search_type = st.radio(
        "Tipo de búsqueda:", ["Letra (texto)", "Canción (audio)"], horizontal=True
    )

    query_text = ""
    audio_file = None

    if search_type == "Letra (texto)":
        query_text = st.text_input(
            "Ingresa un extracto de la letra de la canción:",
            placeholder="Ej. Is this the real life? Is this just fantasy?",
        )
    else:
        # La canción se envía completa y sin modificar, igual que el probe en
        # consola — así el ranking del frontend es idéntico al de python.
        audio_file = st.file_uploader(
            "Sube una canción (.wav, .mp3, .ogg)",
            type=["wav", "mp3", "ogg"],
        )
        if audio_file is not None:
            st.audio(audio_file.getvalue())

    col1, col2 = st.columns(2)
    with col1:
        side_music = st.radio(
            "Motor de búsqueda (música):",
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
        "Buscar canción", type="primary", use_container_width=True
    )

    if search_music_btn:
        if search_type == "Letra (texto)" and not query_text.strip():
            st.warning("Por favor, ingresa al menos una palabra para buscar.")
        elif search_type == "Canción (audio)" and audio_file is None:
            st.warning("Sube una canción antes de buscar.")
        else:
            st.divider()
            with st.spinner("Buscando coincidencias en la base de datos..."):
                params = {
                    "side": normalize_side(side_music, MUSIC_SIDE_OPTIONS),
                    "k": k_music,
                }
                if search_type == "Letra (texto)":
                    params["q"] = query_text
                    response = call_api("GET", "/music/search", params=params)
                else:
                    # Garantizado por la validación del elif de arriba.
                    assert audio_file is not None
                    files = {"file": (audio_file.name, audio_file.getvalue(),
                                      audio_file.type or "audio/mpeg")}
                    response = call_api("POST", "/music/search_audio", params=params, files=files)

                if response is None:
                    st.stop()

                if response.status_code == 200:
                    display_music_results(
                        response.json().get("results", []),
                        playable_audio=(search_type == "Canción (audio)"),
                    )
                else:
                    st.error(f"Error en la API ({response.status_code}): {response.text}")
