"""Frontend Streamlit — Búsqueda Multimodal (paleta de colores fríos pastel)."""
import io
import os
from urllib.parse import urlparse

import librosa
import requests
import soundfile as sf
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30
DEFAULT_K = 10
MAX_K = 50

# Fragmentos estilo Shazam: nunca se envía la canción completa al backend.
FRAG_MIN_S = 3
FRAG_MAX_S = 15
FRAG_DEFAULT_S = 8

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

st.set_page_config(page_title="Búsqueda Multimodal", page_icon="🌊", layout="wide")

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


@st.cache_data(show_spinner=False, max_entries=4)
def decodificar_audio(data: bytes):
    """Decodifica el archivo subido a PCM mono 22.05 kHz (cacheado por bytes)."""
    y, sr = librosa.load(io.BytesIO(data), sr=22050, mono=True)
    return y, sr


def fragmento_wav(y, sr, inicio_s: float, dur_s: float) -> bytes:
    """Recorta [inicio, inicio+dur] y lo serializa como WAV en memoria."""
    seg = y[int(inicio_s * sr): int((inicio_s + dur_s) * sr)]
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV")
    return buf.getvalue()


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


def display_music_results(results):
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
        "Tipo de búsqueda:", ["Letra (texto)", "Fragmento (audio)"], horizontal=True
    )

    query_text = ""
    fragmento_bytes = None

    if search_type == "Letra (texto)":
        query_text = st.text_input(
            "Ingresa un extracto de la letra de la canción:",
            placeholder="Ej. Is this the real life? Is this just fantasy?",
        )
    else:
        # ── Modo Shazam: solo se envía un fragmento corto de la canción ──
        st.markdown(
            f'<span class="chip chip-agua">Modo Shazam</span>&nbsp;'
            f'<span style="color:{TINTA};opacity:.75;">elige un fragmento de '
            f"{FRAG_MIN_S}–{FRAG_MAX_S} segundos; nunca se envía la canción completa</span>",
            unsafe_allow_html=True,
        )
        audio_file = st.file_uploader(
            "Sube una canción (.wav, .mp3, .ogg)",
            type=["wav", "mp3", "ogg"],
        )
        if audio_file is not None:
            try:
                y, sr = decodificar_audio(audio_file.getvalue())
            except Exception as exc:
                st.error(f"No se pudo decodificar el audio: {exc}")
                y, sr = None, None

            if y is not None:
                dur_total = len(y) / sr
                if dur_total < FRAG_MIN_S:
                    st.warning(
                        f"El audio dura {dur_total:.1f}s; se necesita al menos {FRAG_MIN_S}s."
                    )
                else:
                    with st.container(border=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            dur_frag = st.slider(
                                "Duración del fragmento (s):",
                                min_value=FRAG_MIN_S,
                                max_value=min(FRAG_MAX_S, int(dur_total)),
                                value=min(FRAG_DEFAULT_S, int(dur_total)),
                                key="frag_dur",
                            )
                        with c2:
                            max_inicio = max(dur_total - dur_frag, 0.0)
                            inicio = st.slider(
                                "Inicio del fragmento (s):",
                                min_value=0.0,
                                max_value=float(max(max_inicio, 0.1)),
                                value=min(float(max_inicio) / 2, float(max_inicio)),
                                step=0.5,
                                key="frag_inicio",
                            )
                        fragmento_bytes = fragmento_wav(y, sr, inicio, dur_frag)
                        st.caption(
                            f"Fragmento seleccionado: {inicio:.1f}s → {inicio + dur_frag:.1f}s "
                            f"(de {dur_total:.1f}s totales)"
                        )
                        st.audio(fragmento_bytes, format="audio/wav")

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
        elif search_type == "Fragmento (audio)" and fragmento_bytes is None:
            st.warning("Sube una canción y selecciona un fragmento antes de buscar.")
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
                    files = {"file": ("fragmento.wav", fragmento_bytes, "audio/wav")}
                    response = call_api("POST", "/music/search_audio", params=params, files=files)

                if response is None:
                    st.stop()

                if response.status_code == 200:
                    display_music_results(response.json().get("results", []))
                else:
                    st.error(f"Error en la API ({response.status_code}): {response.text}")
