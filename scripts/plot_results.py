"""Gráficos comparativos de los benchmarks: índice invertido vs GIN / pgvector.

Lee los CSVs generados por scripts.evaluate_performance en results/ y produce
PNGs en results/plots/ para el análisis de trade-offs:

  latencia_<mod>.png          latencia promedio vs escala (Lado A vs Lado B)
  throughput_<mod>.png        QPS promedio vs escala
  precision_recall_<mod>.png  precision@k y recall@k por sistema y escala
  memoria_<mod>.png           RAM delta y pico de memoria por sistema y escala
  io_<mod>.png                bloques shared_hit / shared_read del Lado B

Si hay varios CSV para la misma (escala, modalidad) usa el más reciente.

Ejecución:
  python -m scripts.plot_results [--input results] [--output results/plots]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana: solo PNGs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

ORDEN_ESCALAS = ["1k", "10k", "100k"]
PATRON = re.compile(r"results_(1k|10k|100k)_(text|image|audio)_(\d{8}_\d{6})\.csv")

# Nombre legible del Lado B según la modalidad
LADO_B = {"text": "GIN (Postgres)", "image": "pgvector", "audio": "pgvector"}
LADO_A = "Índice invertido"

COLOR_A, COLOR_B = "#1f77b4", "#d62728"


def cargar(input_dir: Path) -> pd.DataFrame:
    """Concatena el CSV más reciente de cada (escala, modalidad).

    Busca recursivamente: los CSVs viven en results/<escala>/.
    """
    ultimos: dict[tuple[str, str], Path] = {}
    for f in sorted(input_dir.rglob("results_*.csv")):
        m = PATRON.match(f.name)
        if m:
            ultimos[(m.group(1), m.group(2))] = f  # sorted => se queda el último ts
    if not ultimos:
        print(f"[ERROR] No hay CSVs results_*.csv en {input_dir}. "
              "Corre antes scripts.evaluate_performance o scripts.run_scale_tests.")
        sys.exit(1)
    frames = [pd.read_csv(f) for f in ultimos.values()]
    df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] {len(ultimos)} CSVs cargados, {len(df)} filas: "
          f"{sorted(ultimos.keys())}")
    return df


def _promedios(df: pd.DataFrame, columna: str) -> pd.Series:
    """Promedio por escala (ignora vacíos y centinelas -1), ordenado 1k→100k."""
    d = df.copy()
    d[columna] = pd.to_numeric(d[columna], errors="coerce")
    d = d[d[columna].notna() & (d[columna] != -1)]
    medias = d.groupby("scale")[columna].mean()
    return medias.reindex([e for e in ORDEN_ESCALAS if e in medias.index])


def _guardar(fig, out_dir: Path, nombre: str) -> None:
    ruta = out_dir / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  [OK] {ruta}")


def graficar_lineas(df, mod, out_dir, col_a, col_b, titulo, ylabel, nombre,
                    log_y=False):
    serie_a, serie_b = _promedios(df, col_a), _promedios(df, col_b)
    if serie_a.empty and serie_b.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not serie_a.empty:
        ax.plot(serie_a.index, serie_a.values, "o-", color=COLOR_A, label=LADO_A)
    if not serie_b.empty:
        ax.plot(serie_b.index, serie_b.values, "s--", color=COLOR_B, label=LADO_B[mod])
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Escala (chunks)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{titulo} — {mod}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _guardar(fig, out_dir, nombre)


def graficar_precision_recall(df, mod, out_dir):
    metricas = [("lado_a_precision", f"Precision {LADO_A}", COLOR_A, 1.0),
                ("lado_b_precision", f"Precision {LADO_B[mod]}", COLOR_B, 1.0),
                ("lado_a_recall",    f"Recall {LADO_A}",    COLOR_A, 0.45),
                ("lado_b_recall",    f"Recall {LADO_B[mod]}", COLOR_B, 0.45)]
    series = {t: _promedios(df, c) for c, t, _, _ in metricas}
    if all(s.empty for s in series.values()):
        print(f"  [AVISO] {mod}: sin datos de precision/recall (¿sources sin etiquetas?)")
        return
    escalas = [e for e in ORDEN_ESCALAS if any(e in s.index for s in series.values())]
    x = np.arange(len(escalas))
    ancho = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (col, titulo, color, alpha) in enumerate(metricas):
        vals = [series[titulo].get(e, np.nan) for e in escalas]
        ax.bar(x + (i - 1.5) * ancho, vals, ancho, label=titulo,
               color=color, alpha=alpha)
    ax.set_xticks(x, escalas)
    ax.set_xlabel("Escala (chunks)")
    ax.set_ylabel("Valor promedio")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision@k y Recall@k — {mod}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _guardar(fig, out_dir, f"precision_recall_{mod}.png")


def graficar_memoria(df, mod, out_dir):
    series = {f"Pico {LADO_A}":       _promedios(df, "lado_a_mem_pico_mb"),
              f"Pico {LADO_B[mod]}":  _promedios(df, "lado_b_mem_pico_mb"),
              f"Delta {LADO_A}":      _promedios(df, "lado_a_ram_delta_mb"),
              f"Delta {LADO_B[mod]}": _promedios(df, "lado_b_ram_delta_mb")}
    if all(s.empty for s in series.values()):
        return
    escalas = [e for e in ORDEN_ESCALAS if any(e in s.index for s in series.values())]
    x = np.arange(len(escalas))
    ancho = 0.2
    colores = [COLOR_A, COLOR_B, COLOR_A, COLOR_B]
    alphas = [1.0, 1.0, 0.45, 0.45]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (titulo, serie) in enumerate(series.items()):
        vals = [serie.get(e, np.nan) for e in escalas]
        ax.bar(x + (i - 1.5) * ancho, vals, ancho, label=titulo,
               color=colores[i], alpha=alphas[i])
    ax.set_xticks(x, escalas)
    ax.set_xlabel("Escala (chunks)")
    ax.set_ylabel("MB")
    ax.set_title(f"Memoria por consulta — {mod}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _guardar(fig, out_dir, f"memoria_{mod}.png")


def graficar_io(df, mod, out_dir):
    """I/O por consulta en MB: Lado A (carga índice + búsqueda) vs Lado B
    (bloques de 8 KB tocados en el buffer pool de Postgres)."""
    hit, read = _promedios(df, "lado_b_io_shared_hit"), _promedios(df, "lado_b_io_shared_read")
    a_load = _promedios(df, "lado_a_io_load_mb")
    a_search = _promedios(df, "lado_a_io_read_mb")
    if hit.empty and read.empty and a_load.empty:
        return
    escalas = [e for e in ORDEN_ESCALAS
               if e in hit.index or e in read.index or e in a_load.index]
    x = np.arange(len(escalas))
    ancho = 0.35
    BLOQUE_MB = 8 / 1024  # bloque Postgres de 8 KB en MB

    va_load = [a_load.get(e, 0) for e in escalas]
    va_search = [max(a_search.get(e, 0), 0) for e in escalas]
    vb_hit = [hit.get(e, 0) * BLOQUE_MB for e in escalas]
    vb_read = [read.get(e, 0) * BLOQUE_MB for e in escalas]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - ancho / 2, va_load, ancho, label=f"{LADO_A}: carga índice",
           color=COLOR_A, alpha=0.45)
    ax.bar(x - ancho / 2, va_search, ancho, bottom=va_load,
           label=f"{LADO_A}: búsqueda", color=COLOR_A)
    ax.bar(x + ancho / 2, vb_hit, ancho, label=f"{LADO_B[mod]}: shared_hit (cache)",
           color="#2ca02c")
    ax.bar(x + ancho / 2, vb_read, ancho, bottom=vb_hit,
           label=f"{LADO_B[mod]}: shared_read (disco)", color="#ff7f0e")
    ax.set_xticks(x, escalas)
    ax.set_xlabel("Escala (chunks)")
    ax.set_ylabel("MB por consulta")
    ax.set_title(f"Accesos I/O — {mod}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _guardar(fig, out_dir, f"io_{mod}.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Gráficos comparativos de los benchmarks")
    p.add_argument("--input", default="results")
    p.add_argument("--output", default="results/plots")
    args = p.parse_args()

    df = cargar(Path(args.input))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mod in df["modality"].dropna().unique():
        sub = df[df["modality"] == mod]
        print(f"\n[{mod.upper()}] escalas presentes: {sorted(sub['scale'].unique())}")
        graficar_lineas(sub, mod, out_dir, "lado_a_latencia_ms", "lado_b_latencia_ms",
                        "Latencia promedio", "ms", f"latencia_{mod}.png", log_y=True)
        graficar_lineas(sub, mod, out_dir, "lado_a_qps", "lado_b_qps",
                        "Throughput promedio", "consultas/segundo",
                        f"throughput_{mod}.png", log_y=True)
        graficar_precision_recall(sub, mod, out_dir)
        graficar_memoria(sub, mod, out_dir)
        graficar_io(sub, mod, out_dir)

    print(f"\n[OK] Gráficos en: {out_dir}")


if __name__ == "__main__":
    main()
