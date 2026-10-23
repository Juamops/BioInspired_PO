# bioinspired-po

Réplica con pymoo del experimento de selección de portafolios de Hidalgo-Marín, Nebro y García-Nieto, *Algorithms* 2026, 19, 20, [doi:10.3390/a19010020](https://doi.org/10.3390/a19010020), más una mejora híbrida (activos por algoritmo evolutivo, pesos por programación cuadrática) y una prueba fuera de muestra en 2024.

El comando `bioinspired-po` solo imprime `Hello from bioinspired-po!`. El experimento está en los notebooks.

## Póster

Póster A0 vertical, en español, de la réplica y la mejora híbrida: [reports/poster/poster.pdf](reports/poster/poster.pdf). Fuente LaTeX: [reports/poster/poster.tex](reports/poster/poster.tex).

Recompilarlo es independiente de los experimentos. Desde `reports/poster`:

```bash
tectonic poster.tex
```

o `pdflatex poster.tex`.

## Cómo ejecutarlo

Python >= 3.13 (`.python-version` fija `3.13`). Este repo ya tiene un `.venv` con Python 3.13. En un clon limpio, desde la raíz:

```bash
uv sync
```

El directorio de trabajo es la raíz del repositorio. Cada notebook hace `ROOT = Path.cwd()` y, si esa carpeta se llama `notebooks`, sube al padre. Desde la raíz (o desde `notebooks/`) escribe en `data/raw`, `data/processed` y `reports/figures`.

Abrir los notebooks con el kernel de ese entorno:

```bash
source .venv/bin/activate
jupyter lab
```

Orden obligatorio:

1. `notebooks/pymoo_rep.ipynb` — descarga los datos y corre la réplica.
2. `notebooks/pymoo_hrp_hybrid.ipynb` — reutiliza esas descargas y esas corridas, y corre las variantes HRP e híbrida. Exige las mismas `INSTANCES`, `N_RUNS` y `EVAL_SCALE` que la réplica.
3. `notebooks/pymoo_oos_2024.ipynb` — lee los frentes guardados y baja los precios de 2023–2024 de Yahoo Finance.

Los tres usan las mismas 12 instancias: `NYSE_5_100`, `NASDAQ_5_100`, `OTC_5_100`, `AMEX_5_100`, `ARCA_5_100`, `BATS_5_100`, `ALL_5_100`, `ALL_20_100`, `ALL_20_500`, `ALL_20_1000`, `ALL_20_3000`, `ALL_20_6000`. En la réplica y en el híbrido, `N_RUNS = 10`, `EVAL_SCALE = 1.0` y `N_JOBS = -1`. El notebook de 2024 también usa `N_RUNS = 10`; su presupuesto de evaluaciones es `BASE_EVALS` por el multiplicador de la instancia, sin `EVAL_SCALE`.

## Pasada ligera / RAM

`N_JOBS = -1` usa todos los núcleos. `ALL_20_3000` y `ALL_20_6000` piden unos 320 MB de descarga, y sus covarianzas en CSV ocupan 0,2 GB y 0,8 GB.

Para una pasada ligera, en `pymoo_rep.ipynb` y `pymoo_hrp_hybrid.ipynb`: bajar `EVAL_SCALE` (el notebook sugiere 0,1), quitar esas dos instancias de `INSTANCES` y fijar `N_JOBS` en 1 o 2. Las corridas completas ya están resumidas en las figuras y en el póster; leer los resultados no exige repetirlas. `pymoo_oos_2024.ipynb` busca los CSV del presupuesto completo, así que una pasada ligera no le sirve de entrada.

## Dónde quedan datos y figuras

- `data/raw`: retornos, matrices de covarianza y frentes de referencia de cada instancia, y `yahoo_close_2023_2024.csv`.
- `data/processed`: corridas cacheadas (`pymoo_<instancia>_<algoritmo>_<evals>evals.csv`, y los prefijos `pymoo-hrp_` y `pymoo-hybrid_`).
- `reports/figures`: figuras de los notebooks, incluidas [poster_sharpe_k20.png](reports/figures/poster_sharpe_k20.png), [poster_ranks.png](reports/figures/poster_ranks.png) y [poster_oos_sharpe.png](reports/figures/poster_oos_sharpe.png).
