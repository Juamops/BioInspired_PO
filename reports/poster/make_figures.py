"""Figures and numbers for the poster (reports/poster/poster_en.tex and poster_es.tex).

Reads the saved runs of notebooks/pymoo_hrp_hybrid.ipynb ("this work"), the paper's published reference fronts and,
for the out-of-sample panel, the paper's ALL_20_6000 reference-front solutions and Yahoo Finance prices (cached in
data/raw/yahoo_close_2023_2024.csv, shared with notebooks/pymoo_oos_2024.ipynb). Missing inputs are downloaded.
Run from anywhere: `uv run python reports/poster/make_figures.py`.
"""

import io
import logging
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT = Path(__file__).resolve().parents[2]
RAW, PROCESSED, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INSTANCES_20 = ["ALL_20_100", "ALL_20_500", "ALL_20_1000", "ALL_20_3000", "ALL_20_6000"]
INST = "ALL_20_6000"
ALGORITHMS = ["MOEA/D", "SMS-EMOA", "NSGA-II"]
EVAL_MULTIPLIER = {"ALL_20_500": 2, "ALL_20_1000": 3, "ALL_20_3000": 4, "ALL_20_6000": 5}
N_RUNS = 10
K = 20
RISK_FREE = 0.0697
TRADING_DAYS = 250
MIN_COVERAGE = 0.80  # out of sample: keep portfolios with at least this share of their weight tradeable
MM = 1 / 25.4
ZIP_URL = ("https://github.com/AntHidMar/Portfolio_Study_DataSources/"
           "releases/download/financial_data/Portfolio_Study.zip")

# Poster palette: the paper is the neutral baseline, this work the one deep-blue accent.
INK, MUTED, GRID = "#1A1D24", "#5A6170", "#E3E6EB"
BLUE, PAPER = "#1A4D9C", "#BCC1C9"
STYLE = {
    "paper": dict(color=PAPER, lw=13, solid_capstyle="round", zorder=1),
    "hybrid": dict(color=BLUE, lw=4.5, solid_capstyle="round", zorder=3),
}
VARIANTS = list(STYLE)

TEXT = {
    "en": {
        "paper": "Paper [1]", "hybrid": "This work", "risk": "Risk (%)", "return": "Return (%)",
        "a": "(a) Reference fronts", "zoom": "zoomed in (b)", "b": "(b) Low-risk end, best Sharpe",
        "c": "(c) Sharpe along the front", "c_y": "Sharpe ratio", "c_x": "Risk (%), log scale",
        "sharpe": "Best Sharpe ratio", "lowest": "Lowest risk (%)",
        "size": "Universe size N (k = 20)", "oos_x": "Sharpe ratio, 2023", "oos_y": "Sharpe ratio, 2024",
        "no_change": "no change", "median": "medians",
    },
    "es": {
        "paper": "Artículo [1]", "hybrid": "Este trabajo", "risk": "Riesgo (%)", "return": "Rendimiento (%)",
        "a": "(a) Frentes de referencia", "zoom": "ampliado en (b)", "b": "(b) Riesgo bajo y mejor Sharpe",
        "c": "(c) Sharpe a lo largo del frente", "c_y": "Ratio de Sharpe",
        "c_x": "Riesgo (%), escala log.", "sharpe": "Mejor ratio de Sharpe",
        "lowest": "Riesgo mínimo (%)", "size": "Universo N (k = 20)", "oos_x": "Ratio de Sharpe, 2023",
        "oos_y": "Ratio de Sharpe, 2024", "no_change": "sin cambio", "median": "medianas",
    },
}

for f in font_manager.findSystemFonts():
    if Path(f).name.startswith("Inter_FXH"):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Inter_FXH", "font.size": 24, "axes.labelsize": 26, "axes.titlesize": 29,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 18,
    "xtick.labelsize": 23, "ytick.labelsize": 23, "legend.fontsize": 25,
    "axes.edgecolor": "#8A909B", "axes.linewidth": 1.4, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 7, "ytick.major.size": 7, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.axisbelow": True,
})
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# --- data ---------------------------------------------------------------------------------------
def cache_path(instance, algorithm):
    slug = algorithm.replace("/", "").replace("-", "")
    return PROCESSED / f"pymoo-hybrid_{instance}_{slug}_{50_000 * EVAL_MULTIPLIER.get(instance, 1)}evals.csv"


def nondominated(df, cols=("risk", "neg_return")):
    keep = NonDominatedSorting().do(df[list(cols)].to_numpy(), only_non_dominated_front=True)
    return df.iloc[keep].sort_values(cols[0])


def hybrid_front(instance):
    """This work's reference front: the non-dominated portfolios of all runs of all three algorithms."""
    df = pd.concat([pd.read_csv(cache_path(instance, alg)) for alg in ALGORITHMS], ignore_index=True)
    return nondominated(df[df["run"] < N_RUNS])


def paper_front(instance):
    F = np.loadtxt(RAW / f"{instance}_paper_reference_front.csv", delimiter=",")
    return nondominated(pd.DataFrame(F, columns=["risk", "neg_return"]))


def fetch_paper_solutions(instance):
    """Decision vectors behind the paper's reference front, from the authors' release zip (range requests)."""
    dest = RAW / f"{instance}_paper_reference_solutions.csv"
    if not dest.exists():
        with zipfile.ZipFile(RemoteFile(ZIP_URL)) as z, z.open(f"Portfolio_Study/referenceFronts/{instance}.ps") as src, \
                open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return np.loadtxt(dest, delimiter=",")


class RemoteFile(io.RawIOBase):
    """Seekable, read-only view of a remote file that downloads only the byte ranges read."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data


def decode_pointers(S, n_assets):
    """The paper's decoding: asset floor(s * N), linear probing on collisions."""
    n, k = S.shape
    rows = np.arange(n)
    taken = np.zeros((n, n_assets), dtype=bool)
    assets = np.empty((n, k), dtype=int)
    for j in range(k):
        c = np.minimum((S[:, j] * n_assets).astype(int), n_assets - 1)
        clash = taken[rows, c]
        while clash.any():
            c[clash] = (c[clash] + 1) % n_assets
            clash = taken[rows, c]
        taken[rows, c] = True
        assets[:, j] = c
    return assets


def bound_weights(W, w_min, w_max):
    """The paper's weight repair: normalise to sum 1 with every weight in [w_min, w_max]."""
    W = W / W.sum(axis=1, keepdims=True)
    breaks = np.sort(np.concatenate([w_min / W, w_max / W], axis=1), axis=1)
    total = np.clip(breaks[:, :, None] * W[:, None, :], w_min, w_max).sum(axis=2)
    j = np.clip((total < 1).sum(axis=1), 1, breaks.shape[1] - 1)
    r = np.arange(len(W))
    l0, l1, t0, t1 = breaks[r, j - 1], breaks[r, j], total[r, j - 1], total[r, j]
    lam = l0 + (1 - t0) * (l1 - l0) / np.maximum(t1 - t0, 1e-300)
    return np.clip(lam[:, None] * W, w_min, w_max)


def best_return_within(F, risk):
    """Highest return on front F among portfolios with risk <= `risk` (NaN below the front's lowest risk)."""
    order = np.argsort(F[:, 0])
    cummax = np.maximum.accumulate(-F[order, 1])
    i = np.searchsorted(F[order, 0], risk, side="right") - 1
    return np.where(i >= 0, cummax[np.clip(i, 0, None)], np.nan)


def front_return_at(F, risk):
    """Return of front F at `risk`, interpolating linearly between neighbouring portfolios (as the plotted line)."""
    return np.interp(risk, F[:, 0], -F[:, 1], left=np.nan, right=np.nan)


frames = {("paper", inst): paper_front(inst) for inst in INSTANCES_20}
frames.update({("hybrid", inst): hybrid_front(inst) for inst in INSTANCES_20})
fronts = {key: df[["risk", "neg_return"]].to_numpy() for key, df in frames.items()}

# --- in-sample numbers ----------------------------------------------------------------------------
rows = []
for inst in INSTANCES_20:
    union = np.vstack([fronts[v, inst] for v in VARIANTS])
    union = union[NonDominatedSorting().do(union, only_non_dominated_front=True)]
    ideal, nadir = union.min(axis=0), union.max(axis=0)
    igd, hv = IGDPlus((union - ideal) / (nadir - ideal)), HV(ref_point=np.full(2, 1.1))
    for v in VARIANTS:
        F = fronts[v, inst]
        sharpe = (-F[:, 1] - RISK_FREE) / F[:, 0]
        best = sharpe.argmax()
        rows.append({"instance": inst, "variant": v, "HV": hv((F - ideal) / (nadir - ideal)),
                     "IGD+": igd((F - ideal) / (nadir - ideal)), "best Sharpe": sharpe[best],
                     "its risk": F[best, 0], "its return": -F[best, 1], "lowest risk": F[:, 0].min(),
                     "highest return": -F[:, 1].min(), "return at 5% risk": float(front_return_at(F, 0.05))})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "poster_numbers.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(summary.round(4).to_string(index=False))

# --- out of sample (2024), ALL_20_6000 reference fronts -------------------------------------------
mu = pd.read_csv(RAW / f"{INST}_mean_return.csv", index_col=0).iloc[:, 0]
universe = np.array([s.rsplit("\\", 1)[-1].removeprefix("Day_") for s in mu.index])
X = fetch_paper_solutions(INST)
paper_assets = decode_pointers(X[:, :K], len(universe))
paper_weights = bound_weights(X[:, K:], 1 / (2 * K), 2 / K)  # 656 of 658 are stored repaired already
published = np.loadtxt(RAW / f"{INST}_paper_reference_front.csv", delimiter=",")  # same row order as X
assert np.abs((paper_weights * mu.to_numpy()[paper_assets]).sum(axis=1) + published[:, 1]).max() < 1e-3, \
    "the paper's solutions do not decode to its published front"
holdings = {
    "paper": (universe[paper_assets], paper_weights),
    "hybrid": (universe[frames["hybrid", INST].filter(regex=r"^a\d+$").to_numpy().astype(int)],
               frames["hybrid", INST].filter(regex=r"^w\d+$").to_numpy()),
}


def to_yahoo(ticker):
    """Interactive Brokers symbol -> Yahoo symbol (ANG.PRB -> ANG-PB, CHEB.WS -> CHEB-WT, AGM.A -> AGM-A)."""
    t = re.sub(r"\.PR([A-Z]?)$", lambda m: "-P" + m.group(1), ticker)
    return t.replace(".WS", "-WT").replace(".U", "-UN").replace(".RT", "-RI").replace(".", "-")


def complete_year(series, year):
    s = series.loc[str(year)].dropna()
    return (len(s) >= 240 and s.index[0] <= pd.Timestamp(f"{year}-01-06")
            and s.index[-1] >= pd.Timestamp(f"{year}-12-20"))


def year_statistics(prices, year):
    """Total log return and daily covariance per asset, computed exactly as the authors' 2023 files."""
    px = prices.loc[f"{year}-01-01":f"{year}-12-31"].resample("D").last().ffill()
    r = np.log1p(px.pct_change(fill_method=None)).replace([np.inf, -np.inf], np.nan)
    return r.sum().to_numpy(), r.ffill().bfill().cov().to_numpy()


PRICES = RAW / "yahoo_close_2023_2024.csv"
needed = sorted({t for assets, _ in holdings.values() for t in np.unique(assets)})
close = pd.read_csv(PRICES, index_col=0, parse_dates=True)
missing = sorted({to_yahoo(t) for t in needed} - set(close.columns))
if missing:
    df = yf.download(missing, start="2023-01-01", end="2025-01-01", auto_adjust=False, progress=False, threads=True)
    close = pd.concat([close, df["Close"].reindex(columns=missing)], axis=1).sort_index()
    close.to_csv(PRICES)
    print(f"downloaded {len(missing)} tickers from Yahoo Finance")
tradeable = [t for t in needed if to_yahoo(t) in close and complete_year(close[to_yahoo(t)], 2023)
             and complete_year(close[to_yahoo(t)], 2024)]
panel = close[[to_yahoo(t) for t in tradeable]]
year_stats = {year: year_statistics(panel, year) for year in (2023, 2024)}
position = {t: i for i, t in enumerate(tradeable)}

oos = []
for v, (assets, W) in holdings.items():
    idx = np.vectorize(lambda t: position.get(t, -1))(assets)
    ok = idx >= 0
    coverage = np.sum(W * ok, axis=1)
    Wn = np.where(ok, W, 0.0) / np.maximum(coverage, 1e-12)[:, None]
    idx = np.where(ok, idx, 0)
    part = pd.DataFrame({"variant": v, "coverage": coverage})
    for year in (2023, 2024):
        mu_y, cov_y = year_stats[year]
        ret = np.sum(Wn * mu_y[idx], axis=1)
        risk = np.sqrt(TRADING_DAYS * np.einsum("ni,nij,nj->n", Wn, cov_y[idx[:, :, None], idx[:, None, :]], Wn))
        part[f"Sharpe {year}"] = (ret - RISK_FREE) / risk
    oos.append(part)
oos = pd.concat(oos, ignore_index=True)
tested = oos[oos["coverage"] >= MIN_COVERAGE]
both_negative = (tested["Sharpe 2023"] < 0) & (tested["Sharpe 2024"] < 0)
print("left out, negative Sharpe in both years:", tested[both_negative].groupby("variant").size().to_dict())
tested = tested[~both_negative]
oos_summary = tested.groupby("variant").agg(
    tested=("coverage", "size"), median_2023=("Sharpe 2023", "median"), median_2024=("Sharpe 2024", "median"),
    share_fell=("Sharpe 2024", lambda s: np.mean(s < tested.loc[s.index, "Sharpe 2023"])))
oos_summary["front"] = oos.groupby("variant").size()
oos_summary["spearman"] = tested.groupby("variant")[["Sharpe 2023", "Sharpe 2024"]].apply(
    lambda g: g["Sharpe 2023"].rank().corr(g["Sharpe 2024"].rank()))
oos_summary.to_csv(OUT / "oos_numbers.csv")
print(f"{len(tradeable)} of {len(needed)} assets tradeable in 2023 and 2024")
print(oos_summary.round(3).to_string())


# --- figures --------------------------------------------------------------------------------------
def figure_fronts(t):
    fig, axes = plt.subplots(1, 3, figsize=(600 * MM, 246 * MM), gridspec_kw=dict(width_ratios=[1.08, 1, 1]),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.25, wspace=0.05)
    ax_full, ax_zoom, ax_sharpe = axes
    zoom = (0.15, 1.30)

    for v in VARIANTS:
        F = fronts[v, INST] * [100, -100]
        ax_full.plot(F[:, 0], F[:, 1], label=t[v], **STYLE[v])
        ax_zoom.plot(F[:, 0], F[:, 1], **STYLE[v])
    ax_full.add_patch(Rectangle((0, 0), zoom[0] * 100, zoom[1] * 100, facecolor=BLUE, alpha=0.06,
                                edgecolor=BLUE, lw=1.6, ls=(0, (4, 3)), zorder=0))
    ax_full.text(zoom[0] * 100 + 1.5, 4, t["zoom"], color=MUTED, fontsize=22, va="bottom")
    ax_full.set(xlim=(0, 70), ylim=(0, 230), xlabel=t["risk"], ylabel=t["return"], title=t["a"])

    # (b) low-risk end, each front's best-Sharpe portfolio and its tangent line from the risk-free rate
    x = np.linspace(0, zoom[0], 50)
    label_offset = {"paper": (16, -24), "hybrid": (-16, 22)}
    for v in VARIANTS:
        r = summary[(summary["instance"] == INST) & (summary["variant"] == v)].iloc[0]
        color = BLUE if v == "hybrid" else "#8C939E"
        ax_zoom.plot(100 * x, 100 * (RISK_FREE + r["best Sharpe"] * x), color=color, lw=2.2, zorder=1.5)
        ax_zoom.scatter(100 * r["its risk"], 100 * r["its return"], s=340, color=color, edgecolor="white",
                        linewidth=3, zorder=5)
        ax_zoom.annotate(f"SR {r['best Sharpe']:.2f}", xy=(100 * r["its risk"], 100 * r["its return"]),
                         xytext=label_offset[v], textcoords="offset points", va="center", fontsize=24,
                         ha="right" if v == "hybrid" else "left", color=INK, zorder=6,
                         fontweight="bold" if v == "hybrid" else "normal")
    ax_zoom.scatter([0], [100 * RISK_FREE], s=140, color=MUTED, zorder=5, clip_on=False)  # R_f, named in the caption
    ax_zoom.set(xlim=(0, zoom[0] * 100), ylim=(0, zoom[1] * 100), xlabel=t["risk"], ylabel=t["return"],
                xticks=[0, 5, 10, 15], title=t["b"])

    # (c) Sharpe ratio of every portfolio along each front, with each front's best one labelled
    for v in VARIANTS:
        F = fronts[v, INST]
        ax_sharpe.plot(100 * F[:, 0], (-F[:, 1] - RISK_FREE) / F[:, 0],
                       **{**STYLE[v], "lw": 9 if v == "paper" else STYLE[v]["lw"]})
        r = summary[(summary["instance"] == INST) & (summary["variant"] == v)].iloc[0]
        ax_sharpe.annotate(f"{r['best Sharpe']:.2f}", xy=(100 * r["its risk"], r["best Sharpe"]),
                           xytext=(0, 16) if v == "hybrid" else (0, -22), textcoords="offset points", ha="center",
                           va="bottom" if v == "hybrid" else "top", fontsize=23,
                           fontweight="bold" if v == "hybrid" else "normal", color=INK if v == "hybrid" else MUTED)
    ax_sharpe.set_xscale("log")
    ax_sharpe.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax_sharpe.minorticks_off()
    ax_sharpe.set(xlim=(0.8, 70), ylim=(-2, 10.5), xlabel=t["c_x"], ylabel=t["c_y"], title=t["c"])

    fig.legend(*ax_full.get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False,
               handlelength=2.6, columnspacing=4)
    return fig


def figure_scaling(t):
    sizes = [int(i.rsplit("_", 1)[1]) for i in INSTANCES_20]
    fig, axes = plt.subplots(1, 2, figsize=(250 * MM, 150 * MM), layout="constrained")
    fig.get_layout_engine().set(wspace=0.1)
    for ax, metric, title, scale in [(axes[0], "best Sharpe", t["sharpe"], 1), (axes[1], "lowest risk", t["lowest"], 100)]:
        for v in VARIANTS:
            y = scale * summary[summary["variant"] == v].set_index("instance").loc[INSTANCES_20, metric]
            style = {**STYLE[v], "lw": 9 if v == "paper" else STYLE[v]["lw"]}
            ax.plot(sizes, y, **style)
            if v == "hybrid":
                ax.scatter(sizes, y, s=120, color=BLUE, edgecolor="white", linewidth=2, zorder=4)
            ax.annotate(f"{y.iloc[-1]:.2f}", xy=(sizes[-1], y.iloc[-1]), xytext=(14, 0), textcoords="offset points",
                        va="center", fontsize=22, fontweight="bold" if v == "hybrid" else "normal",
                        color=INK if v == "hybrid" else MUTED)
        ax.set_xscale("log")
        ax.set_xticks(sizes, ["100", "500", "1k", "3k", "6k"])
        ax.minorticks_off()
        ax.set(xlim=(75, 14_000), xlabel=t["size"], title=title)
        ax.title.set_fontsize(25)
    axes[0].set_ylim(3.5, 9.5)
    axes[1].set_ylim(0, 4.8)
    return fig


def figure_oos(t):
    fig, ax = plt.subplots(figsize=(250 * MM, 114 * MM), layout="constrained")
    lim_x, lim_y = (min(-0.3, tested["Sharpe 2023"].min() - 0.3), 8), (-4, 4.5)
    ax.plot([lim_x[0], lim_y[1]], [lim_x[0], lim_y[1]], color="#8C939E", lw=1.8, zorder=1)  # 2024 = 2023
    ax.text(lim_y[1] + 0.15, lim_y[1] - 0.1, t["no_change"], ha="left", va="top", fontsize=21, color=MUTED)
    ax.axhline(0, color="#8A909B", lw=1.4, zorder=1)
    for v, style in [("paper", dict(s=70, color="#9AA1AC", alpha=0.75, lw=0)),
                     ("hybrid", dict(s=80, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.8))]:
        d = tested[tested["variant"] == v]
        ax.scatter(d["Sharpe 2023"], d["Sharpe 2024"], label=t[v], zorder=3 if v == "hybrid" else 2, **style)
        m = oos_summary.loc[v]
        ax.scatter(m["median_2023"], m["median_2024"], s=420, facecolor="none",
                   edgecolor=BLUE if v == "hybrid" else "#5A6170", linewidth=3.2, zorder=4)
    ax.annotate(t["median"], xy=(oos_summary.loc["hybrid", "median_2023"], oos_summary.loc["hybrid", "median_2024"]),
                xytext=(40, -62), textcoords="offset points", fontsize=21, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.4, shrinkB=12))
    ax.set(xlim=lim_x, ylim=lim_y, xlabel=t["oos_x"], ylabel=t["oos_y"], yticks=[-4, -2, 0, 2, 4])
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False,
               handletextpad=0.3, columnspacing=3, markerscale=1.6)
    return fig


for lang, t in TEXT.items():
    for name, make in [("fig_all20_6000", figure_fronts), ("fig_scaling", figure_scaling), ("fig_oos", figure_oos)]:
        fig = make(t)
        fig.savefig(OUT / f"{name}_{lang}.pdf")
        plt.close(fig)
print("figures written to", OUT)
