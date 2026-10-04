"""Stellar mass versus sSFR, coloured by beta, in three redshift bins.

Run alongside utils/beta_utils.py on COSMA with CAESAR installed.
Uses the original observed M1500 cuts and three-band Calbeta estimator.
Pools both boxes and both snapshots in each panel (no volume weighting).
Saves PNG and PDF; no CSV. Nonpositive mass/SFR cannot appear on log axes.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

# User settings
DUST_LAW = "calzetti"
BETA_KIND = "attenuated"  # "attenuated" or "intrinsic"; selection stays observed
MASS_BIN_WIDTH = 0.3       # dex; identical bin edges in all three panels
MIN_PER_BIN = 10
POINT_SIZE = 5
BETA_LIMITS = None        # e.g. (-2.8, -1.0); None includes the full sample
OUTPUT_DIR = Path(".")
SHOW = True

redshift_bins = [
    (r"$z \approx 6$–$7$", ["036", "030"]),
    (r"$z \approx 8$–$9$", ["026", "022"]),
    (r"$z \approx 10$–$11$", ["019", "016"]),
]
template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)
template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])


def load_samples():
    import caesar
    from utils.beta_utils import Calbeta

    if BETA_KIND not in ("attenuated", "intrinsic"):
        raise ValueError("BETA_KIND must be 'attenuated' or 'intrinsic'.")
    samples = []
    for label, snapshots in redshift_bins:
        chunks = []
        for snap in snapshots:
            for template, magnitude_cut in ((template_m25, -16), (template_m50, -17.5)):
                filename = template.format(snap, DUST_LAW)
                catalogue = caesar.load(filename)
                galaxies = catalogue.galaxies
                if not len(galaxies):
                    print(f"Empty catalogue: {filename}")
                    continue
                observed = np.array([[g.absmag[b] for g in galaxies] for b in bands])
                mags = observed if BETA_KIND == "attenuated" else np.array([
                    [g.absmag_nodust[b] for g in galaxies] for b in bands
                ])
                mass = np.array([float(g.masses["stellar"].to("Msun").value) for g in galaxies])
                sfr = np.array([float(g.sfr.to("Msun/yr").value) for g in galaxies])
                selected = np.isfinite(observed[0]) & (observed[0] < magnitude_cut)
                valid = (selected & np.all(np.isfinite(mags), axis=0)
                         & np.isfinite(mass) & (mass > 0)
                         & np.isfinite(sfr) & (sfr > 0))
                indices = np.flatnonzero(valid)
                if indices.size:
                    beta = np.asarray(Calbeta(mags[:, indices], wavelengths), dtype=float).reshape(-1)
                    if beta.size != indices.size:
                        raise ValueError("Calbeta must return one slope per galaxy.")
                    rows = np.column_stack((np.log10(mass[indices]),
                                            np.log10(sfr[indices] / mass[indices]), beta))
                    rows = rows[np.all(np.isfinite(rows), axis=1)]
                    chunks.append(rows)
                    count = len(rows)
                else:
                    count = 0
                print(f"{snap}, {Path(filename).parent.name}: {count}/{selected.sum()} selected galaxies plotted")
                del catalogue, galaxies
        samples.append(np.concatenate(chunks) if chunks else np.empty((0, 3)))
    return samples


def binned_median(rows, edges):
    """Median log sSFR at median log mass; omit bins below MIN_PER_BIN."""
    x = np.full(len(edges) - 1, np.nan)
    y = x.copy()
    for i in range(len(x)):
        inside = (rows[:, 0] >= edges[i]) & (
            (rows[:, 0] <= edges[i + 1]) if i == len(x) - 1 else (rows[:, 0] < edges[i + 1]))
        if inside.sum() >= MIN_PER_BIN:
            x[i], y[i] = np.median(rows[inside, :2], axis=0)
    return x, y


def make_figure(samples):
    nonempty = [r for r in samples if len(r)]
    if not nonempty:
        raise RuntimeError("No valid galaxies passed the selection.")
    pooled = np.concatenate(nonempty)
    lo, hi = pooled[:, 0].min(), pooled[:, 0].max()
    start = np.floor(lo / MASS_BIN_WIDTH) * MASS_BIN_WIDTH
    stop = (np.floor(hi / MASS_BIN_WIDTH) + 1) * MASS_BIN_WIDTH
    edges = np.arange(start, stop + MASS_BIN_WIDTH * 0.5, MASS_BIN_WIDTH)
    vmin, vmax = BETA_LIMITS if BETA_LIMITS is not None else (pooled[:, 2].min(), pooled[:, 2].max())
    if vmax == vmin:
        vmin, vmax = vmin - 0.05, vmax + 0.05
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharex=True, sharey=True,
                             layout="constrained")
    rng = np.random.default_rng(42)
    for i, (ax, rows, (label, _)) in enumerate(zip(axes, samples, redshift_bins)):
        if len(rows):
            # Shuffle draw order to avoid systematically hiding one box/snapshot.
            points = rows[rng.permutation(len(rows))]
            ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="viridis",
                       norm=norm, s=POINT_SIZE, linewidths=0, rasterized=True, zorder=1)
            x, y = binned_median(rows, edges)
            line, = ax.plot(x, y, "o-", color="black", lw=1.5, ms=3,
                            markerfacecolor="white", label="Median", zorder=3)
            line.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
        else:
            ax.text(0.5, 0.5, "No valid galaxies", transform=ax.transAxes, ha="center")
        ax.set_title(label)
        ax.text(0.04, 0.96, f"({chr(97+i)})  N = {len(rows):,}", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
        ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        ax.minorticks_on()
        if len(rows):
            ax.legend(loc="lower left", frameon=True, framealpha=0.85, edgecolor="none", fontsize=9)
    axes[0].set_ylabel(r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$")
    # Include every valid point, with identical limits across panels.
    for dim, setter in ((0, axes[0].set_xlim), (1, axes[0].set_ylim)):
        low, high = pooled[:, dim].min(), pooled[:, dim].max()
        pad = max(0.04 * (high - low), 0.05)
        setter(low - pad, high + pad)
    below, above = pooled[:, 2].min() < vmin, pooled[:, 2].max() > vmax
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                            ax=axes, pad=0.025, fraction=0.035, extend=extend)
    colourbar.set_label(r"Attenuated $\beta$ (Calzetti)" if BETA_KIND == "attenuated" and DUST_LAW == "calzetti"
                       else f"{BETA_KIND.title()} " + r"$\beta$")
    return fig


def main():
    fig = make_figure(load_samples())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"StellarMass_sSFR_Beta_{BETA_KIND}_{DUST_LAW}_RedshiftBins"
    for suffix in ("png", "pdf"):
        filename = stem.with_suffix("." + suffix)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
