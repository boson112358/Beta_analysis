#!/usr/bin/env python3
"""DTM and dust mass versus stellar metallicity in three redshift groups.

Run: python dust_metallicity_two_panels.py
Requires numpy, matplotlib and caesar in your simulation analysis environment.
Writes PNG and PDF only; no CSV or beta_utils dependency.

DTM = M_dust / (M_gas * Z_gas), matching the supplied reference script.
Z_gas is the absolute mass-weighted metal mass fraction, not Z_gas/Z_sun.
This expression is a fraction of total metals only if Z_gas includes dust-bound
metals. Otherwise it is dust mass divided by gas-phase metal mass.
Stellar metallicity is used ONLY for the x-axis, not for the denominator.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- User settings -----------------------
DUST_LAW = "calzetti"
BASE_DIR = Path("/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit")
BOXES = [("m25n1024", -16.0), ("m50n1024", -17.5)]
REDSHIFT_BINS = [
    (r"$z\approx6$–7", ["036", "030"]),
    (r"$z\approx8$–9", ["026", "022"]),
    (r"$z\approx10$–11", ["019", "016"]),
]
STYLES = [
    dict(color="#482878", marker="o", linestyle="-"),
    dict(color="#238A8D", marker="s", linestyle="--"),
    dict(color="#D8A800", marker="^", linestyle="-."),
]
APPLY_MAGNITUDE_CUTS = True  # observed Calzetti M1500, as in the beta sample
N_BINS = 10
MIN_GALAXIES_PER_BIN = 10
# Shared, linear stellar-metallicity bins for all redshift groups.
# None uses the full combined range; e.g. (0.0, 0.7) limits the analysis range.
STELLAR_METALLICITY_RANGE = None
LOG_DTM = True  # False plots the ratio linearly and includes dust-free galaxies.
OUTPUT_DIR = Path(".")
SHOW_PLOT = True
# Display limits use populated-bin median x values; binning still uses all data.
X_PADDING_FRACTION = 0.07


def load_sample(snapshots):
    import caesar
    stellar_z, ratios, dust_masses = [], [], []
    for snap in snapshots:
        for box, magnitude_limit in BOXES:
            path = BASE_DIR / box / f"caesar_{box}_{snap}_{DUST_LAW}.hdf5"
            if not path.is_file():
                raise FileNotFoundError(f"Catalogue not found: {path}; edit BASE_DIR.")
            catalogue = caesar.load(str(path))
            selected = invalid = zero_dust = 0
            for galaxy in catalogue.galaxies:
                if APPLY_MAGNITUDE_CUTS:
                    magnitude = float(galaxy.absmag["i1500"])
                    if not np.isfinite(magnitude) or magnitude >= magnitude_limit:
                        continue
                selected += 1
                zstar = float(galaxy.metallicities["stellar"].to("Zsun").value)
                zgas = float(galaxy.metallicities["mass_weighted"].to("dimensionless").value)
                mgas = float(galaxy.masses["gas"].to("Msun").value)
                mdust = float(galaxy.masses["dust"].to("Msun").value)
                if (not np.all(np.isfinite([zstar, zgas, mgas, mdust]))
                        or zstar < 0 or zgas <= 0 or mgas <= 0 or mdust < 0):
                    invalid += 1
                    continue
                if LOG_DTM and mdust == 0:
                    zero_dust += 1
                    continue
                ratio = mdust / (mgas * zgas)
                if not np.isfinite(ratio) or (LOG_DTM and ratio <= 0):
                    invalid += 1
                    continue
                stellar_z.append(zstar)
                ratios.append(ratio)
                dust_masses.append(mdust)
            print(f"{box}, snapshot {snap}: selected={selected}, "
                  f"invalid={invalid}, zero dust excluded={zero_dust}, "
                  f"retained={selected-invalid-zero_dust}")
            del catalogue
    return np.asarray(stellar_z), np.asarray(ratios), np.asarray(dust_masses)


def binned_relation(x, y, edges):
    """Median x and y, and 16th/84th y percentiles in shared x bins.

    Percentiles describe galaxy scatter, not uncertainty on the median.
    NaNs leave gaps in bins with fewer than MIN_GALAXIES_PER_BIN galaxies.
    """
    result = np.full((len(edges) - 1, 4), np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= lo) & ((x <= hi) if i == len(edges)-2 else (x < hi))
        counts[i] = np.count_nonzero(mask)
        if counts[i] >= MIN_GALAXIES_PER_BIN:
            p16, p50, p84 = np.percentile(y[mask], [16, 50, 84])
            result[i] = [np.median(x[mask]), p50, p16, p84]
    return result, counts


def main():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix",
        "font.size": 10, "axes.labelsize": 12, "legend.fontsize": 10,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "savefig.dpi": 300,
    })
    # Pool galaxies equally across the two boxes and snapshots, as in the
    # existing beta-property plots; this is not a volume-weighted statistic.
    samples = [load_sample(snaps) for _, snaps in REDSHIFT_BINS]
    nonempty_x = [x for x, _, _ in samples if x.size]
    if not nonempty_x:
        raise RuntimeError("No valid galaxies remain after the selection.")
    if STELLAR_METALLICITY_RANGE is None:
        combined_x = np.concatenate(nonempty_x)
        lo, hi = float(combined_x.min()), float(combined_x.max())
    else:
        lo, hi = STELLAR_METALLICITY_RANGE
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Stellar metallicity bin limits must be finite and increasing.")
    edges = np.linspace(lo, hi, N_BINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)
    plotted_x = []
    for (label, _), (x, ratio, mdust), style in zip(REDSHIFT_BINS, samples, STYLES):
        y_dtm = np.log10(ratio) if LOG_DTM else ratio
        positive_dust = mdust > 0
        panel_data = [(x, y_dtm),
                      (x[positive_dust], np.log10(mdust[positive_dust]))]
        for panel_index, (ax, (xp, yp)) in enumerate(zip(axes, panel_data)):
            result, counts = binned_relation(xp, yp, edges)
            print(f"{label}, panel {panel_index + 1}: counts = {counts.tolist()}; "
                  f"outside range = {np.count_nonzero((xp < lo) | (xp > hi))}")
            valid = np.isfinite(result[:, 1])
            if not np.any(valid):
                print(f"  No bins have at least {MIN_GALAXIES_PER_BIN} galaxies.")
                continue
            xm, ym, p16, p84 = result.T
            plotted_x.extend(xm[valid])
            ax.errorbar(xm, ym, yerr=[ym-p16, p84-ym], label=label,
                        linewidth=1.3, markersize=4, capsize=2,
                        elinewidth=0.8, capthick=0.8, alpha=0.85, **style)
    if not plotted_x:
        plt.close(fig)
        raise RuntimeError("No populated bins: reduce N_BINS or MIN_GALAXIES_PER_BIN.")

    # Zoom to the points actually displayed, without changing sample selection
    # or bin edges. Sparse unplotted bins no longer create empty axis space.
    xmin, xmax = min(plotted_x), max(plotted_x)
    span = xmax - xmin
    padding = X_PADDING_FRACTION * (span if span > 0 else hi - lo)
    axes[0].set_xlim(max(0.0, xmin - padding), xmax + padding)
    axes[0].set_ylabel(r"$\log_{10}[M_{\rm dust}/(Z_{\rm gas}M_{\rm gas})]$"
                       if LOG_DTM else r"$M_{\rm dust}/(Z_{\rm gas}M_{\rm gas})$")
    axes[1].set_ylabel(r"$\log_{10}(M_{\rm dust}/M_\odot)$")
    for ax, panel_label in zip(axes, ["(a)", "(b)"]):
        ax.set_xlabel(r"Stellar metallicity $Z_\star/Z_\odot$")
        ax.text(0.04, 0.96, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=12)
        ax.minorticks_on()
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, handlelength=2.5)
    fig.tight_layout(pad=0.8, w_pad=1.5, rect=(0, 0, 1, 0.91))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"DustToMetal_and_DustMass_vs_StellarMetallicity_{DUST_LAW}_RedshiftBins"
    for extension in ("png", "pdf"):
        output = OUTPUT_DIR / f"{stem}.{extension}"
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved {output}")
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
