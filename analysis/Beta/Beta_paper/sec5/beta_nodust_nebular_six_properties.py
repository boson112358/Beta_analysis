"""Dust-free UV beta with/without nebular emission versus six properties.

Run: python beta_nodust_nebular_six_properties.py
Requires numpy, matplotlib and CAESAR; no beta_utils dependency.
Outputs one 3-row x 2-column figure as PDF and PNG.

Both beta values use absmag_nodust at 1500, 2300 and 2800 Angstrom.
The OLS fit is beta = slope(-0.4 * AB magnitude versus log10 wavelength) - 2.
Catalogue pairs must describe the same simulation snapshot and galaxy finder.
Galaxies are joined by GroupID, never by catalogue position.
Selection uses observed M1500 in the WITH-nebular reference catalogue:
 m25: M1500 < -16; m50: M1500 < -17.5 (as in the supplied script).
Properties, including Av, are taken from that same reference catalogue.
Av is a reference galaxy property even though both plotted slopes are dust-free.
Curves show medians; error bars show the 16th--84th percentile distribution,
not uncertainties on the median. Each pair uses identical galaxies and bins.
The original z~6--7 metallicity-only cut Zstar/Zsun < 0.7 is retained.
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.formatter.use_mathtext": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


DATA_ROOT = Path("/cosma8/data/dp376/dc-xian3/simba-eor/EoRData")
WITH_NEBULAR_DIR = "FullSpectra_Fit"
WITHOUT_NEBULAR_DIR = "No_NebularEmi"
dust_law = "calzetti"
stellar_age_key = "mass_weighted"
stellar_metallicity_max_z6_z7 = 0.7  # Set None to disable this panel-only cut.
N_BINS = 10
MIN_PER_BIN = 5  # Set 1 to include every occupied bin.
OUTPUT_DIR = Path(".")
Y_LIMITS = None  # e.g. (-3.3, -1.5); None includes all plotted error bars.
SHOW_BETA_MINUS_THREE = True
# Same fixed x limits as the supplied script; use {} to show full ranges.
FIXED_XLIM = {
    "stellar_mass": (6.7, 10.15), "ssfr": (-9.55, -7.40),
    "stellar_metallicity": (0.05, 0.65), "Av": (0.0, 3.2),
}
redshift_bins = [
    (r"$z \approx 6,\,7$", ["036", "030"]),
    (r"$z \approx 8,\,9$", ["026", "022"]),
    (r"$z \approx 10,\,11$", ["019", "016"]),
]
COLORS = ["#482878", "#238A8D", "#D8A800"]
MARKERS = ["o", "s", "^"]
PANEL_ORDER = ["stellar_mass", "ssfr", "stellar_metallicity",
               "stellar_age", "Av", "dust_mass"]
BANDS = ["i1500", "i2300", "i2800"]
WAVELENGTHS = np.array([1500., 2300., 2800.])

property_settings = {
    "stellar_mass": {
        "label": r"Stellar mass $\log_{10}(M_\star/M_\odot)$",
        "take_log": True,
    },
    "ssfr": {
        "label": r"Specific SFR $\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
        "take_log": True,
    },
    "stellar_metallicity": {
        "label": r"Stellar metallicity $Z_\star/Z_\odot$",
        "take_log": False,
    },
    "Av": {
        "label": r"$V$-band attenuation $A_V$ (mag)",
        "take_log": False,
    },
    "stellar_age": {
        "label": r"Mass-weighted stellar age (Myr)",
        "take_log": False,
    },
    "dust_mass": {
        "label": r"Dust mass $\log_{10}(M_{\rm dust}/M_\odot)$",
        "take_log": True,
    },

}


def extract_properties(galaxies):
    """Extract properties; missing new fields affect only their own panels."""
    stellar_mass = np.array([
        g.masses["stellar"].to("Msun").value for g in galaxies
    ])

    ssfr = np.array([
        (g.sfr / g.masses["stellar"]).to("1/yr").value
        for g in galaxies
    ])

    stellar_metallicity = np.array([
        g.metallicities["stellar"].to("Zsun").value
        for g in galaxies
    ], dtype=float)
    
    Av = np.array([
        g.absmag["v"] - g.absmag_nodust["v"]
        for g in galaxies
    ])

    def optional_values(group, key, unit):
        values = []
        missing = 0
        for g in galaxies:
            try:
                value = getattr(g, group)[key]
            except (AttributeError, KeyError):
                values.append(np.nan)
                missing += 1
                continue
            # Require explicit units; do not silently guess ages or Z units.
            values.append(float(value.to(unit).value))
        if missing:
            warnings.warn(f"{group}[{key!r}] missing for {missing} galaxies; "
                          "excluded from affected panels only.")
        return np.asarray(values, dtype=float)

    stellar_age = optional_values("ages", stellar_age_key, "Myr")
    dust_mass = optional_values("masses", "dust", "Msun")
    return {
        "stellar_age": stellar_age,
        "dust_mass": dust_mass,
        "stellar_mass": stellar_mass,
        "ssfr": ssfr,
        "stellar_metallicity": stellar_metallicity,
        "Av": Av,
    }

def beta_nodust(galaxies):
    """Three-point unweighted power-law fit to AB magnitudes."""
    mags = np.array([[float(g.absmag_nodust[b]) for b in BANDS]
                     for g in galaxies], dtype=float).reshape(-1, 3)
    result = np.full(len(galaxies), np.nan)
    good = np.all(np.isfinite(mags), axis=1)
    x = np.log10(WAVELENGTHS)
    x -= x.mean()
    result[good] = (-0.4 * mags[good]) @ x / (x @ x) - 2.0
    return result


def galaxy_index(galaxies, path):
    index = {}
    for g in galaxies:
        try:
            gid = int(g.GroupID)
        except AttributeError as exc:
            raise RuntimeError(f"Missing GroupID in {path}; cannot safely match galaxies") from exc
        if gid in index:
            raise ValueError(f"Duplicate galaxy GroupID={gid} in {path}")
        index[gid] = g
    return index


def load_pair(snapshot, box, magnitude_cut):
    import caesar
    filename = f"caesar_{box}_{snapshot}_{dust_law}.hdf5"
    with_path = DATA_ROOT / WITH_NEBULAR_DIR / box / filename
    without_path = DATA_ROOT / WITHOUT_NEBULAR_DIR / box / filename
    for path in (with_path, without_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required catalogue not found: {path}")
    reference = caesar.load(str(with_path))
    alternative = caesar.load(str(without_path))
    ref_index = galaxy_index(reference.galaxies, with_path)
    alt_index = galaxy_index(alternative.galaxies, without_path)
    selected = [gid for gid, g in ref_index.items()
                if np.isfinite(float(g.absmag["i1500"]))
                and float(g.absmag["i1500"]) < magnitude_cut]
    matched = [gid for gid in selected if gid in alt_index]
    missing = len(selected) - len(matched)
    if missing:
        warnings.warn(f"{box} {snapshot}: {missing} selected IDs absent from no-nebular catalogue")
    if not matched:
        raise ValueError(f"{box} {snapshot}: no matched selected galaxies")
    with_gals = [ref_index[gid] for gid in matched]
    without_gals = [alt_index[gid] for gid in matched]
    # Guard against accidentally joining different underlying galaxy catalogues.
    m1 = np.array([g.masses["stellar"].to("Msun").value for g in with_gals])
    m2 = np.array([g.masses["stellar"].to("Msun").value for g in without_gals])
    if not np.allclose(m1, m2, rtol=1e-5, atol=0, equal_nan=False):
        raise ValueError(f"{box} {snapshot}: matched IDs have different stellar masses; check catalogues")
    beta_with = beta_nodust(with_gals)
    beta_without = beta_nodust(without_gals)
    finite = np.isfinite(beta_with) & np.isfinite(beta_without)
    props = extract_properties(with_gals)
    print(f"{box} snapshot {snapshot}: selected={len(selected)}, matched={len(matched)}, "
          f"finite beta pairs={finite.sum()}")
    return beta_with[finite], beta_without[finite], {k: v[finite] for k, v in props.items()}


def paired_bins(x, y_with, y_without):
    """Use exactly the same bin membership for both spectral variants."""
    edges = np.linspace(x.min(), x.max(), N_BINS + 1)
    ids = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_BINS - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    summaries = np.full((2, 3, N_BINS), np.nan)
    for i in range(N_BINS):
        mask = ids == i
        if mask.sum() >= MIN_PER_BIN:
            for j, y in enumerate((y_with, y_without)):
                summaries[j, :, i] = np.percentile(y[mask], [16, 50, 84])
    return centers, summaries


def main():
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 8.6), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.89,
                        hspace=0.31, wspace=0.10)
    plotted_bounds = []
    for zi, (label, snapshots) in enumerate(redshift_bins):
        chunks = [load_pair(snap, box, cut) for snap in snapshots
                  for box, cut in (("m25n1024", -16), ("m50n1024", -17.5))]
        bw = np.concatenate([c[0] for c in chunks])
        bn = np.concatenate([c[1] for c in chunks])
        for ax, name in zip(axes.flat, PANEL_ORDER):
            settings = property_settings[name]
            x = np.concatenate([c[2][name] for c in chunks])
            valid = np.isfinite(x) & np.isfinite(bw) & np.isfinite(bn)
            if settings["take_log"]:
                valid &= x > 0
            if name == "stellar_age":
                valid &= x >= 0
            if zi == 0 and name == "stellar_metallicity" and stellar_metallicity_max_z6_z7 is not None:
                valid &= x < stellar_metallicity_max_z6_z7
            print(f"{label}, {name}: {valid.sum()} paired galaxies")
            x = x[valid]
            if settings["take_log"]:
                x = np.log10(x)
            if x.size < MIN_PER_BIN or np.ptp(x) == 0:
                warnings.warn(f"Skipping {label}, {name}: insufficient data/range")
                continue
            centers, summaries = paired_bins(x, bw[valid], bn[valid])
            for qi, (p16, median, p84) in enumerate(summaries):
                _, caps, bars = ax.errorbar(
                    centers, median, yerr=[median-p16, p84-median],
                    fmt="none", ecolor=COLORS[zi], elinewidth=0.65,
                    capsize=1.8 if qi == 0 else 0, capthick=0.6, zorder=1)
                for artist in (*caps, *bars):
                    artist.set_alpha(0.23)
                for bar in bars:
                    bar.set_linestyle("solid" if qi == 0 else "dashed")
                ax.plot(centers, median, color=COLORS[zi], marker=MARKERS[zi],
                        linestyle="-" if qi == 0 else (0, (4, 2.5)),
                        markerfacecolor=COLORS[zi] if qi == 0 else "white",
                        markersize=3.2, markeredgewidth=0.65, linewidth=1.25,
                        zorder=3 if qi == 0 else 4)
                plotted_bounds.extend(p16[np.isfinite(p16)])
                plotted_bounds.extend(p84[np.isfinite(p84)])
    if not plotted_bounds:
        raise RuntimeError("No bins meet MIN_PER_BIN; inspect selections and catalogue fields")
    if SHOW_BETA_MINUS_THREE:
        plotted_bounds.append(-3.0)
    if Y_LIMITS is None:
        lo, hi = min(plotted_bounds), max(plotted_bounds)
        pad = 0.08 * max(hi-lo, 0.1)
        limits = lo-pad, hi+pad
    else:
        limits = Y_LIMITS
    for i, (ax, name) in enumerate(zip(axes.flat, PANEL_ORDER)):
        ax.set_xlabel(property_settings[name]["label"])
        if i % 2 == 0:
            ax.set_ylabel(r"Dust-free UV slope $\beta$")
        ax.set_ylim(*limits)
        if name in FIXED_XLIM:
            ax.set_xlim(*FIXED_XLIM[name])
        if SHOW_BETA_MINUS_THREE:
            ax.axhline(-3, color="0.65", ls=":", lw=0.7, zorder=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.minorticks_on()
        ax.tick_params(which="major", length=3.5, width=0.8)
        ax.tick_params(which="minor", length=1.8, width=0.6)
        ax.text(0.04, 0.96, f"({chr(97+i)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9)
    redshift_handles = [Line2D([], [], color=c, marker=m, ls="none", label=label)
                        for c, m, (label, _) in zip(COLORS, MARKERS, redshift_bins)]
    spectral_handles = [
        Line2D([], [], color="0.2", marker="o", ls="-", label="With nebular emission (no dust)"),
        Line2D([], [], color="0.2", marker="o", markerfacecolor="white",
               ls="--", label="Without nebular emission (no dust)")]
    fig.legend(handles=redshift_handles, frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.54, 0.985))
    fig.legend(handles=spectral_handles, frameon=False, ncol=2, handlelength=3,
               loc="upper center", bbox_to_anchor=(0.54, 0.955))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"Beta_nodust_nebular_six_properties_{dust_law}"
    for extension in ("pdf", "png"):
        output = stem.with_suffix("." + extension)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
