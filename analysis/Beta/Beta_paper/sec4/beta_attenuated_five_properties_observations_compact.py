"""Attenuated UV beta versus five properties, with observed measurements.

Run in the same project as utils/beta_utils.py; requires CAESAR, numpy and
matplotlib. Put the five supplied CSV files in obs_data/ beside this script.
The catalogue paths and observed-M1500 selection are retained from the input.
For stellar age and A_V, only point_type='binned' rows are plotted.
Updated filenames ending in '(1).csv' are preferred when present in obs_data;
otherwise the canonical filenames are used (replace those with updated CSVs).
Output: one five-panel figure, saved beside this script as PNG and PDF.

Simulation: three-band attenuated Calbeta; solid median curves and shaded
16th--84th percentile bands. Observations use white-filled, coloured outlines.
Observations: supplied x/y and y-error magnitudes, without re-binning. The CSV
mass and sSFR x values are ALREADY log10; Z is Z/Zsun, age Myr, A_V magnitudes.
Observed binned errors are the supplied scatter, not errors on the mean.
Colours for pre-binned points use their recorded representative redshift;
the original bins may contain galaxies spanning several redshift groups.
No additional UV-luminosity selection is imposed on observations.
The metallicity CSV labels Z/Zsun without identifying stellar vs gas Z;
verify that definition before interpreting it as a stellar-metallicity test.
"""
from pathlib import Path
import csv
import io
import warnings
import numpy as np
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


SCRIPT_DIR = Path(__file__).resolve().parent
OBS_DIR = SCRIPT_DIR / "obs_data"
dust_law = "calzetti"
stellar_age_key = "mass_weighted"
# Retain the paths in the attached script; change these if running on COSMA.
template_m25 = "/home/zxiang/simba-eor/my_dustext_output_updated/m25n1024/caesar_m25n1024_{}_{}.hdf5"
template_m50 = "/home/zxiang/simba-eor/my_dustext_output_updated/m50n1024/caesar_m50n1024_{}_{}.hdf5"
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])
redshift_bins = [
    (r"$z\approx6$--7", ["036", "030"]),
    (r"$z\approx8$--9", ["026", "022"]),
    (r"$z\approx10$--11", ["019", "016"]),
]
# Half-open intervals [5.5,7.5), [7.5,9.5), [9.5,11.5].
OBS_Z_EDGES = np.array([5.5, 7.5, 9.5, 11.5])
redshift_styles = [
    {"color": "#482878", "marker": "o"},
    {"color": "#238A8D", "marker": "s"},
    {"color": "#D8A800", "marker": "^"},
]
DATASET_MARKERS = {"Morales 2024": "D", "Napolitano 2026": "v"}
# Preserve the original panel-specific metallicity cut; None disables it.
stellar_metallicity_max_z6_z7 = 0.7
main_ylim = None
# Automatic x ranges include observations and their errors.
fixed_xlim = {}
property_settings = {
    "stellar_mass": dict(label=r"$\log_{10}(M_\star/M_\odot)$", take_log=True,
                         filename="beta_vs_stellar_mass.csv", unit="log stellar mass [Msun]"),
    "ssfr": dict(label=r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$", take_log=True,
                 filename="beta_vs_sSFR.csv", unit="log sSFR [1/yr]"),
    "stellar_metallicity": dict(label=r"Stellar metallicity $Z_\star/Z_\odot$", take_log=False,
                               filename="beta_vs_metallicity.csv", unit="metallicity (Z/Z_sun)"),
    "Av": dict(label=r"$A_V$ (mag)", take_log=False,
               filename="beta_vs_dust_attenuation.csv", unit="A_V [mag]"),
    "stellar_age": dict(label="Mass-weighted stellar age (Myr)", take_log=False,
                        filename="beta_vs_stellar_age.csv", unit="mass-weighted stellar age [Myr]"),
}


def observation_group(z):
    if not np.isfinite(z) or z < OBS_Z_EDGES[0] or z > OBS_Z_EDGES[-1]:
        return -1
    return min(int(np.searchsorted(OBS_Z_EDGES, z, side="right") - 1), 2)


def load_observations():
    """Validate all five files before spending time loading catalogues."""
    observations = {}
    required = {"x", "y", "redshift", "dataset", "point_type", "x_quantity",
                "y_err_lower", "y_err_upper"}
    for name, settings in property_settings.items():
        path = OBS_DIR / settings["filename"]
        binned_only = name in {"Av", "stellar_age"}
        if binned_only:
            updated_path = path.with_name(path.stem + "(1).csv")
            if updated_path.exists():
                path = updated_path
        rows = []
        excluded = 0
        skipped_individual = 0
        content = path.read_text(encoding="utf-8-sig")
        # Repair ONLY the exact truncated row in the supplied updated A_V file.
        # Verified against the complete matching row in the previous CSV;
        # every numerical value is unchanged. The source CSV is not modified.
        broken = '2024,binned,0.10038461538461538,-2.5046153846153847,0.13954872322542458,0.13954872322542458,26.0,9.375,-18.145000000000003,,"(-0.001, 0.2]",'
        if name == "Av" and broken in content.splitlines():
            restored = "beta_vs_dust_attenuation,A_V [mag],UV beta slope,Morales " + broken
            content = "\n".join(restored if line == broken else line
                                for line in content.splitlines())
            warnings.warn(f"{path.name}: restored missing labels in the first binned row; numerical values unchanged.")
        with io.StringIO(content) as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                if row["point_type"] not in {"individual", "binned"}:
                    raise ValueError(f"{path}:{line}: unknown point_type {row['point_type']!r}")
                if binned_only and row["point_type"] != "binned":
                    skipped_individual += 1
                    continue
                if row["x_quantity"].strip() != settings["unit"]:
                    raise ValueError(f"{path}:{line}: unexpected x units {row['x_quantity']!r}")
                try:
                    x, y, z = (float(row[key]) for key in ("x", "y", "redshift"))
                    errors = [float(row[key]) if row[key].strip() else np.nan
                              for key in ("y_err_lower", "y_err_upper")]
                except ValueError as exc:
                    raise ValueError(f"{path}:{line}: invalid numeric value") from exc
                if any(np.isfinite(v) and v < 0 for v in errors):
                    raise ValueError(f"{path}:{line}: negative error magnitude")
                group = observation_group(z)
                if group < 0 or not np.isfinite([x, y]).all():
                    excluded += 1
                    continue
                if row["dataset"] not in DATASET_MARKERS:
                    raise ValueError(f"Add a marker for dataset {row['dataset']!r}")
                if row["point_type"] not in {"individual", "binned"}:
                    raise ValueError(f"Unknown point_type {row['point_type']!r}")
                rows.append(dict(x=x, y=y, z=z, errors=errors, group=group,
                                 dataset=row["dataset"], point_type=row["point_type"]))
        if binned_only and not rows:
            raise ValueError(f"{path}: no usable binned observations in the selected redshift range")
        observations[name] = rows
        counts = [sum(r["group"] == i for r in rows) for i in range(3)]
        print(f"{path.name}: observations per redshift group {counts}; "
              f"excluded (redshift/invalid values) {excluded}; "
              f"skipped individual rows {skipped_individual}")
    return observations


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
    return {
        "stellar_age": stellar_age,
        "stellar_mass": stellar_mass,
        "ssfr": ssfr,
        "stellar_metallicity": stellar_metallicity,
        "Av": Av,
    }

def load_simulation():
    import caesar
    from utils.beta_utils import Calbeta, bin_xy_median
    results = {name: [] for name in property_settings}
    for group, (label, snapshots) in enumerate(redshift_bins):
        beta_samples = []
        properties = {name: [] for name in property_settings}
        for snap in snapshots:
            for template, mag_cut in ((template_m25, -16), (template_m50, -17.5)):
                path = template.format(snap, dust_law)
                print(f"Loading {path}")
                obj = caesar.load(path)
                if not len(obj.galaxies):
                    continue
                mags = np.asarray([[g.absmag[b] for g in obj.galaxies] for b in bands])
                beta = np.asarray(Calbeta(mags, wavelengths), dtype=float)
                selection = (mags[0] < mag_cut) & np.isfinite(beta)
                beta_samples.append(beta[selection])
                values = extract_properties(obj.galaxies)
                for name in properties:
                    properties[name].append(values[name][selection])
        if not beta_samples:
            raise ValueError(f"No galaxies in simulation group {label}")
        beta = np.concatenate(beta_samples)
        for name, settings in property_settings.items():
            x = np.concatenate(properties[name])
            valid = np.isfinite(x) & np.isfinite(beta)
            if settings["take_log"]:
                valid &= x > 0
            if name == "stellar_age":
                valid &= x >= 0
            if group == 0 and name == "stellar_metallicity" and stellar_metallicity_max_z6_z7 is not None:
                print(f"{label}: excluding {np.count_nonzero(valid & (x >= stellar_metallicity_max_z6_z7))} metallicity outliers")
                valid &= x < stellar_metallicity_max_z6_z7
            x, y = x[valid], beta[valid]
            if settings["take_log"]:
                x = np.log10(x)
            print(f"{label}, {name}: {len(x)} valid galaxies")
            if len(x) < 2 or np.ptp(x) == 0:
                results[name].append(None)
                continue
            centers, median, p16, p84, _ = bin_xy_median(
                x_values=x, y_values=y, mask_values=None, mask_cut=None, N_bins=10)
            results[name].append(tuple(np.asarray(v) for v in (centers, median, p16, p84)))
    return results


def make_figure(simulation, observations):
    # Three panels above, two centred below: exactly five plotting axes.
    fig = plt.figure(figsize=(7.5, 4.5))
    grid = fig.add_gridspec(2, 6, left=0.085, right=0.99, bottom=0.11,
                            top=0.86, wspace=1.05, hspace=0.38)
    locations = [grid[0, 0:2], grid[0, 2:4], grid[0, 4:6],
                 grid[1, 1:3], grid[1, 3:5]]
    axes = []
    for index, ((name, settings), location) in enumerate(zip(property_settings.items(), locations)):
        ax = fig.add_subplot(location)
        axes.append(ax)
        for group, series in enumerate(simulation[name]):
            if series is None:
                continue
            x, median, p16, p84 = series
            style = redshift_styles[group]
            # Bands encode simulation scatter; error bars are reserved for observations.
            ax.fill_between(x, p16, p84, color=style["color"], alpha=0.10,
                            linewidth=0, zorder=1)
            ax.plot(x, median, color=style["color"], linestyle="-",
                    linewidth=1.8, zorder=3)
        for row in observations[name]:
            color = redshift_styles[row["group"]]["color"]
            binned = row["point_type"] == "binned"
            errors = np.asarray(row["errors"])
            yerr = errors[:, None] if np.isfinite(errors).all() else None
            ax.errorbar([row["x"]], [row["y"]], yerr=yerr, linestyle="none",
                marker=DATASET_MARKERS[row["dataset"]], markerfacecolor="white",
                markeredgecolor=color, ecolor=color, color=color,
                markersize=6.0 if binned else 4.2,
                markeredgewidth=1.25 if binned else 0.95,
                elinewidth=1.0, capsize=2.5 if binned else 0,
                alpha=1.0 if binned else 0.70, zorder=5 if binned else 4)
        ax.set_xlabel(settings["label"])
        ax.set_ylabel(r"UV slope $\beta$")
        ax.text(0.04, 0.95, f"({chr(97+index)})", transform=ax.transAxes,
                va="top", fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.minorticks_on()
        ax.tick_params(which="major", length=3.5, width=0.8)
        ax.tick_params(which="minor", length=1.8, width=0.6)
        ax.margins(x=0.08)
        if name in fixed_xlim:
            ax.set_xlim(*fixed_xlim[name])
    bounds = np.asarray([ax.dataLim.intervaly for ax in axes])
    finite = bounds[np.isfinite(bounds)]
    if main_ylim is not None:
        limits = main_ylim
    elif finite.size:
        pad = 0.08 * max(np.ptp(finite), 0.1)
        limits = (finite.min()-pad, finite.max()+pad)
    else:
        limits = (-3.5, -1)
    for ax in axes:
        ax.set_ylim(*limits)
    redshift_handles = [Line2D([], [], color=s["color"], linewidth=2,
                                label=label)
                        for (label, _), s in zip(redshift_bins, redshift_styles)]
    fig.legend(handles=redshift_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.53, 0.944), frameon=False,
               borderaxespad=0, borderpad=0, handletextpad=0.5)
    source_handles = [Line2D([], [], color="0.25", linestyle="-",
                            linewidth=1.8, label="SIMBA-EoR (attenuated)")]
    source_handles.extend(Line2D([], [], color="0.3", marker=marker,
        markerfacecolor="white", markeredgewidth=1.25, linestyle="none",
        markersize=6, label=dataset)
        for dataset, marker in DATASET_MARKERS.items())
    fig.legend(handles=source_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.53, 0.896), frameon=False, columnspacing=1.4,
               borderaxespad=0, borderpad=0, handletextpad=0.5)
    #fig.text(0.53, 0.865, "SIMBA-EoR: median lines + 16–84% bands; observations: open symbols + error bars",
    #         ha="center", fontsize=7.5)
    return fig


def main():
    observations = load_observations()
    simulation = load_simulation()
    fig = make_figure(simulation, observations)
    stem = SCRIPT_DIR / f"BetaAttenuated_FiveProperties_Observations_{dust_law.title()}"
    for extension in ("png", "pdf"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {path}")
    plt.show()


if __name__ == "__main__":
    main()
