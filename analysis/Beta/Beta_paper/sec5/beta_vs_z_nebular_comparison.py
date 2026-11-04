"""UV beta evolution comparing dust and nebular-emission settings.

Run in the same project/environment as beta_vs_z.py (requires CAESAR and
utils.beta_utils.Calbeta). Optional observations: beta_redshift_binned.csv
beside this script. Outputs are written to the current working directory.
Magnitude cuts are applied independently to the two nebular catalogues;
changes in selection can therefore contribute to their difference. The two
no-nebular curves use the same observed-M1500 selection and common finite
photometry, with beta measured from absmag or absmag_nodust respectively.
"""
from pathlib import Path

import caesar
import matplotlib.pyplot as plt
import numpy as np

from utils.beta_utils import Calbeta

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.labelsize": 9, "axes.linewidth": 0.8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.grid": False, "savefig.dpi": 300,
})

SNAPSHOTS = ['016', '019', '022', '026', '030', '036']
DUST_LAW = "calzetti"
BANDS = ["i1500", "i2300", "i2800"]
WAVELENGTHS = np.array([1500, 2300, 2800])
BASE = Path("/cosma8/data/dp376/dc-xian3/simba-eor/EoRData")
BOX_CUTS = {"m25n1024": -16.0, "m50n1024": -17.5}
MODELS = {
    "with_nebular": {
        "folder": "FullSpectra_Fit", "label": "Dust + nebular emission",
        "color": "#0072B2", "fmt": "o-",
    },
    "without_nebular": {
        "folder": "No_NebularEmi", "label": "Dust, no nebular emission",
        "color": "#D55E00", "fmt": "s--",
    },
}
MODELS["nodust_without_nebular"] = {
    "folder": "No_NebularEmi", "label": "No dust, no nebular emission",
    "color": "#009E73", "fmt": "^:", "magnitude_field": "absmag_nodust",
}
SHOW_OBSERVATIONS = True
OBSERVATION_FILE = Path(__file__).with_name("beta_redshift_binned.csv")
OBSERVATION_MARKERS = {
    "Bouwens 2014": "s", "Cullen 2024": "^", "Morales 2024": "D",
    "Napolitano 2026": "v", "Whitler 2025": "p", "Topping 2024": "X",
}


def snapshot_summary(folder, snapshot, magnitude_field="absmag"):
    """Combine both boxes with the original observed-M1500 cuts."""
    selected = []
    z_values = []
    for box, magnitude_cut in BOX_CUTS.items():
        path = BASE / folder / box / f"caesar_{box}_{snapshot}_{DUST_LAW}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing catalogue: {path}")
        obj = caesar.load(str(path))
        z_values.append(float(obj.simulation.redshift))
        # absmag retains dust attenuation in BOTH nebular settings.
        mags = np.array([
            [g.absmag[band] for g in obj.galaxies] for band in BANDS
        ], dtype=float)
        valid = np.all(np.isfinite(mags), axis=0)
        nodust_mags = None
        if folder == "No_NebularEmi":
            nodust_mags = np.array([
                [g.absmag_nodust[band] for g in obj.galaxies] for band in BANDS
            ], dtype=float)
            # Both no-nebular curves retain exactly the same galaxies.
            valid &= np.all(np.isfinite(nodust_mags), axis=0)
        mask = valid & (mags[0] < magnitude_cut)
        beta_mags = nodust_mags if magnitude_field == "absmag_nodust" else mags
        selected.append(beta_mags[:, mask])
        print(f"  {box}: {mask.sum()} selected")
    if not np.allclose(z_values, z_values[0], rtol=0, atol=1e-3):
        raise ValueError(f"Box redshifts disagree for snapshot {snapshot}: {z_values}")
    mags = np.concatenate(selected, axis=1)
    if mags.shape[1] == 0:
        raise ValueError(f"No selected galaxies: {folder}, snapshot {snapshot}")
    beta = np.asarray(Calbeta(mags, WAVELENGTHS), dtype=float).ravel()
    if not np.all(np.isfinite(beta)):
        raise ValueError(f"Non-finite beta: {folder}, {snapshot}, {magnitude_field}")
    if beta.size == 0:
        raise ValueError(f"No finite beta values: {folder}, snapshot {snapshot}")
    p16, median, p84 = np.percentile(beta, [16, 50, 84])
    print(f"  z={z_values[0]:.3f}, N={beta.size}, beta={median:.3f}, "
          f"16-84%=[{p16:.3f}, {p84:.3f}]")
    return z_values[0], median, p16, p84


def print_fit(label, summary):
    # Percentile ranges describe scatter, not uncertainty in the median;
    # use the same unweighted fit as the original script.
    z, med = summary[:, 0], summary[:, 1]
    if len(z) < 3:
        print(f"{label}: at least 3 snapshots needed for fit covariance.")
        return
    (slope, beta6), cov = np.polyfit(z - 6.0, med, 1, cov=True)
    print(f"\n{label}: beta(z) = beta_6 + slope * (z - 6)")
    print(f"  slope = {slope:.4f} +/- {np.sqrt(cov[0, 0]):.4f}")
    print(f"  beta_6 = {beta6:.4f} +/- {np.sqrt(cov[1, 1]):.4f}")


def main():
    results = {}
    for key, model in MODELS.items():
        rows = []
        for snapshot in SNAPSHOTS:
            print(f"\n{model['label']}, snapshot {snapshot}")
            rows.append(snapshot_summary(
                model['folder'], snapshot, model.get('magnitude_field', 'absmag')
            ))
        summary = np.asarray(rows)
        summary = summary[np.argsort(summary[:, 0])]
        results[key] = summary
        print_fit(model['label'], summary)

    with_neb = results['with_nebular']
    without_neb = results['without_nebular']
    if not np.allclose(with_neb[:, 0], without_neb[:, 0], rtol=0, atol=1e-3):
        raise ValueError("Redshifts differ between nebular settings.")
    print("\nDifference of sample medians (with minus without nebular emission):")
    for z, delta in zip(with_neb[:, 0], with_neb[:, 1] - without_neb[:, 1]):
        print(f"  z={z:.3f}: delta beta={delta:+.4f}")

    fig, ax = plt.subplots(figsize=(3.55, 3.35), constrained_layout=True)
    for key, model in MODELS.items():
        z, med, p16, p84 = results[key].T
        ax.errorbar(
            z, med, yerr=[med - p16, p84 - med], fmt=model['fmt'],
            color=model['color'], capsize=2.5, elinewidth=0.9,
            linewidth=1.5, markersize=4.5,
            markerfacecolor=model['color'] if key == 'with_nebular' else 'white',
            markeredgecolor=model['color'], markeredgewidth=0.8,
            zorder=10, label=model['label'],
        )

    if SHOW_OBSERVATIONS and OBSERVATION_FILE.is_file():
        import pandas as pd
        observations = pd.read_csv(OBSERVATION_FILE)
        for dataset, data in observations.groupby('dataset', sort=False):
            ax.errorbar(
                data['redshift'], data['beta'], yerr=data['uncertainty'],
                fmt=OBSERVATION_MARKERS.get(dataset, 'o'), linestyle='none',
                color='0.40', ecolor='0.72', markerfacecolor='white',
                markeredgewidth=0.9, markersize=4.0, capsize=1.5,
                elinewidth=0.7, alpha=0.9, zorder=5, label=dataset,
            )
    elif SHOW_OBSERVATIONS:
        print(f"\nObservational overlay omitted: {OBSERVATION_FILE} not found.")

    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'UV slope, $\beta$')
    ax.text(0.03, 0.97, '(a) Redshift evolution', transform=ax.transAxes,
            ha='left', va='top')
    ax.margins(x=0.06, y=0.15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.19),
              ncol=2, columnspacing=0.8, handletextpad=0.4)
    output = f'Beta_evolution_nebular_comparison_{DUST_LAW}'
    for extension in ('pdf', 'png'):
        fig.savefig(f'{output}.{extension}', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
