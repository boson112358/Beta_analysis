#!/usr/bin/env python3
"""Plot only the two-panel matched nebular comparison (PDF and PNG).

Run: python section5_nebular_blue_tail.py
Dependencies: caesar, numpy, pandas, matplotlib.

(a) Dust-free beta with and without nebular emission for matched galaxies.
(b) Cumulative distributions of beta for the four dust/nebular combinations.

Selection and three-band beta fits are unchanged: observed i1500 from
FullSpectra_Fit, m25 < -16 and m50 < -17.5, matched by GroupID within each
box and snapshot. All four cases use the same finite matched sample.
The pooled distributions are unweighted galaxy-snapshot entries at z=6-11.
GroupID matching assumes the same underlying galaxy catalogue in both runs.
Only this figure is saved; no other figures or data tables are generated.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/cosma8/data/dp376/dc-xian3/simba-eor/EoRData")
SNAPSHOTS = {"036": 6, "030": 7, "026": 8, "022": 9, "019": 10, "016": 11}
CUTS = {"m25n1024": -16.0, "m50n1024": -17.5}
BANDS = ("i1500", "i2300", "i2800")
WAVELENGTHS = np.array([1500., 2300., 2800.])
CASES = {
    "beta_star": ("Stars only", "#009E73", ":", "^"),
    "beta_star_neb": ("Stars + nebular", "#CC79A7", "-.", "D"),
    "beta_star_dust": ("Stars + dust", "#D55E00", "--", "s"),
    "beta_fiducial": ("Stars + nebular + dust", "#0072B2", "-", "o"),
}
GROUPS = {"6-7": (6, 7), "8-9": (8, 9), "10-11": (10, 11)}


def beta_from_ab(mags):
    """Unweighted three-band fit to log10(f_lambda) vs log10(wavelength).

    AB magnitudes give log10(f_nu) = -0.4*m + constant;
    f_lambda is proportional to f_nu/lambda**2. Constants cancel in slope.
    Input shape (N,3); a row must have all three finite magnitudes.
    """
    mags = np.asarray(mags, dtype=float)
    if mags.ndim != 2 or mags.shape[1] != 3:
        raise ValueError("Expected magnitudes with shape (N, 3)")
    x = np.log10(WAVELENGTHS)
    xc = x - x.mean()
    return ((-0.4 * mags - 2.0 * x) @ xc) / np.sum(xc**2)


def scalar(value):
    return float(np.asarray(value))


def catalogue_frame(obj, nebular):
    """Read documented CAESAR magnitude dictionaries and stable GroupIDs."""
    rows = []
    for g in obj.galaxies:
        if not hasattr(g, "GroupID"):
            raise ValueError("Galaxy has no GroupID; cannot safely match catalogues.")
        raw_id = scalar(g.GroupID)
        if not np.isfinite(raw_id) or raw_id != int(raw_id):
            raise ValueError(f"Invalid GroupID: {raw_id}")
        row = {"galaxy_id": int(raw_id)}
        for field, suffix in (("absmag", "dust"), ("absmag_nodust", "nodust")):
            try:
                values = getattr(g, field)
                for band in BANDS:
                    row[f"{band}_{suffix}"] = scalar(values[band])
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError(f"Galaxy {row['galaxy_id']}: missing {field}/{BANDS}") from exc
        rows.append(row)
    columns = ["galaxy_id"] + [f"{b}_{s}" for s in ("dust", "nodust") for b in BANDS]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.galaxy_id.duplicated().any():
        raise ValueError("Duplicate GroupIDs within a catalogue")
    for suffix, key in (("dust", "beta_fiducial" if nebular else "beta_star_dust"),
                        ("nodust", "beta_star_neb" if nebular else "beta_star")):
        mags = frame[[f"{b}_{suffix}" for b in BANDS]].to_numpy(dtype=float)
        # Nonfinite and obvious missing-data sentinels cannot enter any fit.
        valid = np.isfinite(mags).all(axis=1) & (np.abs(mags) < 90).all(axis=1)
        values = np.full(len(frame), np.nan)
        values[valid] = beta_from_ab(mags[valid])
        frame[key] = values
    return frame.rename(columns={c: f"{c}_{'on' if nebular else 'off'}"
                                 for c in columns if c != "galaxy_id"})


def match_pair(on, off, cut):
    """Fixed fiducial selection followed by ID matching and common validity."""
    m = on.i1500_dust_on.to_numpy(dtype=float)
    valid_selection = np.isfinite(m) & (np.abs(m) < 90)
    selected = on.loc[valid_selection & (m < cut)]
    merged = selected.merge(off, on="galaxy_id", how="inner", validate="one_to_one")
    valid = np.isfinite(merged[list(CASES)].to_numpy(dtype=float)).all(axis=1)
    audit = dict(n_on=len(on), n_off=len(off),
                 n_invalid_selection_mag=int((~valid_selection).sum()),
                 n_selected=len(selected), n_matched=len(merged),
                 n_selected_unmatched=len(selected) - len(merged),
                 n_matched_invalid=int((~valid).sum()), n_used=int(valid.sum()))
    return merged.loc[valid].copy(), audit


def load_data(args):
    import caesar  # lazy import: --help and numerical checks work without CAESAR
    frames = []
    # Fail before processing if any of the expected files is absent.
    pairs = []
    for snap, nominal_z in SNAPSHOTS.items():
        for box, cut in CUTS.items():
            filename = f"caesar_{box}_{snap}_{args.dust_law}.hdf5"
            on_path = args.root / args.nebular_dir / box / filename
            off_path = args.root / args.no_nebular_dir / box / filename
            pairs.append((snap, nominal_z, box, cut, on_path, off_path))
    missing = [str(p) for pair in pairs for p in pair[-2:] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing catalogues (no snapshots silently skipped):\n" + "\n".join(missing))
    for snap, nominal_z, box, cut, on_path, off_path in pairs:
        print(f"Loading {box}, snapshot {snap}", flush=True)
        on_obj, off_obj = caesar.load(str(on_path)), caesar.load(str(off_path))
        z = scalar(on_obj.simulation.redshift)
        if not np.isclose(z, scalar(off_obj.simulation.redshift), atol=1e-5, rtol=0):
            raise ValueError(f"Redshift mismatch: {box}/{snap}")
        data, audit = match_pair(catalogue_frame(on_obj, True),
                                 catalogue_frame(off_obj, False), cut)
        data["box"], data["snapshot"] = box, snap
        data["redshift"], data["nominal_z"] = z, nominal_z
        data["global_id"] = box + ":" + snap + ":" + data.galaxy_id.astype(str)
        audit.update(box=box, snapshot=snap, redshift=z, magnitude_cut=cut,
                     on_path=str(on_path), off_path=str(off_path))
        frames.append(data)
        print(f"  selected={audit['n_selected']}, unmatched={audit['n_selected_unmatched']}, "
              f"invalid={audit['n_matched_invalid']}, used={audit['n_used']}", flush=True)
        del on_obj, off_obj
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError("No valid matched galaxies pass the fixed magnitude cuts")
    return result


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 10,
                         "axes.labelsize": 11, "legend.fontsize": 8,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "xtick.minor.visible": True, "ytick.minor.visible": True,
                         "axes.linewidth": .8, "legend.frameon": False})


def save(fig, name, args):
    for extension in ("pdf", "png"):
        fig.savefig(args.output / f"{name}_{args.dust_law}.{extension}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def cdf_panel(ax, sub, xmin):
    if sub.empty:
        ax.text(.5, .5, "No selected galaxies", transform=ax.transAxes, ha="center")
    else:
        # Direct evaluation gives the exact STRICT fraction beta < b, even with ties.
        for key, (label, colour, line, _) in CASES.items():
            values = np.sort(sub[key].to_numpy())
            grid = np.unique(np.r_[xmin, values, np.nextafter(values, np.inf), -1.8])
            ax.plot(grid, np.searchsorted(values, grid, side="left") / len(values),
                    color=colour, ls=line, lw=1.5, label=label)
    ax.set(xlim=(xmin, -1.8), ylim=(0, 1.01), xlabel=r"UV continuum slope",
           ylabel="Cumulative distribution")


def make_plots(data, args):
    style()
    xmin = min(-3.15, float(data[list(CASES)].min().min()) - .05)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3), constrained_layout=True)
    ax = axes[0]
    for label, zs in GROUPS.items():
        sub = data.loc[data.nominal_z.isin(zs)]
        ax.scatter(sub.beta_star, sub.beta_star_neb, s=3, alpha=.22,
                   linewidths=0, rasterized=True, label=fr"$z\simeq {label}$")
    lo = min(data.beta_star.min(), data.beta_star_neb.min(), -3.05) - .05
    hi = max(data.beta_star.max(), data.beta_star_neb.max()) + .05
    ax.plot([lo, hi], [lo, hi], color=".3", ls="--", lw=.8)
    ax.axvline(-3, color=".6", ls=":", lw=.8)
    ax.axhline(-3, color=".6", ls=":", lw=.8)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=r"Stellar-only $\beta_\star$",
           ylabel=r"Dust-free $\beta_{\star+\mathrm{neb}}$")
    ax.text(.05, .95, "(a)", transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(loc="lower right", markerscale=2)
    cdf_panel(axes[1], data, xmin)
    axes[1].text(.05, .95, "(b)", transform=axes[1].transAxes, va="top", fontsize=12)
    axes[1].legend(loc="upper left", bbox_to_anchor=(.02, .88))
    save(fig, "matched_comparison", args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--nebular-dir", default="FullSpectra_Fit")
    parser.add_argument("--no-nebular-dir", default="No_NebularEmi")
    parser.add_argument("--dust-law", default="calzetti")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.output = args.output or Path(f"Section5_nebular_{args.dust_law}")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    make_plots(data, args)
    print(f"Saved matched_comparison_{args.dust_law}.pdf and .png to {args.output.resolve()}")


if __name__ == "__main__":
    main()
