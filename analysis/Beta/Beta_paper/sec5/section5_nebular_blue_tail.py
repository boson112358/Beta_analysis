#!/usr/bin/env python3
"""Section 5: nebular emission and the extreme blue tail in SIMBA-EoR.

Run in your CAESAR environment:
    python section5_nebular_blue_tail.py
    python section5_nebular_blue_tail.py --delta-scale log

Dependencies: caesar, numpy, pandas, matplotlib. No local beta_utils needed.
Default selection: observed i1500 from FullSpectra_Fit (dust + nebular),
M25 < -16 and M50 < -17.5. The SAME selected, matched galaxies enter all
four cases. No additional beta cut is applied to the parent sample.

Outputs in Section5_nebular_calzetti/:
  * matched_comparison: direct beta comparison + four-case cumulative curves
  * blue_tail_by_redshift: cumulative curves for z~6-7, 8-9, 10-11
  * nebular_delta: stellar-only beta versus nebular-induced change
  * beta_evolution: all four cases, median and 16th-84th percentile ranges
  * matched_galaxies.csv: four slopes, magnitudes, deltas and candidate flags
  * threshold_statistics.csv: counts/fractions for every snapshot, group, box
    and the pooled selected catalogue (NOT a volume-weighted cosmic sample)
  * threshold_transitions.csv: same-galaxy threshold crossings
  * selection_audit.csv, distribution_statistics.csv, run_metadata.json

All figures are saved as PDF and PNG. Percentile ranges describe population
scatter, NOT errors on the median. Snapshots can include related galaxies;
pooled counts are galaxy-snapshot entries, not unique evolutionary systems.

GroupID matching assumes both post-processing runs use the same underlying
galaxy catalogue. Never substitutes row position for a missing identifier.
No escape fractions are inferred. Nebular-off is a limiting experiment.
The optional log delta plot displays ONLY positive differences and reports
the omitted count; CSVs, other figures and statistics retain all finite data.
"""

import argparse
import json
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
    frames, audits = [], []
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
        audits.append(audit)
        frames.append(data)
        print(f"  selected={audit['n_selected']}, unmatched={audit['n_selected_unmatched']}, "
              f"invalid={audit['n_matched_invalid']}, used={audit['n_used']}", flush=True)
        del on_obj, off_obj
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError("No valid matched galaxies pass the fixed magnitude cuts")
    result["delta_neb_nodust"] = result.beta_star_neb - result.beta_star
    result["delta_neb_dust"] = result.beta_fiducial - result.beta_star_dust
    result["delta_dust_neb_on"] = result.beta_fiducial - result.beta_star_neb
    result["delta_dust_neb_off"] = result.beta_star_dust - result.beta_star
    result["stellar_candidate"] = result.beta_star < args.candidate_threshold
    return result, pd.DataFrame(audits)


def subsets(data):
    yield "all", data
    for box in CUTS:
        yield box, data.loc[data.box == box]
    for snap in SNAPSHOTS:
        yield f"snapshot_{snap}", data.loc[data.snapshot == snap]
    for label, redshifts in GROUPS.items():
        yield f"z_{label}", data.loc[data.nominal_z.isin(redshifts)]
    for box in CUTS:
        for snap in SNAPSHOTS:
            yield f"{box}_snapshot_{snap}", data.loc[(data.box == box) & (data.snapshot == snap)]


def write_statistics(data, args):
    thresholds, transitions, summaries = [], [], []
    for group, sub in subsets(data):
        n = len(sub)
        for key in CASES:
            vals = sub[key].to_numpy()
            for threshold in args.thresholds:
                k = int((vals < threshold).sum())
                thresholds.append(dict(group=group, case=key, threshold=threshold,
                                       n_total=n, n_blue=k, fraction=k/n if n else np.nan))
            q = np.percentile(vals, [1, 5, 16, 50, 84]) if n else [np.nan]*5
            summaries.append(dict(group=group, case=key, n=n,
                                  minimum=float(vals.min()) if n else np.nan,
                                  p01=q[0], p05=q[1], p16=q[2], median=q[3], p84=q[4]))
        for threshold in args.thresholds:
            for start, end in (("beta_star", "beta_star_neb"),
                               ("beta_star_neb", "beta_fiducial"),
                               ("beta_star", "beta_star_dust"),
                               ("beta_star", "beta_fiducial")):
                a, b = sub[start] < threshold, sub[end] < threshold
                transitions.append(dict(group=group, threshold=threshold, start=start, end=end,
                                        n_total=n, n_blue_start=int(a.sum()),
                                        n_remain_blue=int((a & b).sum()),
                                        n_leave_blue=int((a & ~b).sum()),
                                        n_enter_blue=int((~a & b).sum())))
    pd.DataFrame(thresholds).to_csv(args.output / "threshold_statistics.csv", index=False)
    pd.DataFrame(transitions).to_csv(args.output / "threshold_transitions.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "distribution_statistics.csv", index=False)


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


def cdf_panel(ax, sub, thresholds, xmin):
    if sub.empty:
        ax.text(.5, .5, "No selected galaxies", transform=ax.transAxes, ha="center")
    else:
        # Direct evaluation gives the exact STRICT fraction beta < b, even with ties.
        for key, (label, colour, line, _) in CASES.items():
            values = np.sort(sub[key].to_numpy())
            grid = np.unique(np.r_[xmin, values, np.nextafter(values, np.inf), -1.8])
            ax.plot(grid, np.searchsorted(values, grid, side="left") / len(values),
                    color=colour, ls=line, lw=1.5, label=label)
    for t in thresholds:
        ax.axvline(t, color=".65", lw=.65, ls=":", zorder=0)
    ax.set(xlim=(xmin, -1.8), ylim=(0, 1.01), xlabel=r"Threshold, $b$",
           ylabel=r"Selected fraction $F(\beta<b)$")


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
           ylabel=r"Dust-free $\beta_{\star+\mathrm{neb}}$", title="(a) Matched galaxies")
    ax.legend(loc="lower right", markerscale=2)
    cdf_panel(axes[1], data, args.thresholds, xmin)
    axes[1].set_title(f"(b) Blue tail: N = {len(data):,}")
    axes[1].legend(loc="upper left")
    save(fig, "matched_comparison", args)

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.9), sharey=True, constrained_layout=True)
    for ax, (label, zs) in zip(axes, GROUPS.items()):
        sub = data.loc[data.nominal_z.isin(zs)]
        cdf_panel(ax, sub, args.thresholds, xmin)
        ax.set_title(fr"$z\simeq {label}$; $N={len(sub):,}$")
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="outside upper center", ncol=2) if \
        tuple(map(int, matplotlib.__version__.split('.')[:2])) >= (3, 7) else \
        axes[-1].legend(handles, labels, loc="lower right", fontsize=6)
    save(fig, "blue_tail_by_redshift", args)

    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    delta = data.delta_neb_nodust
    mask = delta > 0 if args.delta_scale == "log" else np.ones(len(data), dtype=bool)
    ax.scatter(data.loc[mask, "beta_star"], delta[mask], s=4, alpha=.18,
               color="#0072B2", linewidths=0, rasterized=True)
    ax.set(xlabel=r"Stellar-only $\beta_\star$", ylabel=r"$\Delta\beta_{\mathrm{neb}}$",
           yscale=args.delta_scale)
    if args.delta_scale != "log":
        ax.axhline(0, color=".4", ls="--", lw=.8)
    omitted = len(data) - int(np.sum(mask))
    ax.text(.97, .96, f"N = {int(np.sum(mask)):,}" +
            (f"\nNonpositive omitted: {omitted}" if args.delta_scale == "log" else ""),
            transform=ax.transAxes, va="top", ha="right", fontsize=8)
    save(fig, "nebular_delta", args)
    print(f"Delta <= 0: {int((delta <= 0).sum())}; log-plot omissions: {omitted}")

    fig, ax = plt.subplots(figsize=(4.3, 3.4), constrained_layout=True)
    for key, (label, colour, line, marker) in CASES.items():
        points = []
        for _, sub in data.groupby("snapshot"):
            points.append([sub.redshift.iloc[0], *np.percentile(sub[key], [16, 50, 84])])
        points = np.array(sorted(points))
        z, p16, p50, p84 = points.T
        ax.errorbar(z, p50, yerr=[p50-p16, p84-p50], color=colour, ls=line,
                    marker=marker, mfc="white", ms=4, lw=1.2, elinewidth=.8,
                    capsize=2, alpha=.85, label=label)
    ax.set(xlabel="Redshift", ylabel=r"UV slope, $\beta$")
    ax.legend(loc="best")
    save(fig, "beta_evolution", args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--nebular-dir", default="FullSpectra_Fit")
    parser.add_argument("--no-nebular-dir", default="No_NebularEmi")
    parser.add_argument("--dust-law", default="calzetti")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-2.6, -2.8, -3.0])
    parser.add_argument("--candidate-threshold", type=float, default=-2.8,
                        help="Stellar-only candidate flag; does not select the parent sample")
    parser.add_argument("--delta-scale", choices=["linear", "log", "symlog"], default="linear")
    args = parser.parse_args()
    if not np.isfinite(args.thresholds + [args.candidate_threshold]).all():
        parser.error("Thresholds must be finite")
    args.output = args.output or Path(f"Section5_nebular_{args.dust_law}")
    args.output.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(args)
    audit.to_csv(args.output / "selection_audit.csv", index=False)
    data.to_csv(args.output / "matched_galaxies.csv", index=False)
    write_statistics(data, args)
    make_plots(data, args)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(selection="Fixed observed i1500 from nebular-on, dust-attenuated catalogue",
                    cuts=CUTS, bands=BANDS, wavelengths_angstrom=WAVELENGTHS.tolist(),
                    n_used=len(data), n_delta_nonpositive=int((data.delta_neb_nodust <= 0).sum()),
                    weighting="Unweighted pooled catalogue; not volume weighted",
                    threshold_operator="strictly less than", snapshots=SNAPSHOTS)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved four PDF/PNG figures and audit/statistics tables to {args.output.resolve()}")
    for key in CASES:
        print(f"{key}: minimum={data[key].min():.4f}; "
              f"N(beta < -3)={int((data[key] < -3).sum())}/{len(data)}")


if __name__ == "__main__":
    main()
