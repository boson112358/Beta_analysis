#!/usr/bin/env python3
"""Figure B: properties of blue stellar-only candidates and matched controls.

Run AFTER section5_nebular_blue_tail.py, in the same CAESAR environment:
    python section5_figureB_blue_properties.py
    python section5_figureB_blue_properties.py --threshold -3.0

Requires numpy, pandas, matplotlib, caesar. Reads matched_galaxies.csv from
Section5_nebular_calzetti and the corresponding FullSpectra_Fit catalogues.
No dependency on the earlier Python script itself. Its fixed magnitude cuts
and galaxy selection are preserved by using only rows of its output CSV.

Candidates: beta_star < threshold (default -2.8). Controls: beta_star >=
threshold, nearest log stellar mass within 0.1 dex, SAME box and snapshot.
Matching is with replacement: each candidate has one equally weighted control
entry, and the same control galaxy can occur more than once. Unique-control
counts, reuse, unmatched candidates and mass differences are saved. This is
an exploratory distribution comparison; no independent-sample significance
or uncertainty estimates are claimed. Results describe the matched subset.

Figure: three cumulative-distribution panels, age, stellar metallicity and
Delta beta_dust = beta_fiducial - beta_star_neb. Ages and metallicities are
mass-weighted catalogue quantities, NOT UV-weighted. No young-star fractions
are inferred from a mass-weighted age. The dust delta has nebular emission ON.

Catalogue fields (configurable): masses['stellar'], ages['mass_weighted'],
metallicities['stellar']. The metallicity key MUST refer to STARS; do not
substitute a gas mass-weighted metallicity. Unit-bearing fields are converted.
For bare numeric fields, mass is assumed Msun, age Gyr (configurable), and
metallicity an absolute metal mass fraction (Zsun=0.0134, configurable).
Use --metallicity-numeric solar if bare metallicities already mean Z/Zsun.
Check these assumptions against your catalogue conventions.

PDF/PNG and matched pairs, enriched galaxies, excluded entries, matching audit,
percentile summaries and metadata are written under Section5_figureB_calzetti.
No candidate threshold is silently relaxed if no galaxies satisfy it.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/cosma8/data/dp376/dc-xian3/simba-eor/EoRData')
PROPERTIES = [('age_myr', r'Mass-weighted stellar age (Myr)'),
              ('stellar_z_solar', r'Stellar metallicity, $Z_\star/Z_\odot$'),
              ('delta_dust', r'$\Delta\beta_{\mathrm{dust}}$ (nebular on)')]


def quantity(value, unit, bare_factor=1.):
    if hasattr(value, 'to'):
        return float(np.asarray(value.to(unit)))
    return float(np.asarray(value)) * bare_factor


def metallicity(value, args):
    if hasattr(value, 'to'):
        # Explicitly adopt the same numeric solar mass fraction for all galaxies.
        try:
            return float(np.asarray(value.to('dimensionless'))) / args.z_sun
        except Exception as exc:
            raise ValueError('Cannot convert stellar metallicity to a dimensionless mass fraction') from exc
    return float(np.asarray(value)) / (args.z_sun if args.metallicity_numeric == 'mass_fraction' else 1.)


def enrich(data, args):
    import caesar
    rows = []
    for (box, snap), sub in data.groupby(['box', 'snapshot'], sort=True):
        path = args.root / args.nebular_dir / box / f'caesar_{box}_{snap}_{args.dust_law}.hdf5'
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f'Loading properties: {box}/{snap}', flush=True)
        obj = caesar.load(str(path))
        cat = {}
        for g in obj.galaxies:
            gid = int(g.GroupID)
            if gid in cat:
                raise ValueError(f'Duplicate GroupID {gid} in {path}')
            cat[gid] = g
        z = float(obj.simulation.redshift)
        if not np.allclose(sub.redshift, z, rtol=0, atol=1e-5):
            raise ValueError(f'Redshift mismatch between CSV and {path}')
        for row in sub.to_dict('records'):
            gid = int(row['galaxy_id'])
            if gid not in cat:
                raise ValueError(f'CSV GroupID {gid} absent from {path}')
            g = cat[gid]
            try:
                mass = quantity(g.masses['stellar'], 'Msun')
                age = quantity(g.ages[args.age_key], 'Myr', 1000 if args.age_numeric == 'Gyr' else 1)
                zstar = metallicity(g.metallicities[args.metallicity_key], args)
            except (KeyError, AttributeError) as exc:
                raise ValueError(f'Missing physical field for {box}/{snap}/{gid}. '
                                 f'Available metallicity keys: {list(getattr(g, "metallicities", {}))}; '
                                 f'age keys: {list(getattr(g, "ages", {}))}. '
                                 'Specify the correct STELLAR metallicity key explicitly.') from exc
            row.update(stellar_mass_msun=mass, log_mass=np.log10(mass) if mass > 0 else np.nan,
                       age_myr=age, stellar_z_solar=zstar,
                       delta_dust=row['beta_fiducial'] - row['beta_star_neb'])
            rows.append(row)
        del cat, obj
    return pd.DataFrame(rows)


def match_controls(data, threshold, tolerance):
    """Exact snapshot/box, nearest mass with replacement; deterministic ID tie-break."""
    pairs, audit = [], []
    for (box, snap), sub in data.groupby(['box', 'snapshot'], sort=True):
        blue = sub.loc[sub.beta_star < threshold].sort_values('global_id')
        control = sub.loc[sub.beta_star >= threshold].sort_values(['log_mass', 'global_id'])
        masses = control.log_mass.to_numpy()
        # Keep one representative per exact mass to make ties reproducible.
        unique, first = np.unique(masses, return_index=True)
        matched = 0
        for _, candidate in blue.iterrows():
            if not len(unique):
                continue
            j = np.searchsorted(unique, candidate.log_mass)
            options = sorted(set(max(0, min(len(unique)-1, k)) for k in (j-1, j)))
            best = min(options, key=lambda k: (abs(unique[k]-candidate.log_mass), k))
            c = control.iloc[first[best]]
            offset = float(c.log_mass - candidate.log_mass)
            if abs(offset) > tolerance:
                continue
            record = dict(box=box, snapshot=snap, nominal_z=candidate.nominal_z,
                          candidate_id=candidate.global_id, control_id=c.global_id,
                          delta_log_mass=offset)
            for field in ['log_mass', 'beta_star'] + [p[0] for p in PROPERTIES]:
                record[f'candidate_{field}'] = candidate[field]
                record[f'control_{field}'] = c[field]
            pairs.append(record)
            matched += 1
        audit.append(dict(box=box, snapshot=snap, n_candidates=len(blue),
                          n_control_pool=len(control), n_matched=matched,
                          n_unmatched=len(blue)-matched))
    return pd.DataFrame(pairs), pd.DataFrame(audit)


def ecdf(ax, values, label, colour, linestyle):
    x, counts = np.unique(np.asarray(values), return_counts=True)
    y = np.cumsum(counts) / counts.sum()
    # F(X <= x): repeated controls contribute their matching multiplicity.
    ax.step(np.r_[x[0], x], np.r_[0, y], where='post', label=label,
            color=colour, ls=linestyle, lw=1.6)


def plot_figure(pairs, args):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 10, 'axes.labelsize': 10,
                         'legend.fontsize': 9, 'legend.frameon': False,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True,
                         'xtick.minor.visible': True, 'ytick.minor.visible': True})
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.85), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.73, wspace=.13)
    for i, (ax, (field, label)) in enumerate(zip(axes, PROPERTIES)):
        ecdf(ax, pairs[f'candidate_{field}'], fr'Blue candidates: $\beta_\star<{args.threshold:g}$', '#0072B2', '-')
        ecdf(ax, pairs[f'control_{field}'], 'Mass-matched controls', '#D55E00', '--')
        ax.set_xlabel(label)
        ax.set_ylim(0, 1.02)
        ax.text(.04, .96, f'({"abc"[i]})', transform=ax.transAxes, va='top')
    axes[0].set_ylabel('Cumulative fraction')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=2)
    fig.text(.5, .81, f'{len(pairs):,} matched candidates; '
             f'{pairs.control_id.nunique():,} unique controls (reuse allowed)', ha='center', fontsize=9)
    for ext in ('pdf', 'png'):
        fig.savefig(args.output / f'FigureB_blue_candidate_properties_{args.dust_law}.{ext}', dpi=300)
    plt.close(fig)


def summaries(pairs):
    rows=[]
    groups=[('all', pairs)] + [(f'z_{lo}-{hi}', pairs.loc[pairs.nominal_z.between(lo, hi)])
                               for lo, hi in [(6,7),(8,9),(10,11)]]
    for label, group in groups:
        if group.empty:
            continue
        for field in ['log_mass', 'beta_star'] + [p[0] for p in PROPERTIES]:
            for population in ['candidate', 'control']:
                p16, p50, p84 = np.percentile(group[f'{population}_{field}'], [16,50,84])
                rows.append(dict(group=label, property=field, population=population,
                                 n_entries=len(group), p16=p16, median=p50, p84=p84))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=Path('Section5_nebular_calzetti/matched_galaxies.csv'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--nebular-dir', default='FullSpectra_Fit')
    p.add_argument('--dust-law', default='calzetti')
    p.add_argument('--output', type=Path, default=Path('Section5_figureB_calzetti'))
    p.add_argument('--threshold', type=float, default=-2.8)
    p.add_argument('--mass-tolerance', type=float, default=.1, help='Maximum absolute log10 mass difference, dex')
    p.add_argument('--metallicity-key', default='stellar')
    p.add_argument('--age-key', default='mass_weighted')
    p.add_argument('--metallicity-numeric', choices=['mass_fraction', 'solar'], default='mass_fraction')
    p.add_argument('--z-sun', type=float, default=.0134)
    p.add_argument('--age-numeric', choices=['Gyr','Myr'], default='Gyr')
    args=p.parse_args()
    if not np.isfinite([args.threshold,args.mass_tolerance,args.z_sun]).all() or args.mass_tolerance <= 0 or args.z_sun <= 0:
        p.error('Require finite threshold and positive finite mass tolerance and Zsun')
    data=pd.read_csv(args.input, dtype={'snapshot':str, 'box':str, 'global_id':str})
    required=['global_id','galaxy_id','box','snapshot','redshift','nominal_z','beta_star','beta_star_neb','beta_fiducial']
    if set(required)-set(data):
        raise ValueError(f'Missing input columns: {set(required)-set(data)}')
    data['snapshot']=data.snapshot.str.zfill(3)
    if data.global_id.duplicated().any() or data.duplicated(['box','snapshot','galaxy_id']).any():
        raise ValueError('Duplicate galaxy identities in input')
    if data.empty:
        raise ValueError('Input catalogue is empty')
    args.output.mkdir(parents=True, exist_ok=True)
    enriched=enrich(data,args)
    fields=['log_mass','age_myr','stellar_z_solar','delta_dust','beta_star','beta_star_neb','beta_fiducial']
    valid=np.isfinite(enriched[fields].to_numpy(dtype=float)).all(axis=1)
    valid &= (enriched.age_myr >= 0) & (enriched.stellar_z_solar >= 0)
    enriched['valid_properties']=valid
    enriched.to_csv(args.output/'galaxies_with_properties.csv',index=False)
    enriched.loc[~valid].to_csv(args.output/'excluded_properties.csv',index=False)
    usable=enriched.loc[valid].copy()
    pairs,audit=match_controls(usable,args.threshold,args.mass_tolerance)
    audit.to_csv(args.output/'matching_audit.csv',index=False)
    meta={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    meta.update(n_input=len(data),n_invalid_properties=int((~valid).sum()),n_pairs=len(pairs),
                matching='Nearest log mass within same box and snapshot; with replacement',
                stellar_age='mass-weighted',metallicity_solar_mass_fraction=args.z_sun)
    (args.output/'figureB_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if pairs.empty:
        print('No matched candidates. Audit saved; no comparison figure generated. '
              'The threshold has NOT been changed automatically.')
        return
    pairs.to_csv(args.output/'matched_candidate_control_pairs.csv',index=False)
    reuse=pairs.control_id.value_counts().rename_axis('control_id').reset_index(name='times_used')
    reuse.to_csv(args.output/'control_reuse.csv',index=False)
    summaries(pairs).to_csv(args.output/'property_percentiles.csv',index=False)
    plot_figure(pairs,args)
    print(f'Saved Figure B: {len(pairs)} matched candidates, {pairs.control_id.nunique()} unique controls.')
    print(f'Maximum mass mismatch: {pairs.delta_log_mass.abs().max():.4f} dex; '
          f'invalid properties: {int((~valid).sum())}; unmatched candidates: {audit.n_unmatched.sum()}')
    print(f'Outputs: {args.output.resolve()}')


if __name__=='__main__':
    main()
