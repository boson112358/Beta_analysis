"""Scatter of dust-free beta versus the per-galaxy nebular contribution.

Requires CAESAR, numpy and matplotlib. Match GroupID within each box/snapshot
(the catalogues must originate from the same CAESAR galaxy catalogue).
Keep galaxies passing the observed (dust-attenuated) M1500 cut in BOTH
nebular settings: m25 < -16; m50 < -17.5. All six snapshots and both boxes
are combined, without subsampling. Beta uses the 1500/2300/2800 A AB mags.
The y-axis is logarithmic; non-positive delta beta values are reported and
excluded from the plot. Outputs: PDF and PNG in the current working directory.
"""
from pathlib import Path
import caesar
import numpy as np
import matplotlib.pyplot as plt

BASE = Path('/cosma8/data/dp376/dc-xian3/simba-eor/EoRData')
SNAPSHOTS = ['016', '019', '022', '026', '030', '036']
BOX_CUTS = {'m25n1024': -16., 'm50n1024': -17.5}
DUST_LAW = 'calzetti'
BANDS = ['i1500', 'i2300', 'i2800']
WAVELENGTHS = np.array([1500., 2300., 2800.])


def beta_from_ab(mags):
    """Unweighted three-band fit: beta = -0.4 dM_AB/dlog10(lambda) - 2."""
    x = np.log10(WAVELENGTHS)
    x -= x.mean()
    return -0.4 * (mags @ x) / (x @ x) - 2.


def load_selected(folder, box, snapshot, cut):
    path = BASE / folder / box / f'caesar_{box}_{snapshot}_{DUST_LAW}.hdf5'
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = caesar.load(str(path))
    selected, seen = {}, set()
    for g in obj.galaxies:
        gid = int(g.GroupID)
        if gid in seen:
            raise ValueError(f'Duplicate GroupID {gid} in {path}')
        seen.add(gid)
        m1500 = float(g.absmag['i1500'])
        mags = np.array([g.absmag_nodust[b] for b in BANDS], dtype=float)
        if np.isfinite(m1500) and m1500 < cut and np.all(np.isfinite(mags)):
            selected[gid] = mags
    return float(obj.simulation.redshift), selected


def main():
    xs, ys = [], []
    for snapshot in SNAPSHOTS:
        for box, cut in BOX_CUTS.items():
            z_on, on = load_selected('FullSpectra_Fit', box, snapshot, cut)
            z_off, off = load_selected('No_NebularEmi', box, snapshot, cut)
            if not np.isclose(z_on, z_off, rtol=0, atol=1e-3):
                raise ValueError(f'Redshift mismatch: {box}, {snapshot}')
            ids = sorted(on.keys() & off.keys())
            print(f'{box}, snapshot {snapshot}, z={z_on:.3f}: '
                  f'selected nebular={len(on)}, no-nebular={len(off)}, '
                  f'matched={len(ids)}')
            if not ids:
                continue
            beta_on = beta_from_ab(np.stack([on[i] for i in ids]))
            beta_off = beta_from_ab(np.stack([off[i] for i in ids]))
            xs.append(beta_off)
            ys.append(beta_on - beta_off)
    if not xs:
        raise ValueError('No matched galaxies pass the magnitude cuts.')
    x, y = np.concatenate(xs), np.concatenate(ys)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError('Non-finite beta values.')
    print(f'Total plotted galaxy-snapshot entries: {len(x)}')
    print(f'Delta beta percentiles (16, 50, 84): {np.percentile(y, [16, 50, 84])}')
    positive = y > 0
    print(f'Excluded from logarithmic y-axis: {np.count_nonzero(~positive)} '
          'galaxies with delta beta <= 0')
    x, y = x[positive], y[positive]
    if not len(y):
        raise ValueError('No positive delta beta values for a logarithmic y-axis.')
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.linewidth': .8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
    })
    fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
    ax.scatter(x, y, s=4, alpha=.18, color='#0072B2',
               linewidths=0, rasterized=True)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\beta_{\mathrm{no\ dust,\ no\ nebular}}$')
    ax.set_ylabel(r'$\Delta\beta_{\mathrm{nebular}}$')
    ax.text(.97, .97, f'$N={len(x):,}$', transform=ax.transAxes,
            ha='right', va='top')
    ax.margins(.04)
    output = f'Beta_nodust_nebular_delta_scatter_{DUST_LAW}'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{output}.{ext}', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
