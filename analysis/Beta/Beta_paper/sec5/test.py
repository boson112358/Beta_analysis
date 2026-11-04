import os
import numpy as np
import caesar

from utils.beta_utils import Calbeta

template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/FullSpectra_Fit/m25n1024/"
    "caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/FullSpectra_Fit/m50n1024/"
    "caesar_m50n1024_{}_{}.hdf5"
)

redshifts = ["016", "019", "022", "026", "030", "036"]
dust_law = "calzetti"

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])
threshold = -3.0

for snapshot in redshifts:
    for box, template, magnitude_cut in [
        ("m25n1024", template_m25, -16.0),
        ("m50n1024", template_m50, -17.5),
    ]:
        filename = template.format(snapshot, dust_law)

        print("\n" + "=" * 100)
        print(f"{box} | snapshot {snapshot} | {dust_law}", flush=True)

        if not os.path.isfile(filename):
            print(f"Missing file: {filename}")
            continue

        obj = caesar.load(filename)
        print(f"Redshift: {obj.simulation.redshift:.3f}")

        if len(obj.galaxies) == 0:
            print("No galaxies in this file.")
            del obj
            continue

        mags = np.array([
            [g.absmag[band] for g in obj.galaxies]
            for band in bands
        ])

        mags_nodust = np.array([
            [g.absmag_nodust[band] for g in obj.galaxies]
            for band in bands
        ])

        beta = np.asarray(Calbeta(mags, wavelengths))
        beta_nodust = np.asarray(Calbeta(mags_nodust, wavelengths))

        for sample, mask in [
            ("All galaxies", np.ones(len(obj.galaxies), dtype=bool)),
            (
                f"Observed M1500 < {magnitude_cut:g}",
                np.isfinite(mags[0]) & (mags[0] < magnitude_cut),
            ),
        ]:
            print(f"\n{sample}")
            print(
                f"{'Case':<12} {'N valid':>8} {'Median':>9} "
                f"{'P16':>9} {'P84':>9} {'Std':>9} "
                f"{'Minimum':>9} {'N(beta<-3)':>11} {'Percent':>9}"
            )

            for case, values in [
                ("Attenuated", beta),
                ("No dust", beta_nodust),
            ]:
                selected = values[mask & np.isfinite(values)]

                if selected.size == 0:
                    print(f"{case:<12} No valid beta values")
                    continue

                p16, median, p84 = np.percentile(
                    selected, [16, 50, 84]
                )
                std = np.std(selected)
                count = np.count_nonzero(selected < threshold)
                percent = 100.0 * count / selected.size

                print(
                    f"{case:<12} {selected.size:8d} {median:9.4f} "
                    f"{p16:9.4f} {p84:9.4f} {std:9.4f} "
                    f"{selected.min():9.4f} {count:11d} "
                    f"{percent:8.2f}%",
                    flush=True,
                )

        del obj, mags, mags_nodust, beta, beta_nodust
