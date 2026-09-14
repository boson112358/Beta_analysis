import numpy as np
import caesar


# ============================================================
# Files
# ============================================================

redshifts = ['016', '019', '022', '026', '030', '036']

dust_law = "calzetti"

template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/Dust_extin/m25n1024/"
    "caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/Dust_extin/m50n1024/"
    "caesar_m50n1024_{}_{}.hdf5"
)


# ============================================================
# Magnitude bins
# ============================================================

# Remember: more negative = brighter
mag_bins = [
    (-17.0, -16.0),
    (-18.0, -17.0),
    (-19.0, -18.0),
    (-20.0, -19.0),
    (-np.inf, -20.0)
]

bin_labels = [
    "-16 to -17",
    "-17 to -18",
    "-18 to -19",
    "-19 to -20",
    "< -20"
]


# ============================================================
# Loop over redshifts
# ============================================================

for z_str in redshifts:

    obj_m25 = caesar.load(
        template_m25.format(z_str, dust_law)
    )

    obj_m50 = caesar.load(
        template_m50.format(z_str, dust_law)
    )


    # --------------------------------------------------------
    # Get M1500
    # --------------------------------------------------------

    M1500_m25 = np.array([
        g.absmag["i1500"]
        for g in obj_m25.galaxies
    ])

    M1500_m50 = np.array([
        g.absmag["i1500"]
        for g in obj_m50.galaxies
    ])


    # --------------------------------------------------------
    # Apply your original sample selection
    # --------------------------------------------------------

    M1500_m25 = M1500_m25[M1500_m25 < -16]

    M1500_m50 = M1500_m50[M1500_m50 < -17.5]


    # --------------------------------------------------------
    # Count galaxies in each magnitude bin
    # --------------------------------------------------------

    counts_total = []

    for mag_min, mag_max in mag_bins:

        # M25
        count_m25 = np.sum(
            (M1500_m25 >= mag_min) &
            (M1500_m25 < mag_max)
        )

        # M50
        count_m50 = np.sum(
            (M1500_m50 >= mag_min) &
            (M1500_m50 < mag_max)
        )

        # Total
        count_total = count_m25 + count_m50

        counts_total.append(count_total)


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    z_sim = obj_m25.simulation.redshift

    print(f"\nz = {z_sim:.2f}")

    for label, count in zip(bin_labels, counts_total):
        print(f"  {label:15s}: {count}")

    print(f"  {'Total':15s}: {sum(counts_total)}")
