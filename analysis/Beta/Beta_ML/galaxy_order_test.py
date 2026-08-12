import numpy as np
import caesar


# ============================================================
# Settings
# ============================================================

snap = "036"

template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)


dust_laws = [
    "calzetti",
    "lmc",
    "smc",
    "mw"
]


# ============================================================
# Load all dust catalogues
# ============================================================

catalogues = {}

for dust in dust_laws:

    filename = template_m25.format(
        snap,
        dust
    )

    print("Loading:", dust)

    catalogues[dust] = caesar.load(filename)



# ============================================================
# Compare number of galaxies
# ============================================================

print("\n========== Number of galaxies ==========")

for dust, obj in catalogues.items():

    print(
        dust,
        len(obj.galaxies)
    )



# ============================================================
# Compare GroupID ordering
# ============================================================

print("\n========== GroupID comparison ==========")


group_ids = {}

for dust, obj in catalogues.items():

    ids = np.array([
        g.GroupID
        for g in obj.galaxies
    ])

    group_ids[dust] = ids


reference = group_ids["calzetti"]


for dust in dust_laws:

    same = np.array_equal(
        reference,
        group_ids[dust]
    )

    print(
        "calzetti vs",
        dust,
        ":",
        same
    )



# ============================================================
# Compare stellar masses
# ============================================================

print("\n========== Stellar mass comparison ==========")


stellar_mass = {}

for dust, obj in catalogues.items():

    stellar_mass[dust] = np.array([
        g.masses["stellar"].value
        for g in obj.galaxies
    ])


for dust in dust_laws:

    diff = np.max(
        np.abs(
            stellar_mass["calzetti"]
            -
            stellar_mass[dust]
        )
    )

    print(
        "calzetti vs",
        dust,
        "max difference =",
        diff
    )



# ============================================================
# Compare positions
# ============================================================

print("\n========== Position comparison ==========")


positions = {}

for dust, obj in catalogues.items():

    positions[dust] = np.array([
        g.pos.value
        for g in obj.galaxies
    ])


for dust in dust_laws:

    diff = np.max(
        np.abs(
            positions["calzetti"]
            -
            positions[dust]
        )
    )

    print(
        "calzetti vs",
        dust,
        "max position difference =",
        diff
    )



# ============================================================
# Compare SFR
# ============================================================

print("\n========== SFR comparison ==========")


sfr = {}

for dust, obj in catalogues.items():

    sfr[dust] = np.array([
        g.sfr.to("Msun/yr").value
        for g in obj.galaxies
    ])


for dust in dust_laws:

    diff = np.max(
        np.abs(
            sfr["calzetti"]
            -
            sfr[dust]
        )
    )

    print(
        "calzetti vs",
        dust,
        "max SFR difference =",
        diff
    )
