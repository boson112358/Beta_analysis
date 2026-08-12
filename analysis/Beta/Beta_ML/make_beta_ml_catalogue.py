import numpy as np
import pandas as pd
import caesar

from utils.beta_utils import Calbeta


# ============================================================
# Configuration
# ============================================================

dust_laws = [
    "calzetti",
    "lmc",
    "smc",
    "mw",
    "nodust"
]


snapshots = [
    "016",
    "019",
    "022",
    "026",
    "030",
    "036"
]


template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)


bands = [
    "i1500",
    "i2300",
    "i2800"
]

wavelengths = np.array(
    [1500, 2300, 2800]
)


MUV_cut = {
    "m25": -16,
    "m50": -17.5
}


output_file = (
    "simba_eor_beta_ml_catalogue.csv"
)



# ============================================================
# Extract physical quantities
# (same for all dust laws)
# ============================================================

def extract_physical_properties(obj):

    N = len(obj.galaxies)


    galaxy_id = np.arange(N)


    stellar_mass = np.array([
        g.masses["stellar"].to("Msun").value
        for g in obj.galaxies
    ])


    sfr = np.array([
        g.sfr.to("Msun/yr").value
        for g in obj.galaxies
    ])


    ssfr = np.array([
        (g.sfr / g.masses["stellar"])
        .to("1/yr")
        .value
        for g in obj.galaxies
    ])


    gas_mass = np.array([
        g.masses["gas"].to("Msun").value
        for g in obj.galaxies
    ])


    dust_mass = np.array([
        g.masses["dust"].to("Msun").value
        for g in obj.galaxies
    ])



    metallicity_mass_weighted = np.array([
        g.metallicities["mass_weighted"].value
        for g in obj.galaxies
    ])


    metallicity_stellar = np.array([
        g.metallicities["stellar"].value
        for g in obj.galaxies
    ])



    Muv_intrinsic = np.array([
        g.absmag_nodust["i1500"]
        for g in obj.galaxies
    ])


    return {

        "galaxy_id": galaxy_id,

        "stellar_mass": stellar_mass,
        "log_stellar_mass": np.log10(stellar_mass),

        "SFR": sfr,
        "log_SFR": np.log10(sfr),

        "sSFR": ssfr,
        "log_sSFR": np.log10(ssfr),

        "gas_mass": gas_mass,
        "log_gas_mass": np.log10(gas_mass),

        "dust_mass": dust_mass,
        "log_dust_mass": np.log10(dust_mass),

        "metallicity_mass_weighted":
            metallicity_mass_weighted,

        "metallicity_stellar":
            metallicity_stellar,

        "Muv_intrinsic":
            Muv_intrinsic
    }



# ============================================================
# Extract one dust law
# ============================================================

def extract_dust_properties(
        obj,
        physical,
        box,
        redshift,
        dust_law
):


    N = len(obj.galaxies)


    # -----------------------------
    # observed UV magnitude
    # -----------------------------

    Muv_observed = np.array([
        g.absmag["i1500"]
        for g in obj.galaxies
    ])



    # -----------------------------
    # Dust attenuation
    # -----------------------------

    if dust_law == "nodust":

        Av = np.zeros(N)

        beta_mags = np.array([
            [
                g.absmag_nodust[band]
                for g in obj.galaxies
            ]
            for band in bands
        ])

        Muv_observed = physical["Muv_intrinsic"]


    else:

        Av = np.array([
            g.absmag["v"]
            -
            g.absmag_nodust["v"]
            for g in obj.galaxies
        ])


        beta_mags = np.array([
            [
                g.absmag[band]
                for g in obj.galaxies
            ]
            for band in bands
        ])



    # -----------------------------
    # beta
    # -----------------------------

    beta = Calbeta(
        beta_mags,
        wavelengths
    )



    # -----------------------------
    # UV selection
    # -----------------------------

    mask = (
        Muv_observed
        <
        MUV_cut[box]
    )



    data = {}


    # identifiers

    data["galaxy_id"] = (
        physical["galaxy_id"][mask]
    )

    data["box"] = np.repeat(
        box,
        np.sum(mask)
    )

    data["redshift"] = np.repeat(
        redshift,
        np.sum(mask)
    )

    data["dust_law"] = np.repeat(
        dust_law,
        np.sum(mask)
    )


    # physical properties

    for key, value in physical.items():

        if key not in [
            "galaxy_id",
            "Muv_intrinsic"
        ]:

            data[key] = value[mask]



    # UV/dust quantities

    data["Muv_intrinsic"] = (
        physical["Muv_intrinsic"][mask]
    )

    data["Muv_observed"] = (
        Muv_observed[mask]
    )

    data["Av"] = (
        Av[mask]
    )

    data["beta"] = (
        beta[mask]
    )


    df = pd.DataFrame(data)


    return df



# ============================================================
# Main loop
# ============================================================


all_catalogues = []


for snap in snapshots:


    print("\n====================")
    print("Snapshot:", snap)
    print("====================")


    for box in ["m25", "m50"]:


        print("Processing:", box)


        if box == "m25":

            template = template_m25

        else:

            template = template_m50



        # ---------------------------------
        # Load Calzetti as physical reference
        # ---------------------------------

        physical_file = template.format(
            snap,
            "calzetti"
        )

        obj_physical = caesar.load(
            physical_file
        )


        redshift = (
            obj_physical.simulation.redshift
        )


        physical = extract_physical_properties(
            obj_physical
        )



        # ---------------------------------
        # Dust laws
        # ---------------------------------

        for dust in dust_laws:


            print(
                "   dust:",
                dust
            )


            if dust == "nodust":

                obj = obj_physical

            else:

                filename = template.format(
                    snap,
                    dust
                )

                obj = caesar.load(
                    filename
                )



            df = extract_dust_properties(
                obj,
                physical,
                box,
                redshift,
                dust
            )


            all_catalogues.append(df)



# ============================================================
# Combine and save
# ============================================================


catalogue = pd.concat(
    all_catalogues,
    ignore_index=True
)


# remove bad values

catalogue = catalogue.replace(
    [np.inf, -np.inf],
    np.nan
)

catalogue = catalogue.dropna()


print("\n================================")
print("Final catalogue")
print("================================")

print(
    catalogue.shape
)

print(
    catalogue.groupby(
        ["box","dust_law"]
    ).size()
)


catalogue.to_csv(
    output_file,
    index=False
)


print(
    "\nSaved:",
    output_file
)
