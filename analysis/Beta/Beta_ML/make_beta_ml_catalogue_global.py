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
# Extract physical properties
# (independent of dust law)
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



    physical = {

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


    return physical




# ============================================================
# Extract dust-law dependent quantities
# ============================================================

def extract_dust_catalogue(
        obj,
        physical,
        box,
        snapshot,
        redshift,
        dust_law
):


    N = len(obj.galaxies)



    # --------------------------------------------------------
    # Observed UV magnitude and beta magnitudes
    # --------------------------------------------------------

    if dust_law == "nodust":


        Muv_observed = physical["Muv_intrinsic"].copy()


        Av = np.zeros(N)


        mags = np.array([
            [
                g.absmag_nodust[band]
                for g in obj.galaxies
            ]
            for band in bands
        ])


    else:


        Muv_observed = np.array([
            g.absmag["i1500"]
            for g in obj.galaxies
        ])


        Av = np.array([
            g.absmag["v"]
            -
            g.absmag_nodust["v"]
            for g in obj.galaxies
        ])


        mags = np.array([
            [
                g.absmag[band]
                for g in obj.galaxies
            ]
            for band in bands
        ])



    # --------------------------------------------------------
    # Calculate beta
    # --------------------------------------------------------

    beta = Calbeta(
        mags,
        wavelengths
    )



    # --------------------------------------------------------
    # Magnitude selection
    # --------------------------------------------------------

    mask = (
        Muv_observed
        <
        MUV_cut[box]
    )



    galaxy_ids = physical["galaxy_id"][mask]


    df = pd.DataFrame()



    # identifiers

    df["galaxy_id"] = galaxy_ids


    df["snapshot"] = snapshot


    df["box"] = box


    df["global_id"] = [
        f"{box}_{snapshot}_{gid}"
        for gid in galaxy_ids
    ]


    df["redshift"] = redshift


    df["dust_law"] = dust_law



    # --------------------------------------------------------
    # Physical quantities
    # --------------------------------------------------------

    for key, value in physical.items():


        if key == "galaxy_id":
            continue


        df[key] = value[mask]



    # --------------------------------------------------------
    # Dust dependent quantities
    # --------------------------------------------------------

    df["Muv_observed"] = (
        Muv_observed[mask]
    )


    df["Av"] = (
        Av[mask]
    )


    df["beta"] = (
        beta[mask]
    )



    return df




# ============================================================
# Main catalogue generation
# ============================================================

catalogues = []


for snapshot in snapshots:


    print("\n================================")
    print("Snapshot:", snapshot)
    print("================================")


    for box in ["m25", "m50"]:


        print("Box:", box)


        if box == "m25":

            template = template_m25

        else:

            template = template_m50



        # ----------------------------------------------------
        # Load calzetti as physical reference
        # ----------------------------------------------------

        physical_file = template.format(
            snapshot,
            "calzetti"
        )


        obj_reference = caesar.load(
            physical_file
        )


        redshift = (
            obj_reference.simulation.redshift
        )


        physical = extract_physical_properties(
            obj_reference
        )



        # ----------------------------------------------------
        # Loop dust laws
        # ----------------------------------------------------

        for dust in dust_laws:


            print(
                "   Dust law:",
                dust
            )


            if dust == "nodust":

                obj = obj_reference


            else:


                filename = template.format(
                    snapshot,
                    dust
                )


                obj = caesar.load(
                    filename
                )



            df = extract_dust_catalogue(
                obj,
                physical,
                box,
                snapshot,
                redshift,
                dust
            )


            catalogues.append(df)



# ============================================================
# Save
# ============================================================

catalogue = pd.concat(
    catalogues,
    ignore_index=True
)



# remove invalid entries

catalogue = catalogue.replace(
    [np.inf, -np.inf],
    np.nan
)


catalogue = catalogue.dropna()



print("\n================================")
print("Catalogue summary")
print("================================")


print(
    catalogue.shape
)


print(
    catalogue.groupby(
        ["box","snapshot","dust_law"]
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
