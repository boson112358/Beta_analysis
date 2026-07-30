import numpy as np
import matplotlib.pyplot as plt
import caesar

from utils.beta_utils import Calbeta


# ==================================================
# Plot style
# ==================================================

plt.rcParams.update({

    "figure.figsize": (9,6),
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3

})


# ==================================================
# Input
# ==================================================

redshifts = [
    '016','019','022',
    '026','030','036'
]


dust_laws = [
    "calzetti",
    "smc",
    "lmc",
    "mw",
    "nodust"
]


bands = [
    "i1500",
    "i2300",
    "i2800"
]


wavelengths = np.array(
    [1500,2300,2800]
)



template_m25 = (
"/cosma8/data/dp376/dc-xian3/"
"simba-eor/EoRData/Dust_extin/"
"m25n1024/caesar_m25n1024_{}_{}.hdf5"
)


template_m50 = (
"/cosma8/data/dp376/dc-xian3/"
"simba-eor/EoRData/Dust_extin/"
"m50n1024/caesar_m50n1024_{}_{}.hdf5"
)



# ==================================================
# Storage
# ==================================================

results = {

    law:
    {
        "z":[],
        "median":[],
        "p16":[],
        "p84":[]

    }

    for law in dust_laws

}



# ==================================================
# Loop redshift
# ==================================================

for snap in redshifts:


    print("Processing snapshot:", snap)



    # --------------------------------------------------
    # Load NO-DUST selection files
    # --------------------------------------------------

    obj25_nodust = caesar.load(
        template_m25.format(
            snap,
            "calzetti"
        )
    )


    obj50_nodust = caesar.load(
        template_m50.format(
            snap,
            "calzetti"
        )
    )


    z = obj25_nodust.simulation.redshift



    # --------------------------------------------------
    # Define galaxy selection using intrinsic UV
    # --------------------------------------------------

    mags25_nodust = np.array(

        [
        [g.absmag_nodust[b]
        for g in obj25_nodust.galaxies]

        for b in bands

        ]

    )


    mags50_nodust = np.array(

        [
        [g.absmag_nodust[b]
        for g in obj50_nodust.galaxies]

        for b in bands

        ]

    )


    mask25 = mags25_nodust[0] < -16

    mask50 = mags50_nodust[0] < -17.5



    # ==================================================
    # Loop dust models
    # ==================================================

    for law in dust_laws:


        print("  ", law)



        # ------------------------------------------------
        # Load corresponding dust file
        # ------------------------------------------------

        if law == "nodust":

            obj25 = obj25_nodust
            obj50 = obj50_nodust


            mags25 = mags25_nodust

            mags50 = mags50_nodust



        else:


            obj25 = caesar.load(
                template_m25.format(
                    snap,
                    law
                )
            )


            obj50 = caesar.load(
                template_m50.format(
                    snap,
                    law
                )
            )



            mags25 = np.array(

                [
                [g.absmag[b]
                for g in obj25.galaxies]

                for b in bands

                ]

            )


            mags50 = np.array(

                [
                [g.absmag[b]
                for g in obj50.galaxies]

                for b in bands

                ]

            )



        # ------------------------------------------------
        # Apply identical intrinsic selection
        # ------------------------------------------------

        mags_combined = np.concatenate(

            [

            mags25[:,mask25],

            mags50[:,mask50]

            ],

            axis=1

        )



        if mags_combined.shape[1] == 0:
            continue



        # ------------------------------------------------
        # beta
        # ------------------------------------------------

        beta = Calbeta(

            mags_combined,

            wavelengths

        )


        median = np.median(beta)


        p16 = np.percentile(
            beta,
            16
        )


        p84 = np.percentile(
            beta,
            84
        )



        results[law]["z"].append(z)

        results[law]["median"].append(median)

        results[law]["p16"].append(p16)

        results[law]["p84"].append(p84)



# ==================================================
# Plot
# ==================================================

colors = {

    "calzetti":"tab:blue",
    "smc":"tab:red",
    "lmc":"tab:green",
    "mw":"tab:orange",
    "nodust":"black"

}


labels = {

    "calzetti":"Calzetti",
    "smc":"SMC",
    "lmc":"LMC",
    "mw":"MW",
    "nodust":"No dust"

}



plt.figure(figsize=(9,6))



for law in dust_laws:


    z = np.array(
        results[law]["z"]
    )

    median = np.array(
        results[law]["median"]
    )

    p16 = np.array(
        results[law]["p16"]
    )

    p84 = np.array(
        results[law]["p84"]
    )


    color = colors[law]


    # ------------------------------------------
    # scatter region
    # ------------------------------------------

    plt.fill_between(

        z,

        p16,

        p84,

        alpha=0.15,

        color=color

    )


    # ------------------------------------------
    # median
    # ------------------------------------------

    plt.plot(

        z,

        median,

        marker="o",

        linewidth=2,

        color=color,

        label=labels[law]

    )



    # ------------------------------------------
    # linear fit
    # ------------------------------------------

    slope, intercept = np.polyfit(

        z,

        median,

        1

    )


    zfit=np.linspace(

        z.min(),

        z.max(),

        100

    )


    plt.plot(

        zfit,

        slope*zfit+intercept,

        "--",

        color=color,

        alpha=0.7,

        label=f"{labels[law]} slope={slope:.3f}"

    )



plt.xlabel(
    "Redshift"
)


plt.ylabel(
    r"Median $\beta$"
)


plt.title(
    r"$\beta-z$ relation: different dust attenuation laws"
)


plt.legend(
    ncol=2
)


plt.tight_layout()


plt.savefig(
    "beta_z_dustlaws_fixed_selection_correct.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
