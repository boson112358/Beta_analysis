import numpy as np
import caesar
import matplotlib.pyplot as plt
from utils.beta_utils import *


# ------------------------------------------------
# Dust laws
# ------------------------------------------------
dust_laws = ["calzetti", "lmc", "smc", "mw"]

colors = {
    "calzetti": "tab:blue",
    "lmc": "tab:orange",
    "smc": "tab:green",
    "mw": "tab:red",
}


# ------------------------------------------------
# Snapshots
# ------------------------------------------------
snapshots = ["016", "019", "022", "026", "030", "036"]


# ------------------------------------------------
# File templates
# ------------------------------------------------
template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)


bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])


# ------------------------------------------------
# Figure
# ------------------------------------------------
fig, axes = plt.subplots(
    3,
    2,
    figsize=(10, 12),
    sharex=True,
    sharey=True
)

axes = axes.flatten()



# =================================================
# Loop snapshots
# =================================================

for i, snap in enumerate(snapshots):

    ax = axes[i]


    for dust in dust_laws:


        # -----------------------------------------
        # Load CAESAR
        # -----------------------------------------
        f25 = template_m25.format(snap, dust)
        f50 = template_m50.format(snap, dust)


        obj = caesar.load(f25)
        obj_m50 = caesar.load(f50)


        Z = obj.simulation.redshift



        # -----------------------------------------
        # Calculate Av
        # -----------------------------------------

        Av_m25 = np.array([
            g.absmag["v"] - g.absmag_nodust["v"]
            for g in obj.galaxies
        ])


        Av_m50 = np.array([
            g.absmag["v"] - g.absmag_nodust["v"]
            for g in obj_m50.galaxies
        ])



        # -----------------------------------------
        # UV magnitude selection
        # -----------------------------------------

        mask_m25 = np.array([
            g.absmag["i1500"]
            for g in obj.galaxies
        ]) < -16


        mask_m50 = np.array([
            g.absmag["i1500"]
            for g in obj_m50.galaxies
        ]) < -17.5



        # -----------------------------------------
        # UV magnitudes
        # -----------------------------------------

        mags_m25 = np.array([
            [g.absmag[band] for g in obj.galaxies]
            for band in bands
        ])


        mags_m50 = np.array([
            [g.absmag[band] for g in obj_m50.galaxies]
            for band in bands
        ])


        mags_combined = np.concatenate(
            [
                mags_m25[:, mask_m25],
                mags_m50[:, mask_m50]
            ],
            axis=1
        )


        # -----------------------------------------
        # beta
        # -----------------------------------------

        beta = Calbeta(
            mags_combined,
            wavelengths
        )


        # -----------------------------------------
        # Combine Av
        # -----------------------------------------

        Av = np.concatenate(
            [
                Av_m25[mask_m25],
                Av_m50[mask_m50]
            ]
        )


        # -----------------------------------------
        # Remove Av <= 0
        # -----------------------------------------

        valid = Av > 0


        Av = Av[valid]
        beta = beta[valid]


        # log10 Av
        logAv = np.log10(Av)



        # -----------------------------------------
        # Bin
        # -----------------------------------------

        bin_centers, beta_mean, beta_std, count = bin_xy(
            x_values=logAv,
            y_values=beta,
            mask_values=None,
            mask_cut=None,
            N_bins=10
        )



        # -----------------------------------------
        # Plot
        # -----------------------------------------

        ax.errorbar(
            bin_centers,
            beta_mean,
            yerr=beta_std,
            color=colors[dust],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=5,
            capsize=2,
            label=dust if i == 0 else None
        )



    # =================================================
    # No dust case is removed
    # =================================================
    # because log10(Av=0) is undefined
    #
    # Instead it is not plotted.
    # =================================================



    # -----------------------------------------
    # Formatting
    # -----------------------------------------

    ax.set_xlim(-2, 1)

    ax.set_ylim(-3.0, -0.5)


    ax.text(
        -1.8,
        -0.8,
        f"z = {round(Z)}",
        fontsize=11
    )


    if i >= 4:
        ax.set_xlabel(
            r"$\log_{10}(A_V)$"
        )


    if i % 2 == 0:
        ax.set_ylabel(
            r"$\beta$"
        )



# ------------------------------------------------
# Legend
# ------------------------------------------------

axes[0].legend(
    frameon=False,
    fontsize=10
)


plt.tight_layout()


plt.savefig(
    "Beta_vs_logAv_DustComparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
