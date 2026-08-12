import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Load catalogue
# ============================================================

filename = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "Beta_analysis/analysis/Beta/Beta_ML/"
    "simba_eor_beta_ml_catalogue.csv"
)

df = pd.read_csv(filename)


# ============================================================
# Check available boxes
# ============================================================

print("\nAvailable boxes:")
print(df["box"].value_counts())


# ============================================================
# Snapshots
# ============================================================

snapshots = [
    "016",
    "019",
    "022",
    "026",
    "030",
    "036"
]

df["snapshot_int"] = (
    df["snapshot"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(int)
)


# ============================================================
# Dust laws
# ============================================================

dust_laws = [
    "calzetti",
    "lmc",
    "smc",
    "mw"
]


# ============================================================
# Boxes
#
# Currently combine m25 + m50.
#
# If you want only one box, change to:
#
# boxes_to_use = ["m25"]
#
# or
#
# boxes_to_use = ["m50"]
# ============================================================

boxes_to_use = [
    "m25",
    "m50"
]


# ============================================================
# Variables
#
# These are the X-axis quantities.
# Beta is ALWAYS the Y-axis.
# ============================================================

variables = {
    "log_stellar_mass": r"$\log(M_\star)$",
    "log_SFR": r"$\log(\mathrm{SFR})$",
    "metallicity_mass_weighted": r"$Z$",
    "log_dust_mass": r"$\log(M_{\rm dust})$",
    "Av": r"$A_V$"
}


# ============================================================
# Loop over dust laws
# ============================================================

for dust_law in dust_laws:

    print("\n")
    print("=" * 80)
    print(f"DUST LAW: {dust_law.upper()}")
    print("=" * 80)


    # ========================================================
    # Create figure
    #
    # 5 rows = galaxy properties
    # 6 columns = redshifts
    # ========================================================

    fig, axes = plt.subplots(
        nrows=5,
        ncols=6,
        figsize=(20, 14),
        squeeze=False
    )


    # ========================================================
    # Loop over galaxy properties
    # ========================================================

    for row, (variable, xlabel) in enumerate(
        variables.items()
    ):

        # ----------------------------------------------------
        # Loop over snapshots
        # ----------------------------------------------------

        for col, snapshot in enumerate(snapshots):

            ax = axes[row, col]

            snapshot_int = int(snapshot)


            # =================================================
            # Select snapshot + dust law + boxes
            # =================================================

            df_snap = df[
                (df["snapshot_int"] == snapshot_int) &
                (df["dust_law"] == dust_law) &
                (df["box"].isin(boxes_to_use))
            ].copy()


            # =================================================
            # No data
            # =================================================

            if len(df_snap) == 0:

                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes
                )

                continue


            # =================================================
            # Redshift
            # =================================================

            redshift_values = (
                df_snap["redshift"]
                .dropna()
                .values
            )

            if len(redshift_values) > 0:

                redshift = np.median(
                    redshift_values
                )

            else:

                redshift = np.nan


            # =================================================
            # X = galaxy property
            # Y = beta
            # =================================================

            x = df_snap[variable].to_numpy(
                dtype=float
            )

            y = df_snap["beta"].to_numpy(
                dtype=float
            )


            # =================================================
            # Remove invalid values
            # =================================================

            mask = (
                np.isfinite(x) &
                np.isfinite(y)
            )

            x = x[mask]
            y = y[mask]


            # =================================================
            # Need at least 3 galaxies
            # =================================================

            if len(x) < 3:

                ax.text(
                    0.5,
                    0.5,
                    "Too few data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes
                )

                continue


            # =================================================
            # Pearson
            # =================================================

            pearson_r, pearson_p = pearsonr(
                x,
                y
            )


            # =================================================
            # Spearman
            # =================================================

            spearman_rho, spearman_p = spearmanr(
                x,
                y
            )


            # =================================================
            # Scatter
            #
            # X = galaxy property
            # Y = beta
            # =================================================

            ax.scatter(
                x,
                y,
                s=5,
                alpha=0.20,
                rasterized=True
            )


            # =================================================
            # Linear regression
            # =================================================

            slope, intercept = np.polyfit(
                x,
                y,
                1
            )


            x_fit = np.linspace(
                np.min(x),
                np.max(x),
                100
            )

            y_fit = (
                slope * x_fit +
                intercept
            )


            ax.plot(
                x_fit,
                y_fit,
                linewidth=2
            )


            # =================================================
            # Correlation coefficients
            # =================================================

            ax.text(
                0.05,
                0.95,
                (
                    rf"$r_P={pearson_r:.2f}$" "\n"
                    rf"$\rho_S={spearman_rho:.2f}$"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9
            )


            # =================================================
            # Redshift title
            # =================================================

            if row == 0:

                ax.set_title(
                    rf"$z={redshift:.2f}$",
                    fontsize=12
                )


            # =================================================
            # X-axis label
            #
            # Each row gets its own property label.
            # =================================================

            ax.set_xlabel(
                xlabel,
                fontsize=11
            )


            # =================================================
            # Y-axis label
            #
            # Beta is the Y-axis.
            # =================================================

            if col == 0:

                ax.set_ylabel(
                    r"$\beta$",
                    fontsize=12
                )


            # =================================================
            # Grid
            # =================================================

            ax.grid(
                alpha=0.2
            )


            ax.tick_params(
                labelsize=9
            )


    # ========================================================
    # Figure title
    # ========================================================

    fig.suptitle(
        rf"$\beta$ versus galaxy properties — "
        rf"{dust_law.upper()} dust law",
        fontsize=17,
        y=0.995
    )


    # ========================================================
    # Layout
    # ========================================================

    plt.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.06,
        top=0.94,
        wspace=0.18,
        hspace=0.25
    )


    # ========================================================
    # Save
    # ========================================================
    box_label = "_".join(boxes_to_use)

    output_filename = (
        f"beta_scatter_{dust_law}_"
        f"{box_label}_all_redshifts.png"
    )

    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    print(
        f"Saved: {output_filename}"
    )
