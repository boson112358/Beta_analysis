import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Load catalogue
# ============================================================

filename = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "Beta_analysis/analysis/Beta/Beta_ML/"
    "simba_eor_beta_ml_catalogue.csv"
)

df = pd.read_csv(filename)

print("Catalogue columns:")
print(df.columns.tolist())


# ============================================================
# Snapshots
# ============================================================

snapshots = ["016", "019", "022", "026", "030", "036"]


# Convert snapshot to integer
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
# Variables
# ============================================================

variables = [
    "beta",
    "log_stellar_mass",
    "log_SFR",
    "metallicity_mass_weighted",
    "log_dust_mass",
    "Av"
]


# ============================================================
# Loop over dust laws
# ============================================================

for dust_law in dust_laws:

    print("\n")
    print("=" * 80)
    print(f"        DUST LAW: {dust_law.upper()}")
    print("=" * 80)


    # --------------------------------------------------------
    # Create 2 x 3 figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes = axes.flatten()


    # --------------------------------------------------------
    # Loop over snapshots
    # --------------------------------------------------------

    for i, snapshot in enumerate(snapshots):

        snapshot_int = int(snapshot)

        ax = axes[i]


        # ----------------------------------------------------
        # Select snapshot + dust law
        # ----------------------------------------------------

        df_snap = df[
            (df["snapshot_int"] == snapshot_int) &
            (df["dust_law"] == dust_law)
        ].copy()


        print(
            f"\nDust law = {dust_law}, "
            f"snapshot = {snapshot}"
        )

        print(
            "Selected galaxies:",
            len(df_snap)
        )


        # ----------------------------------------------------
        # Check whether data exist
        # ----------------------------------------------------

        if len(df_snap) == 0:

            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            ax.set_title(
                f"Snapshot {snapshot}"
            )

            continue


        # ----------------------------------------------------
        # Determine redshift
        # ----------------------------------------------------

        redshifts = (
            df_snap["redshift"]
            .dropna()
            .unique()
        )

        if len(redshifts) > 0:

            redshift = np.median(redshifts)

        else:

            redshift = np.nan


        print(
            f"Redshift = z = {redshift:.3f}"
        )


        # ----------------------------------------------------
        # Select variables
        # ----------------------------------------------------

        data = df_snap[variables].copy()


        # ----------------------------------------------------
        # Remove invalid values
        # ----------------------------------------------------

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna()


        print(
            "Number after cleaning:",
            len(data)
        )


        # ----------------------------------------------------
        # Rename variables
        # ----------------------------------------------------

        data = data.rename(
            columns={
                "log_stellar_mass": "logMstar",
                "log_SFR": "logSFR",
                "metallicity_mass_weighted": "Z",
                "log_dust_mass": "logMdust"
            }
        )


        # ----------------------------------------------------
        # Pearson correlation
        # ----------------------------------------------------

        corr = data.corr(
            method="pearson"
        )


        # ----------------------------------------------------
        # Print correlation matrix
        # ----------------------------------------------------

        print("\nPearson correlation:")

        print(
            corr.to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )


        # ----------------------------------------------------
        # Plot heatmap
        # ----------------------------------------------------

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar=(i == 2 or i == 5),
            ax=ax,
            annot_kws={"size": 8}
        )


        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        ax.set_title(
            f"Snapshot {snapshot}\n"
            f"$z = {redshift:.2f}$",
            fontsize=11
        )


        # ----------------------------------------------------
        # Axis labels
        # ----------------------------------------------------

        ax.set_xlabel("")
        ax.set_ylabel("")


    # ========================================================
    # Figure title
    # ========================================================

    fig.suptitle(
        f"Pearson Correlation Matrix — {dust_law.upper()}",
        fontsize=16,
        y=1.02
    )


    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout()


    # ========================================================
    # Save figure
    # ========================================================

    output_filename = (
        f"correlation_matrix_"
        f"{dust_law}_all_redshifts.png"
    )

    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    print(
        f"\nSaved: {output_filename}"
    )
