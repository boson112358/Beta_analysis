import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Load catalogue
# ============================================================

filename = "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_ML/simba_eor_beta_ml_catalogue.csv"

df = pd.read_csv(filename)

print("Catalogue columns:")
print(df.columns.tolist())


# ============================================================
# Snapshots to analyse
# ============================================================

snapshots = ["016", "019", "022", "026", "030", "036"]

# Convert snapshot to integer so that "016" and 16 match
df["snapshot_int"] = (
    df["snapshot"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(int)
)


# ============================================================
# Dust law
# ============================================================

dustlaw_select = "calzetti"


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
# Loop over snapshots
# ============================================================

for snapshot in snapshots:

    snapshot_int = int(snapshot)

    print("\n" + "=" * 70)
    print(f"Snapshot {snapshot}")
    print("=" * 70)

    # --------------------------------------------------------
    # Select snapshot and dust law
    # --------------------------------------------------------

    df_snap = df[
        (df["snapshot_int"] == snapshot_int) &
        (df["dust_law"] == dustlaw_select)
    ].copy()

    print("Selected galaxies:", len(df_snap))

    if len(df_snap) == 0:
        print("WARNING: No galaxies found. Skipping.")
        continue


    # --------------------------------------------------------
    # Determine redshift
    # --------------------------------------------------------

    redshifts = df_snap["redshift"].dropna().unique()

    if len(redshifts) == 0:
        print("WARNING: No redshift information found.")
        redshift = np.nan

    else:
        # Usually all galaxies in a snapshot have essentially
        # the same redshift
        redshift = np.median(redshifts)

    print(f"Redshift: z = {redshift:.3f}")


    # --------------------------------------------------------
    # Select variables
    # --------------------------------------------------------

    data = df_snap[variables].copy()


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    print("Number after cleaning:", len(data))


    if len(data) < 2:
        print("WARNING: Not enough galaxies for correlation.")
        continue


    # --------------------------------------------------------
    # Rename variables for plotting
    # --------------------------------------------------------

    data = data.rename(
        columns={
            "log_stellar_mass": "logMstar",
            "log_SFR": "logSFR",
            "metallicity_mass_weighted": "Z",
            "log_dust_mass": "logMdust"
        }
    )


    # --------------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------------

    corr = data.corr(method="pearson")

    print("\nPearson correlation:")
    print(corr)


    # --------------------------------------------------------
    # Plot correlation matrix
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        vmin=-1,
        vmax=1
    )

    plt.title(
        f"Pearson Correlation Matrix\n"
        f"Snapshot {snapshot}  |  z = {redshift:.2f}  |  {dustlaw_select}"
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    output_filename = (
        f"corr_matrix_snap{snapshot}_"
        f"z{redshift:.2f}_{dustlaw_select}.png"
    )

    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()
