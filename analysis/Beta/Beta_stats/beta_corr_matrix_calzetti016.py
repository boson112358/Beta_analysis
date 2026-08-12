import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------
# Load catalogue
# ------------------------------------------------

filename = "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_ML/simba_eor_beta_ml_catalogue.csv"

df = pd.read_csv(filename)

print(df.columns)


# ------------------------------------------------
# Select one redshift and dust law
# ------------------------------------------------

snapshot_select = 16
dustlaw_select = "calzetti"


df = df[
    (df["snapshot"] == snapshot_select) &
    (df["dust_law"] == dustlaw_select)
].copy()


print("\nSelected galaxies:", len(df))


# ------------------------------------------------
# Variables
# ------------------------------------------------

variables = [
    "beta",
    "log_stellar_mass",
    "log_SFR",
    "metallicity_mass_weighted",
    "log_dust_mass",
    "Av"
]


data = df[variables].copy()


# remove invalid values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()


print("\nNumber after cleaning:", len(data))


# ------------------------------------------------
# Rename for plotting
# ------------------------------------------------

data = data.rename(
    columns={
        "log_stellar_mass": "logMstar",
        "log_SFR": "logSFR",
        "metallicity_mass_weighted": "Z",
        "log_dust_mass": "logMdust"
    }
)


# ------------------------------------------------
# Pearson correlation
# ------------------------------------------------

corr = data.corr(method="pearson")

print("\nPearson correlation:")
print(corr)


plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title(
    f"Correlation matrix\nsnapshot={snapshot_select}, dust={dustlaw_select}"
)

plt.tight_layout()

plt.savefig(
    f"corr_matrix_snap{snapshot_select}_{dustlaw_select}.png",
    dpi=300
)

plt.show()
