import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

variables = {
    "log_stellar_mass": r"$\log(M_\star)$",
    "log_SFR": r"$\log(\mathrm{SFR})$",
    "metallicity_mass_weighted": r"$Z$",
    "log_dust_mass": r"$\log(M_{\rm dust})$",
    "Av": r"$A_V$"
}


# ============================================================
# Store results
# ============================================================

results = []


# ============================================================
# Loop over dust laws and snapshots
# ============================================================

for dust_law in dust_laws:

    for snapshot in snapshots:

        snapshot_int = int(snapshot)

        # ----------------------------------------------------
        # Select snapshot and dust law
        # ----------------------------------------------------

        df_snap = df[
            (df["snapshot_int"] == snapshot_int) &
            (df["dust_law"] == dust_law)
        ].copy()


        if len(df_snap) == 0:

            print(
                f"WARNING: No data for "
                f"{dust_law}, snapshot {snapshot}"
            )

            continue


        # ----------------------------------------------------
        # Determine redshift
        # ----------------------------------------------------

        redshift = np.median(
            df_snap["redshift"].dropna()
        )


        # ----------------------------------------------------
        # Select beta + physical variables
        # ----------------------------------------------------

        cols = [
            "beta",
            "log_stellar_mass",
            "log_SFR",
            "metallicity_mass_weighted",
            "log_dust_mass",
            "Av"
        ]

        data = df_snap[cols].copy()


        # ----------------------------------------------------
        # Remove inf / NaN
        # ----------------------------------------------------

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna()


        # ----------------------------------------------------
        # Calculate Pearson correlation with beta
        # ----------------------------------------------------

        correlations = {}

        for variable in variables:

            rho = data["beta"].corr(
                data[variable],
                method="pearson"
            )

            correlations[variable] = rho


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({
            "dust_law": dust_law,
            "snapshot": snapshot,
            "redshift": redshift,
            "N": len(data),
            **correlations
        })


# ============================================================
# Convert results to DataFrame
# ============================================================

results_df = pd.DataFrame(results)


# Sort by dust law and redshift
results_df = results_df.sort_values(
    ["dust_law", "redshift"],
    ascending=[True, False]
)


# ============================================================
# Print results
# ============================================================

print("\n")
print("=" * 90)
print("Pearson correlations with beta")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# Save numerical results
# ============================================================

results_df.to_csv(
    "beta_correlation_vs_redshift.csv",
    index=False
)

print(
    "\nSaved numerical results to:"
    " beta_correlation_vs_redshift.csv"
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    5,
    1,
    figsize=(9, 18),
    sharex=True
)


# ============================================================
# Plot each variable
# ============================================================

for ax, (variable, label) in zip(
    axes,
    variables.items()
):

    # --------------------------------------------------------
    # Plot each dust law
    # --------------------------------------------------------

    for dust_law in dust_laws:

        subset = results_df[
            results_df["dust_law"] == dust_law
        ].sort_values("redshift", ascending=False)


        ax.plot(
            subset["redshift"],
            subset[variable],
            marker="o",
            linewidth=2,
            markersize=6,
            label=dust_law.upper()
        )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_ylabel(
        r"$\rho(\beta,$ " + label + r"$)$",
        fontsize=12
    )

    ax.set_ylim(
        -1,
        1
    )

    ax.grid(
        alpha=0.25
    )


# ============================================================
# X-axis
# ============================================================

axes[-1].set_xlabel(
    "Redshift",
    fontsize=13
)


# ============================================================
# Reverse x-axis
#
# High redshift on the left:
# z ~ 11 -> z ~ 6
# ============================================================

axes[-1].invert_xaxis()


# ============================================================
# Legend
# ============================================================

axes[0].legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.20),
    frameon=False
)


# ============================================================
# Figure title
# ============================================================

fig.suptitle(
    r"Pearson correlation of $\beta$ with galaxy physical properties",
    fontsize=16,
    y=0.995
)


# ============================================================
# Layout
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)


# ============================================================
# Save figure
# ============================================================

plt.savefig(
    "beta_correlations_vs_redshift_all_dustlaws.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
