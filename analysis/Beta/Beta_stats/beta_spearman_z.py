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
# Calculate Spearman correlations
# ============================================================

results = []


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
        # Redshift
        # ----------------------------------------------------

        redshift = np.median(
            df_snap["redshift"].dropna()
        )


        # ----------------------------------------------------
        # Select variables
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
        # Remove invalid values
        # ----------------------------------------------------

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna()


        # ----------------------------------------------------
        # Spearman correlations
        # ----------------------------------------------------

        correlations = {}

        for variable in variables:

            rho = data["beta"].corr(
                data[variable],
                method="spearman"
            )

            correlations[variable] = rho


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append({
            "dust_law": dust_law,
            "snapshot": snapshot,
            "redshift": redshift,
            "N": len(data),
            **correlations
        })


# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    ["dust_law", "redshift"],
    ascending=[True, False]
)


# ============================================================
# Print
# ============================================================

print("\n")
print("=" * 90)
print("Spearman correlations with beta")
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
    "beta_spearman_correlation_vs_redshift.csv",
    index=False
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


for ax, (variable, label) in zip(
    axes,
    variables.items()
):

    # --------------------------------------------------------
    # Plot each dust law
    # --------------------------------------------------------

    for dust_law in dust_laws:

        subset = results_df[
            (results_df["dust_law"] == dust_law)
        ].sort_values("redshift")


        ax.plot(
            subset["redshift"],
            subset[variable],
            marker="o",
            linewidth=2,
            markersize=6,
            label=dust_law.upper()
        )


    # --------------------------------------------------------
    # Zero correlation
    # --------------------------------------------------------

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_ylabel(
        r"$\rho_{\rm Spearman}(\beta,$ "
        + label +
        r"$)$",
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

# High redshift on left
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
# Title
# ============================================================

fig.suptitle(
    r"Spearman correlation of $\beta$ "
    r"with galaxy physical properties",
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
# Save
# ============================================================

plt.savefig(
    "beta_spearman_correlations_vs_redshift_all_dustlaws.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
