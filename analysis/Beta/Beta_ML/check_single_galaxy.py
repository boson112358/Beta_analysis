import pandas as pd


# ============================================================
# Load catalogue
# ============================================================

filename = "simba_eor_beta_ml_catalogue.csv"

df = pd.read_csv(filename)


# ============================================================
# Select one galaxy
# ============================================================

galaxy_id = 100


gal = df[
    df["galaxy_id"] == galaxy_id
]


print("\n====================================")
print("Galaxy ID:", galaxy_id)
print("====================================\n")


print(gal)


# ============================================================
# Check physical quantities
# should be identical for all dust laws
# ============================================================

physical_columns = [
    "box",
    "redshift",
    "stellar_mass",
    "SFR",
    "sSFR",
    "metallicity_mass_weighted",
    "metallicity_stellar",
    "gas_mass",
    "dust_mass"
]


print("\n====================================")
print("Physical properties")
print("====================================")

for col in physical_columns:

    print("\n", col)

    print(
        gal[col].values
    )



# ============================================================
# Check dust-dependent quantities
# should change
# ============================================================

dust_columns = [
    "dust_law",
    "Av",
    "Muv_observed",
    "Muv_intrinsic",
    "beta"
]


print("\n====================================")
print("Dust dependent quantities")
print("====================================")


print(
    gal[dust_columns]
)



# ============================================================
# Check if physical quantities are identical
# ============================================================

print("\n====================================")
print("Consistency checks")
print("====================================")


for col in physical_columns:

    if col in ["box", "redshift"]:
        continue

    unique_values = gal[col].nunique()

    print(
        col,
        "unique values =",
        unique_values
    )
