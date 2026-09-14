import caesar
import numpy as np

# ------------------------------------------------
# Load CAESAR file
# ------------------------------------------------

filename = "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m25n1024/caesar_m25n1024_016_calzetti.hdf5"

obj = caesar.load(filename)

print("=" * 60)
print("CAESAR OBJECT")
print("=" * 60)

print("Object type:", type(obj))
print("Number of galaxies:", len(obj.galaxies))
print("Number of halos:", len(obj.halos))


# ------------------------------------------------
# Inspect first galaxy
# ------------------------------------------------

gal = obj.galaxies[0]

print("\n" + "=" * 60)
print("FIRST GALAXY")
print("=" * 60)

print("Galaxy type:", type(gal))
print("\nGalaxy representation:")
print(gal)


# ------------------------------------------------
# List all attributes stored in the galaxy
# ------------------------------------------------

print("\n" + "=" * 60)
print("GALAXY ATTRIBUTES")
print("=" * 60)

print(gal.__dict__.keys())


# ------------------------------------------------
# Print attributes one by one
# ------------------------------------------------

print("\n" + "=" * 60)
print("GALAXY ATTRIBUTE VALUES")
print("=" * 60)

for key, value in gal.__dict__.items():
    print(f"\n--- {key} ---")
    print("Type:", type(value))
    print("Value:", value)


# ------------------------------------------------
# Inspect masses
# ------------------------------------------------

print("\n" + "=" * 60)
print("MASSES")
print("=" * 60)

print("gal.masses:")
print(gal.masses)

print("\ngal.mass:")
print(gal.mass)


# ------------------------------------------------
# Inspect metallicities
# ------------------------------------------------

print("\n" + "=" * 60)
print("METALLICITIES")
print("=" * 60)

print("gal.metallicities:")
print(gal.metallicities)


# ------------------------------------------------
# Check whether photometry exists
# ------------------------------------------------

print("\n" + "=" * 60)
print("PHOTOMETRY")
print("=" * 60)

if hasattr(gal, "photometry"):
    print("Photometry exists!")
    print("Type:", type(gal.photometry))
    print("Contents:", gal.photometry)

    if isinstance(gal.photometry, dict):
        print("\nPhotometry keys:")
        print(gal.photometry.keys())

        for key, value in gal.photometry.items():
            print(f"\n{key}:")
            print("  Type:", type(value))
            print("  Value:", value)

else:
    print("No 'photometry' attribute found.")


# ------------------------------------------------
# Check magnitudes
# ------------------------------------------------

print("\n" + "=" * 60)
print("MAGNITUDES")
print("=" * 60)

for attr in ["absmag", "absmag_nodust", "appmag", "appmag_nodust"]:

    if hasattr(gal, attr):

        value = getattr(gal, attr)

        print(f"\n--- {attr} ---")
        print("Type:", type(value))
        print("Value:", value)

        if isinstance(value, dict):
            print("Keys:", value.keys())


# ------------------------------------------------
# Check for beta explicitly
# ------------------------------------------------

print("\n" + "=" * 60)
print("BETA CHECK")
print("=" * 60)

if hasattr(gal, "photometry"):

    if "beta" in gal.photometry:
        print("beta =", gal.photometry["beta"])
    else:
        print("No 'beta' found in gal.photometry")

    if "beta_nodust" in gal.photometry:
        print("beta_nodust =", gal.photometry["beta_nodust"])
    else:
        print("No 'beta_nodust' found in gal.photometry")
