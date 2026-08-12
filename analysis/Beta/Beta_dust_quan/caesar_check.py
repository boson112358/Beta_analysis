template_m25 = (
"/cosma8/data/dp376/dc-xian3/"
"simba-eor/EoRData/Dust_extin/"
"m25n1024/caesar_m25n1024_{}_{}.hdf5"
)

import caesar

# ------------------------------------------------
# Load your CAESAR file
# ------------------------------------------------
filename = "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m25n1024/caesar_m25n1024_016_calzetti.hdf5"

obj = caesar.load(filename)

print("Number of galaxies:", len(obj.galaxies))


# ------------------------------------------------
# Inspect first galaxy
# ------------------------------------------------
gal = obj.galaxies[0]

print("\n========== Galaxy attributes ==========")

attrs = dir(gal)

print(gal)
print(gal.__dict__.keys())
print(dir(gal))
print(gal.masses)
print(gal.mass)
print(gal.metallicities)

for a in attrs:
    if not a.startswith("_"):
        print(a)


# ------------------------------------------------
# Search for dust / attenuation / photometry
# ------------------------------------------------
keywords = [
    "av",
    "a_v",
    "dust",
    "atten",
    "extinc",
    "ebv",
    "phot",
    "lum",
    "mag",
    "flux"
]

print("\n========== Possible dust/photometry fields ==========")

for a in attrs:
    for key in keywords:
        if key.lower() in a.lower():
            print(a)
            break


# ------------------------------------------------
# Print values of possible fields
# ------------------------------------------------
print("\n========== Values ==========")

for a in attrs:
    for key in keywords:
        if key.lower() in a.lower():
            try:
                print(a, "=", getattr(gal, a))
            except Exception:
                print(a, "= <cannot print>")
            break
