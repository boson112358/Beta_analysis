import numpy as np
import caesar
import unyt
import matplotlib.pyplot as plt
from utils.beta_utils import *

# Get the input caesar file
infile = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/caesar_m25n1024_036.hdf5'
infile_m50 = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/caesar_m50n1024_036.hdf5'

# Load caesar file
obj = caesar.load(infile)
obj_m50 = caesar.load(infile_m50)

# Number of galaxies
Ngal = obj.ngalaxies

# Redshift
Z = obj.simulation.redshift

# Print the info
print("Number of galaxies: ", Ngal)
print("Redshift: ", Z)
print("magnitude:", obj.galaxies[0].absmag["i1500"])

# Get UV magnitude
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# Define the four cases
cases = [
    ("m25 dust", obj, False, -16),
    ("m50 dust", obj_m50, False, -17.5),
    ("m25 no-dust", obj, True, -16),
    ("m50 no-dust", obj_m50, True, -17.5)
]

# Dictionary to store results
results = {}

for label, obj_case, nodust, mag_cut in cases:

    stellar_mass = get_stellar_mass(obj_case)

    bin_centers, beta_mean, beta_std, bin_count = get_binned_xy(obj_case, bands, wavelengths,
                                                       x_values=stellar_mass, N_bins=10, mag_cut=mag_cut,
                                                       nodust=nodust, log_x=True)
    results[label] = {
        "bin_centers": bin_centers,
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "stellar_mass": stellar_mass
    }

plt.figure(figsize=(6,4))

# Define colors and line styles
plot_styles = {
    "m25 dust":       {"color": "blue", "linestyle": "-", "yerr": True},
    "m50 dust":       {"color": "red",  "linestyle": "-", "yerr": True},
    "m25 no-dust":    {"color": "blue", "linestyle": "--", "yerr": False},
    "m50 no-dust":    {"color": "red",  "linestyle": "--", "yerr": False},
}

for label, data in results.items():
    style = plot_styles[label]
    x = data["bin_centers"]      # now stellar mass
    y = data["beta_mean"]
    yerr = data["beta_std"] if style["yerr"] else None

    if style["yerr"]:
        plt.errorbar(x, y, yerr=yerr, color=style["color"],
                     linestyle=style["linestyle"], label=label)
    else:
        plt.plot(x, y, color=style["color"], linestyle=style["linestyle"], label=label)

plt.xlabel(r"$log(M_*)$ [$M_\odot$]")    # <-- updated
plt.ylabel(r"$\beta$")
plt.xlim(6, 11)
plt.ylim(-2.6, -0.8)

plt.text(7, -1, f"z = {round(Z)}", fontsize=12, color="black")  # adjust for log scale
plt.legend()
plt.tight_layout()
plt.savefig("Beta_vs_StellarMass.png")
   
