import numpy as np
import caesar
import unyt
import matplotlib.pyplot as plt
from beta_utils import Calbeta, bin_beta, get_binned_beta

# File lists for each box (sorted by redshift)
files_25 = sorted([
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/' + f
    for f in ['caesar_m25n1024_016.hdf5', 'caesar_m25n1024_019.hdf5',
              'caesar_m25n1024_022.hdf5', 'caesar_m25n1024_026.hdf5',
              'caesar_m25n1024_030.hdf5', 'caesar_m25n1024_036.hdf5']
])

files_50 = sorted([
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/' + f
    for f in ['caesar_m50n1024_016.hdf5', 'caesar_m50n1024_019.hdf5',
              'caesar_m50n1024_022.hdf5', 'caesar_m50n1024_026.hdf5',
              'caesar_m50n1024_030.hdf5', 'caesar_m50n1024_036.hdf5']
])

boxes = {'25': files_25, '50': files_50}
colors = {'25': 'blue', '50': 'red'}

# Parameters
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])
N_bins = 6

# Create 2x3 subplots (6 snapshots)
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.flatten()

# Loop over snapshots by index
for idx in range(len(files_25)):
    ax = axes[idx]
    
    results = {}
    
    # Loop over boxes
    for box, files in boxes.items():
        f = files[idx]  # same redshift index
        obj = caesar.load(f)
        z = obj.simulation.redshift
        
        # Four cases: dust/no-dust
        cases = [
            ("dust", False, -16 if box=='25' else -17.5),
            ("no-dust", True, -16 if box=='25' else -17.5)
        ]
        
        for label_case, nodust, mag_cut in cases:
            # Compute binned beta
            mags = np.array([[g.absmag_nodust[band] if nodust else g.absmag[band] 
                              for g in obj.galaxies] for band in bands])
            beta = Calbeta(mags, wavelengths)
            M1500 = mags[0]
            bin_centers, beta_mean, beta_std = bin_beta(M1500, beta, N_bins=N_bins, mag_cut=mag_cut)
            
            results[f"{box} {label_case}"] = {
                "bin_centers": bin_centers,
                "beta_mean": beta_mean,
                "beta_std": beta_std
            }
    
    # Define plot styles
    plot_styles = {
        "25 dust":       {"color": "blue", "linestyle": "-", "yerr": True},
        "50 dust":       {"color": "red",  "linestyle": "-", "yerr": True},
        "25 no-dust":    {"color": "blue", "linestyle": "--", "yerr": False},
        "50 no-dust":    {"color": "red",  "linestyle": "--", "yerr": False},
    }
    
    # Plot each curve
    for label, data in results.items():
        box_label = label.split()[0]
        style_key = f"{box_label} {'no-dust' if 'no-dust' in label else 'dust'}"
        style = plot_styles[style_key]
        
        x = data["bin_centers"]
        y = data["beta_mean"]
        yerr = data["beta_std"] if style["yerr"] else None
        
        if style["yerr"]:
            ax.errorbar(x, y, yerr=yerr, color=style["color"], linestyle=style["linestyle"], label=label)
        else:
            ax.plot(x, y, color=style["color"], linestyle=style["linestyle"], label=label)
    
    ax.set_xlabel("M1500")
    ax.set_ylabel("Beta")
    ax.set_xlim(-23, -15)
    ax.set_ylim(-2.6, -1.4)
    ax.set_title(f"z = {round(z)}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("Beta_all_redshifts.png")
 
