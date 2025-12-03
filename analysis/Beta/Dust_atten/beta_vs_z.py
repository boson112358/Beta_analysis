import numpy as np
import matplotlib.pyplot as plt
import caesar
from utils.beta_utils import Calbeta  # replace with your beta function
import matplotlib as mpl

# -------------------------------
# Plot style
# -------------------------------
mpl.rcParams.update({
    "figure.figsize": (8,5),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# -------------------------------
# Files, redshifts, dust laws
# -------------------------------
redshifts = ['016', '019', '022', '026', '030', '036']
dust_laws = ['calzetti', 'salmon', 'smc']
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

template_m25 = "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m25n1024/caesar_m25n1024_{}_{}.hdf5"
template_m50 = "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m50n1024/caesar_m50n1024_{}_{}.hdf5"

# Colors and linestyles for plotting
colors = ['#1f77b4', '#d62728', '#2ca02c']  # one per dust law
linestyles = ['-', '--', '-.']

# -------------------------------
# Storage for plotting
# -------------------------------
results = {law: {'z': [], 'beta_mean': [], 'beta_std': [],
                 'beta_mean_lum': [], 'beta_std_lum': [],
                 'beta_mean_mass': [], 'beta_std_mass': [],
                 'Ngal': []} for law in dust_laws}

# -------------------------------
# Loop over redshifts and dust laws
# -------------------------------
for z in redshifts:
    
    for j, law in enumerate(dust_laws):
        
        # Load Caesar files
        obj_m25 = caesar.load(template_m25.format(z, law))
        obj_m50 = caesar.load(template_m50.format(z, law))
        
        # Use the true redshift from the simulation (take either box, they should match)
        z_sim = obj_m25.simulation.redshift

        # Dust magnitudes
        mags_m25 = np.array([[g.absmag[band] for g in obj_m25.galaxies] for band in bands])
        mags_m50 = np.array([[g.absmag[band] for g in obj_m50.galaxies] for band in bands])
        
        # Stellar mass
        stellar_mass_m25 = np.array([g.masses['stellar'] for g in obj_m25.galaxies])
        stellar_mass_m50 = np.array([g.masses['stellar'] for g in obj_m50.galaxies])
        
        # Apply magnitude cuts
        mask_m25 = mags_m25[0] < -16
        mask_m50 = mags_m50[0] < -17.5
        
        mags_combined = np.concatenate([mags_m25[:, mask_m25], mags_m50[:, mask_m50]], axis=1)
        stellar_mass_combined = np.concatenate([stellar_mass_m25[mask_m25], stellar_mass_m50[mask_m50]])
        
        # Skip if no galaxies
        if mags_combined.shape[1] == 0:
            continue
        
        # Compute beta
        beta_combined = Calbeta(mags_combined, wavelengths)
        M1500_combined = mags_combined[0]
        
        # Mean and std
        beta_mean = np.mean(beta_combined)
        beta_std  = np.std(beta_combined)
        
        # Luminosity-weighted
        luminosity = 10**(-0.4 * M1500_combined)
        beta_mean_lum = np.average(beta_combined, weights=luminosity)
        beta_std_lum  = np.sqrt(np.average((beta_combined - beta_mean_lum)**2, weights=luminosity))
        
        # Stellar-mass-weighted
        beta_mean_mass = np.average(beta_combined, weights=stellar_mass_combined)
        beta_std_mass  = np.sqrt(np.average((beta_combined - beta_mean_mass)**2, weights=stellar_mass_combined))
        
        # Number of galaxies
        Ngal = mags_combined.shape[1]
        
        # Save results
        results[law]['z'].append(z_sim)
        results[law]['beta_mean'].append(beta_mean)
        results[law]['beta_std'].append(beta_std)
        results[law]['beta_mean_lum'].append(beta_mean_lum)
        results[law]['beta_std_lum'].append(beta_std_lum)
        results[law]['beta_mean_mass'].append(beta_mean_mass)
        results[law]['beta_std_mass'].append(beta_std_mass)
        results[law]['Ngal'].append(Ngal)

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(8,5))

for j, law in enumerate(dust_laws):
    zvals = results[law]['z']
    
    # Mean β with std
    plt.errorbar(zvals, results[law]['beta_mean_lum'], yerr=results[law]['beta_std_lum'],
                 color=colors[j], linestyle=linestyles[j], marker='o', capsize=4,
                 label=f"{law} (lum-weighted)")
    
    # Optional: plot stellar-mass weighted mean
    #plt.errorbar(zvals, results[law]['beta_mean_mass'], yerr=results[law]['beta_std_mass'],
    #             color=colors[j], linestyle=linestyles[j], marker='s', capsize=4,
    #             label=f"{law} (mass-weighted)")
    
    # Add number of galaxies above each point
    #for x, y, n in zip(zvals, results[law]['beta_mean_lum'], results[law]['Ngal']):
    #    plt.text(x, y+0.05, str(n), fontsize=8, ha='center', va='bottom')

plt.xlabel("Redshift")
plt.ylabel(r"$\beta$")
plt.title("β vs Redshift for Different Dust Laws")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Beta_vs_redshift_dust_laws.png", dpi=300)
plt.show()

