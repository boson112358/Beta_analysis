import numpy as np
import matplotlib.pyplot as plt
import caesar
from utils.beta_utils import Calbeta  # replace with your beta function

# -------------------------------
# Plot style
# -------------------------------
plt.rcParams.update({
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
colors = ['#1f77b4', '#d62728', '#2ca02c']
linestyles = ['-', '--', '-.']

# -------------------------------
# Storage for results
# -------------------------------
results = {law: {'z': [], 'beta_mean_mass': [], 'beta_std_mass': [], 'Ngal': []} for law in dust_laws}

# -------------------------------
# Loop over redshifts and dust laws
# -------------------------------
for z_str in redshifts:
    for j, law in enumerate(dust_laws):
        
        # Load Caesar files
        obj_m25 = caesar.load(template_m25.format(z_str, law))
        obj_m50 = caesar.load(template_m50.format(z_str, law))
        
        # True redshift
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
        
        # Mass-weighted mean
        beta_mean_mass = np.average(beta_combined, weights=stellar_mass_combined)
        beta_std_mass  = np.sqrt(np.average((beta_combined - beta_mean_mass)**2, weights=stellar_mass_combined))
        
        Ngal = mags_combined.shape[1]
        
        # Store results
        results[law]['z'].append(z_sim)
        results[law]['beta_mean_mass'].append(beta_mean_mass)
        results[law]['beta_std_mass'].append(beta_std_mass)
        results[law]['Ngal'].append(Ngal)

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(8,5))

for j, law in enumerate(dust_laws):
    zvals = np.array(results[law]['z'])
    beta_vals = np.array(results[law]['beta_mean_mass'])
    beta_errs = np.array(results[law]['beta_std_mass'])
    
    # Errorbar plot
    plt.errorbar(zvals, beta_vals, yerr=beta_errs, fmt='o', color=colors[j],
                 linestyle='-', capsize=4, label=f"{law} mass-weighted")
    
    # Linear fit with slope error via Monte Carlo
    weights = 1.0 / beta_errs**2
    slope, intercept = np.polyfit(zvals, beta_vals, 1, w=weights)
    
    # Monte Carlo for slope uncertainty
    Nmc = 1000
    slopes_mc = []
    for i in range(Nmc):
        beta_sample = beta_vals + np.random.normal(0, beta_errs)
        slope_i, _ = np.polyfit(zvals, beta_sample, 1)
        slopes_mc.append(slope_i)
    slope_err = np.std(slopes_mc)
    
    # Plot linear fit
    z_fit = np.linspace(min(zvals), max(zvals), 50)
    plt.plot(z_fit, slope*z_fit + intercept, '-', color=colors[j], label=f"{law} fit (slope={slope:.2f}±{slope_err:.2f})")
    
plt.xlabel("Redshift")
plt.ylabel(r"$\beta$ (stellar-mass weighted)")
plt.title("Mass-weighted β vs Redshift with Linear Fits")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Beta_massweighted_vs_redshift_fit.png", dpi=300)

