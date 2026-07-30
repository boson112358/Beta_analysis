import numpy as np
import matplotlib.pyplot as plt
import caesar

from utils.beta_utils import Calbeta


# ------------------------------------------------
# Plot style
# ------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (8,8),
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3
})


# ------------------------------------------------
# Weighted percentile
# ------------------------------------------------
def weighted_percentile(values, weights, percentile):

    idx = np.argsort(values)

    values = values[idx]
    weights = weights[idx]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    return np.interp(
        percentile/100.,
        cdf,
        values
    )


# ------------------------------------------------
# Input
# ------------------------------------------------
redshifts = ['016','019','022','026','030','036']

dust_law = "calzetti"

bands = ["i1500","i2300","i2800"]

wavelengths = np.array(
    [1500,2300,2800]
)


template_m25 = (
"/cosma8/data/dp376/dc-xian3/"
"simba-eor/EoRData/Dust_extin/"
"m25n1024/caesar_m25n1024_{}_{}.hdf5"
)


template_m50 = (
"/cosma8/data/dp376/dc-xian3/"
"simba-eor/EoRData/Dust_extin/"
"m50n1024/caesar_m50n1024_{}_{}.hdf5"
)



# ------------------------------------------------
# Storage
# ------------------------------------------------
zvals=[]


beta_mean=[]
beta_median=[]
beta_mass=[]
beta_lum=[]

median_p16=[]
median_p84=[]



# ------------------------------------------------
# Loop snapshots
# ------------------------------------------------
for z_str in redshifts:


    obj25 = caesar.load(
        template_m25.format(z_str,dust_law)
    )

    obj50 = caesar.load(
        template_m50.format(z_str,dust_law)
    )


    z = obj25.simulation.redshift


    mags25 = np.array(
        [
        [g.absmag[b] for g in obj25.galaxies]
        for b in bands
        ]
    )


    mags50 = np.array(
        [
        [g.absmag[b] for g in obj50.galaxies]
        for b in bands
        ]
    )


    mass25=np.array(
        [g.masses["stellar"]
        for g in obj25.galaxies]
    )

    mass50=np.array(
        [g.masses["stellar"]
        for g in obj50.galaxies]
    )



    # magnitude selection
    mask25 = mags25[0] < -16
    mask50 = mags50[0] < -17.5



    mags=np.concatenate(
        [
        mags25[:,mask25],
        mags50[:,mask50]
        ],
        axis=1
    )


    mass=np.concatenate(
        [
        mass25[mask25],
        mass50[mask50]
        ]
    )


    if mags.shape[1]==0:
        continue



    # beta
    beta = Calbeta(
        mags,
        wavelengths
    )



    # luminosity weight
    L1500 = 10**(-0.4*mags[0])



    # ---------------------------------------------
    # central values
    # ---------------------------------------------

    mean=np.mean(beta)

    median=np.median(beta)

    mass_weighted=np.average(
        beta,
        weights=mass
    )

    lum_weighted=np.average(
        beta,
        weights=L1500
    )


    # median scatter
    p16=np.percentile(beta,16)
    p84=np.percentile(beta,84)



    # store
    zvals.append(z)

    beta_mean.append(mean)
    beta_median.append(median)
    beta_mass.append(mass_weighted)
    beta_lum.append(lum_weighted)

    median_p16.append(p16)
    median_p84.append(p84)



# arrays
zvals=np.array(zvals)

beta_mean=np.array(beta_mean)
beta_median=np.array(beta_median)
beta_mass=np.array(beta_mass)
beta_lum=np.array(beta_lum)

median_p16=np.array(median_p16)
median_p84=np.array(median_p84)



# ------------------------------------------------
# Plot settings
# ------------------------------------------------

colors={

"Median":"black",
"Mean":"tab:blue",
"Mass weighted":"tab:red",
"Luminosity weighted":"tab:green"

}


# =================================================
# Figure
# =================================================

fig,axs=plt.subplots(
    2,
    1,
    figsize=(8,9),
    sharex=True,
    gridspec_kw={"height_ratios":[2,1]}
)



# =================================================
# TOP PANEL
# =================================================


ax=axs[0]


# median scatter region

ax.fill_between(

    zvals,

    median_p16,

    median_p84,

    color="gray",

    alpha=0.25,

    label="Median 16-84 percentile"

)



# functions

methods={

"Median":beta_median,
"Mean":beta_mean,
"Mass weighted":beta_mass,
"Luminosity weighted":beta_lum

}


for name,value in methods.items():


    ax.plot(

        zvals,

        value,

        marker="o",

        linewidth=2,

        color=colors[name],

        label=name

    )


    # fit
    slope,intercept=np.polyfit(
        zvals,
        value,
        1
    )


    zfit=np.linspace(
        zvals.min(),
        zvals.max(),
        100
    )


    ax.plot(

        zfit,

        slope*zfit+intercept,

        "--",

        color=colors[name],

        alpha=0.7,

        label=f"{name} slope={slope:.3f}"

    )



ax.set_ylabel(r"$\beta$")

ax.legend(
    ncol=2,
    fontsize=8
)


ax.set_title(
    r"UV slope evolution with Calzetti dust"
)



# =================================================
# BOTTOM PANEL
# =================================================


ax=axs[1]


for name in [
    "Mean",
    "Mass weighted",
    "Luminosity weighted"
]:


    if name=="Mean":
        diff=beta_mean-beta_median

    elif name=="Mass weighted":
        diff=beta_mass-beta_median

    elif name=="Luminosity weighted":
        diff=beta_lum-beta_median



    ax.plot(

        zvals,

        diff,

        marker="o",

        linewidth=2,

        color=colors[name],

        label=name

    )



ax.axhline(
    0,
    linestyle="--",
    color="black",
    linewidth=1
)


ax.set_xlabel("Redshift")

ax.set_ylabel(
    r"$\Delta\beta$"
    "\n"
    r"($\beta-\beta_{\rm median}$)"
)


ax.legend(
    fontsize=9
)


plt.tight_layout()


plt.savefig(
    "Beta_z_calzetti_main_difference.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
