import numpy as np
import matplotlib.pyplot as plt
import caesar

from utils.beta_utils import Calbeta


# ------------------------------------------------
# Plot style
# ------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (9,6),
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "axes.grid": True,
})


# ------------------------------------------------
# weighted percentile function
# ------------------------------------------------
def weighted_percentile(values, weights, percentile):

    sorter = np.argsort(values)

    values = values[sorter]
    weights = weights[sorter]

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    return np.interp(
        percentile/100.,
        cumulative,
        values
    )


# ------------------------------------------------
# files
# ------------------------------------------------
redshifts = ['016', '019', '022', '026', '030', '036']

dust_law = "calzetti"

bands = ["i1500", "i2300", "i2800"]

wavelengths = np.array([1500,2300,2800])


template_m25 = "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m25n1024/caesar_m25n1024_{}_{}.hdf5"

template_m50 = "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m50n1024/caesar_m50n1024_{}_{}.hdf5"


# ------------------------------------------------
# storage
# ------------------------------------------------
results = {

    "Mean":{
        "value":[],
        "lower":[],
        "upper":[]
    },

    "Median":{
        "value":[],
        "lower":[],
        "upper":[]
    },

    "Luminosity weighted":{
        "value":[],
        "lower":[],
        "upper":[]
    },

    "Mass weighted":{
        "value":[],
        "lower":[],
        "upper":[]
    }

}


zvals = []


# ------------------------------------------------
# loop over snapshots
# ------------------------------------------------
for z_str in redshifts:


    obj_m25 = caesar.load(
        template_m25.format(z_str,dust_law)
    )

    obj_m50 = caesar.load(
        template_m50.format(z_str,dust_law)
    )


    z_sim = obj_m25.simulation.redshift


    mags_m25 = np.array(
        [[g.absmag[band]
        for g in obj_m25.galaxies]
        for band in bands]
    )


    mags_m50 = np.array(
        [[g.absmag[band]
        for g in obj_m50.galaxies]
        for band in bands]
    )


    stellar_mass_m25 = np.array(
        [g.masses["stellar"]
        for g in obj_m25.galaxies]
    )


    stellar_mass_m50 = np.array(
        [g.masses["stellar"]
        for g in obj_m50.galaxies]
    )


    #-------------------------------------
    # magnitude cuts
    #-------------------------------------
    mask_m25 = mags_m25[0] < -16
    mask_m50 = mags_m50[0] < -17.5


    mags = np.concatenate(

        [mags_m25[:,mask_m25],
         mags_m50[:,mask_m50]],

        axis=1

    )


    stellar_mass = np.concatenate(

        [stellar_mass_m25[mask_m25],
         stellar_mass_m50[mask_m50]]

    )


    if mags.shape[1] == 0:
        continue


    beta = Calbeta(
        mags,
        wavelengths
    )


    #-------------------------------------
    # luminosity weights
    #-------------------------------------
    M1500 = mags[0]

    lum_weights = 10.**(-0.4*M1500)



    #==================================================
    # Mean
    #==================================================
    mean_beta = np.mean(beta)

    p16 = np.percentile(beta,16)
    p84 = np.percentile(beta,84)

    results["Mean"]["value"].append(mean_beta)
    results["Mean"]["lower"].append(mean_beta-p16)
    results["Mean"]["upper"].append(p84-mean_beta)



    #==================================================
    # Median
    #==================================================
    median_beta = np.median(beta)

    p16 = np.percentile(beta,16)
    p84 = np.percentile(beta,84)

    results["Median"]["value"].append(median_beta)
    results["Median"]["lower"].append(median_beta-p16)
    results["Median"]["upper"].append(p84-median_beta)



    #==================================================
    # luminosity weighted
    #==================================================
    lum_beta = np.average(
        beta,
        weights=lum_weights
    )


    p16 = weighted_percentile(
        beta,
        lum_weights,
        16
    )


    p84 = weighted_percentile(
        beta,
        lum_weights,
        84
    )


    results["Luminosity weighted"]["value"].append(
        lum_beta
    )

    results["Luminosity weighted"]["lower"].append(
        lum_beta-p16
    )

    results["Luminosity weighted"]["upper"].append(
        p84-lum_beta
    )



    #==================================================
    # mass weighted
    #==================================================
    mass_beta = np.average(
        beta,
        weights=stellar_mass
    )


    p16 = weighted_percentile(
        beta,
        stellar_mass,
        16
    )


    p84 = weighted_percentile(
        beta,
        stellar_mass,
        84
    )


    results["Mass weighted"]["value"].append(
        mass_beta
    )

    results["Mass weighted"]["lower"].append(
        mass_beta-p16
    )

    results["Mass weighted"]["upper"].append(
        p84-mass_beta
    )


    zvals.append(z_sim)



#------------------------------------------------
# plotting
#------------------------------------------------
colors = {

    "Mean":"tab:blue",
    "Median":"tab:orange",
    "Luminosity weighted":"tab:green",
    "Mass weighted":"tab:red"

}


markers = {

    "Mean":"o",
    "Median":"s",
    "Luminosity weighted":"D",
    "Mass weighted":"^"

}


plt.figure(figsize=(9,6))


for method in results:


    values = np.array(
        results[method]["value"]
    )

    lower = np.array(
        results[method]["lower"]
    )

    upper = np.array(
        results[method]["upper"]
    )


    #----------------------------------
    # errorbar
    #----------------------------------
    plt.errorbar(

        zvals,
        values,

        yerr=[lower,upper],

        marker=markers[method],
        linestyle="-",

        capsize=4,

        color=colors[method],

        label=method

    )


    #----------------------------------
    # linear fit
    #----------------------------------
    slope,intercept = np.polyfit(
        zvals,
        values,
        1
    )


    zfit = np.linspace(
        min(zvals),
        max(zvals),
        100
    )


    plt.plot(

        zfit,

        slope*zfit+intercept,

        "--",

        color=colors[method],

        alpha=0.7,

        label=f"{method} fit ({slope:.3f})"

    )



#------------------------------------------------
# labels
#------------------------------------------------
plt.xlabel("Redshift")
plt.ylabel(r"$\beta$")

plt.title(
    r"Calzetti Dust: $\beta$ Evolution"
)

plt.legend(
    ncol=2,
    fontsize=9
)

plt.tight_layout()

plt.savefig(
    "Beta_z_calzetti_all_methods.png",
    dpi=300
)

plt.show()
