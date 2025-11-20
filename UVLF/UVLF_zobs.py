import numpy as np
import caesar
import unyt
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def cal_UVLF(magnitude, boxsize, nbins=30, mag_range=[-25, -14]):
    """
    Calculate the UV luminosity function.

    Parameters
    ----------
    magnitude : array_like
        Array of galaxy magnitudes.
    boxsize : float
        Box size in Mpc (comoving).
    nbins : int, optional
        Number of magnitude bins (default 10).
    mag_range : list of float, optional
        [min_mag, max_mag] range for the histogram (default [-22, -14]).

    Returns
    -------
    bin_centers : ndarray
        Centers of magnitude bins (only bins with counts > 0).
    logphi : ndarray
        Log10 of the UVLF in Mpc^-3 mag^-1.
    logphi_err : ndarray
        Poisson error on log10(phi).
    """
    # Histogram
    num, bin_edges = np.histogram(magnitude, bins=nbins, range=mag_range)

    # Bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Remove empty bins
    mask = num > 0
    num = num[mask]
    bin_centers = bin_centers[mask]

    # Bin width (assume uniform)
    dm = (mag_range[1] - mag_range[0]) / nbins

    # UVLF and Poisson error
    phi = num / (dm * boxsize**3)
    phi_err = np.sqrt(num) / (dm * boxsize**3)

    # Log values and propagated error
    logphi = np.log10(phi)
    logphi_err = phi_err / (phi * np.log(10))

    return bin_centers, logphi, logphi_err

#? Function which reads in observational data from file.
def read_observational_data(filepath, logY=False, delim="/", errorbars=True):
    """
    logY (bool) -> compute the log of the phi datapoints and errors
    """
    data = defaultdict(lambda: [])
    with open(filepath, "r") as fptr:
        for line in fptr.readlines():
            # Skip commented lines
            if line[0] == "#":
                continue

            # Flase by default, updates if needs be
            data["lolims"].append(False)
            data["uplims"].append(False)

            # Unpack data in line
            linedata = [entry.strip() for entry in line.split(delim)]
            data["magnitude"].append(float(linedata[0]))
            if logY:
                data["phi"].append(np.log10(float(linedata[1])))
            else:
                data["phi"].append(float(linedata[1]))


            if errorbars and len(linedata) == 3:

                """
                    In the case of symmetric errors, the stdev is supplied
                    rather than the coordinate limits of the errorbar -/+ values.
                    To log this value then requires error propogation.

                    y = log10(x) = ln(x) / ln(10)
                    dy/dx = 1 / (x * ln(10))
                    sigma_y = sigma_x / (x * ln(10))
                """
                if logY:
                    numerator   = float(linedata[2])
                    denominator = (float(linedata[1]) * np.log(10))
                    data["sigma_phi"].append(numerator/denominator)
                else:
                    data["sigma_phi"].append(float(linedata[2]))

            elif errorbars and len(linedata) >= 4:

                    if logY:
                        err_minus = np.log10(float(linedata[1])) - np.log10(float(linedata[2]))
                        err_plus  = np.log10(float(linedata[3])) - np.log10(float(linedata[1]))
                    else:
                        err_minus = float(linedata[1]) - float(linedata[2])
                        err_plus  = float(linedata[3]) - float(linedata[1])

                    if err_plus < 0:
                        data["uplims"][-1] = True
                        data["lolims"][-1] = False
                    elif err_minus < 0:
                        data["uplims"][-1] = False
                        data["lolims"][-1] = True

                    data["sigma_phi"].append([err_minus, err_plus])


            if errorbars and len(linedata) == 6:

                if data["uplims"][-1] or data["lolims"][-1]:
                    err_minus = 0
                    err_plus  = 0
                else:
                    err_minus = float(linedata[0]) - float(linedata[4])
                    err_plus  = float(linedata[5]) - float(linedata[0])

                data["sigma_magnitude"].append([err_minus, err_plus])

            else:
                data["sigma_magnitude"].append(0.)

        # Convert to numpy arrays
        for key in data.keys():
            data[key] = np.array(data[key])

            # Tranpose errors array to make it compatible with matplotlib
            if key == "sigma_phi" and len(linedata) >= 4:
                    data[key] = data[key].T
            if key == "sigma_magnitude" and len(linedata) >=6:
                    data[key] = data[key].T

        # # Take log of phi if requested
        # if logY:
        #     filt = data["phi"] > 0.
        #     log_phi = np.log10(data["phi"][filt])

        #     
        #     elif errorbars and len(linedata) >= 4:
        #         sigma_log_phi = np.zeros((2,len(filt)))
        #         for i in range(2):
        #             numerator   = data["sigma_phi"][i,:][filt]
        #             denominator = (data["phi"][filt] * np.log(10))
        #             sigma_log_phi[i,:] = numerator / denominator
        #         data["sigma_phi"] = sigma_log_phi

        #     data["phi"] = log_phi

    return data

# Load observational data
harikane_fit_kwargs = {"delim":",", "errorbars":False, "logY":True}
bouwens2021_kwargs  = {"delim":",", "logY":True}
ishigaki2018_kwargs = {"delim":",", "errorbars":True}
stefanon2021_kwargs = {"delim":",", "errorbars":True, "logY":True}
harikane2023_kwargs = {"delim":",", "errorbars":True, "logY":True}
logy_kwargs         = {"logY":True}

#  hash symbol in filename will be replaced with redshift integer
UVdatadir = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/LuminosityFunctions/'
obsInfo =  {"Harikane+ 2022":[str(os.path.join(UVdatadir, "harikane2022_z#.txt")), -1, "purple", harikane_fit_kwargs],
                     "McLeod+ 2023":[str(os.path.join(UVdatadir, "mcleod2023_z11.txt")), 11, "green", logy_kwargs],
                     "Bouwens+ 2015":[str(os.path.join(UVdatadir, "bouwens2015_z6.txt")), 6, "orange", logy_kwargs],
                     "Bouwens+ 2021":[str(os.path.join(UVdatadir, "bouwens2021_z#.txt")), -1, "black", bouwens2021_kwargs],
                     "Ishigaki+ 2018":[str(os.path.join(UVdatadir, "ishigaki2018_z#.txt")), -1, "grey", ishigaki2018_kwargs],
                     "Stefanon+ 2021":[str(os.path.join(UVdatadir, "stefanon2021_z#.txt")), -1, "magenta", stefanon2021_kwargs],
                     "Harikane+ 2023":[str(os.path.join(UVdatadir, "harikane2023_z9.txt")), 9, "crimson", harikane2023_kwargs]}


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
colors = {'25':'blue', '50':'red'}

# Load observational data
'''
data = np.loadtxt("/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/LuminosityFunctions/mcleod2023_z11.txt", delimiter='/')
m1500 = data[:, 0]
phi = data[:, 1]
phi_err = data[:, 2]

# Convert to log10
logphi = np.log10(phi)
logphi_err = phi_err / (phi * np.log(10))
'''
# Number of subplots = number of redshifts
nred = len(files_25)
fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True, sharey=True)
axes = axes.flatten()  # flatten to 1D array for easy indexing

'''
# Assuming axes[0] is z=11
axes[0].errorbar(m1500, logphi, yerr=logphi_err, fmt='o', color='green',
            ecolor='green', elinewidth=1.2, capsize=3, label='McLeod+23')
'''
# Add observational data
obs_z = [11, 10, 9, 8, 7, 6]
used_colours = {}
for i in range(nred):
    if obsInfo is None:
        continue
    for dname in obsInfo.keys():
        datapath    = obsInfo[dname][0]
        obsRedshift = obsInfo[dname][1]

        if obsRedshift == -1:
            datapath = datapath.replace("#", str(int(obs_z[i])))
            dataexists = os.path.isfile(datapath)
            print(datapath, dataexists)
            if not dataexists:
                continue
        elif int(obs_z[i]) != np.round(obsRedshift):
            continue

        data = read_observational_data(datapath, **obsInfo[dname][3])
        if "errorbars" not in obsInfo[dname][3].keys() or obsInfo[dname][3]["errorbars"]:
            axes[i].errorbar(data["magnitude"], data["phi"], yerr=np.abs(data["sigma_phi"]), xerr=np.abs(data["sigma_magnitude"]),
                                        linewidth=0., elinewidth=1., color=obsInfo[dname][2], capsize=1.5, marker="o",
                                        zorder=10, lolims=data["lolims"], uplims=data["uplims"])
        else:
            axes[i].plot(data["magnitude"], data["phi"], color=obsInfo[dname][2],
                                        marker=None, zorder=10)

        if dname not in used_colours.keys():
            used_colours[dname] = obsInfo[dname][2]


for i in range(nred):
    ax = axes[i]
    ax.set_xlim(-23.5,-14.5)
    ax.set_ylim(-7.5, 0.5)
    for box, filelist in boxes.items():
        obj = caesar.load(filelist[i])

        boxsize = obj.simulation.boxsize.to("Mpccm").d
        magnitude = np.array([g.absmag["i1500"] for g in obj.galaxies])
        
        # No-dust magnitudes
        mag_nodust = np.array([g.absmag_nodust["i1500"] for g in obj.galaxies])

        bin_centers, logphi, logphi_err = cal_UVLF(magnitude, boxsize, nbins=30)
        bin_centers_nodust, logphi_nodust, logphi_nodust_err = cal_UVLF(mag_nodust, boxsize, nbins=30)

        ax.errorbar(bin_centers, logphi, yerr=logphi_err, capsize=3, capthick=1.2,
                    color=colors[box], label=f"Box {box}" if i==0 else "", fmt='-')

        ax.plot(
        bin_centers_nodust, logphi_nodust,
        color=colors[box],
        linestyle="--",
        label=f"no dust" if i == 0 else "",
        )
        
        
    z = int(round(obj.simulation.redshift))  # convert to int
    ax.text(0.05, 0.9, f"z = {z}", transform=ax.transAxes, 
         fontsize=12, verticalalignment='top', color='black')
    ax.grid(alpha=1)

# x-axis labels for bottom row
for ax in axes[4:]:
    ax.set_xlabel(r"$M_{1500}$", fontsize=12)

# y-axis labels for first column
for ax in axes[::2]:
    ax.set_ylabel(r"$\log_{10}(\phi)$", fontsize=12)

# legend inside first subplot
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, fontsize=12)

# minimal spacing
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                    wspace=0.05, hspace=0.05)

plt.savefig("UVLF_subplots.png")

