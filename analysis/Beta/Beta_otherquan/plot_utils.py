import numpy as np
import matplotlib.pyplot as plt

def plot_binned_line(bin_centers, y_values, low_mask, ax, color='green'):
    """
    Plot a binned quantity as a continuous line.
    
    Bins flagged as low-count (low_mask=True) are drawn as dashed segments.
    Bins with low_mask=False are solid.
    
    Parameters
    ----------
    bin_centers : array-like
        x values of the bins (e.g., log stellar mass)
    y_values : array-like
        y values (median, mean, etc.)
    low_mask : array-like of bool
        True where bin count is low (to be drawn dashed)
    color : str
        Line color
    """
    x = np.array(bin_centers)
    y = np.array(y_values)
    mask = np.array(low_mask)

    for i in range(len(x)-1):
        xseg = [x[i], x[i+1]]
        yseg = [y[i], y[i+1]]
        style = '--' if (mask[i] or mask[i+1]) else '-'
        ax.plot(xseg, yseg, linestyle=style, color=color)

