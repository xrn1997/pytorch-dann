# Copyright (c) 2017 Tampere University of Technology (TUT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# python script to visualize WLAN RSS fingerprint data
import numpy as np
from numpy import genfromtxt
import matplotlib as mpl
from matplotlib import pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# enter path to directory where data is stored
path_to_database = '../../../FINGERPRINTING_DB'

def load_data(path_to_data):

    # data
    FILE_NAME_RSS = path_to_data + '/Test_rss_21Aug17.csv'
    FILE_NAME_COORDS = path_to_data + '/Test_coordinates_21Aug17.csv'
    # read test data
    X = genfromtxt(FILE_NAME_RSS, delimiter=',')
    y = genfromtxt(FILE_NAME_COORDS, delimiter=',')
    X[X==100] = np.nan
    return (X, y)

def plot_stats(rss_dB):

# Plot some statistics
    numFpPerAp = np.zeros((rss_dB.shape[1],1), dtype=np.int)
    for idx, vector in enumerate(rss_dB.T):
        a = np.logical_not(np.isnan(vector))
        numFpPerAp[idx] = sum(a)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(rss_dB.shape[1]), numFpPerAp, 1)
    ax.set_title('Fingerprints per Access Point');
    ax.set_xlabel('access point ID');
    ax.set_ylabel('number of fingerprints');

    plt.draw()
    return

def plot_rp_grid(rp_m):
# Plot positions of fingerprints for all floors     

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rp_m[:,0],rp_m[:,1], rp_m[:,2], c='b', marker='o')
    ax.set_title('Positions of fingerprints')
    ax.set_xlabel('easting (m)')
    ax.set_ylabel('northing (m)')
    ax.set_zlabel('height (m)')

    plt.draw()
    return

def plot_fp_per_ap(rss_dB, rp_m, ap):
# Plot 3D scatter plot of RSS for single access point

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(rp_m[:,0],rp_m[:,1], rp_m[:,2], c=rss_dB[:,ap], cmap='jet', marker='o')
    fig.colorbar(p, ax=ax)
    ax.set_title('Fingerprints of access point ' + repr(ap+1))
    ax.set_xlabel('easting (m)')
    ax.set_ylabel('northing (m)')
    ax.set_zlabel('height (m)')

    plt.draw()
    return

def plot_rss(rp_m, rss_dB, ap, pltype='plot3'):
# Plot surface, contour, etc. of RSS values for an access point.
# Plot signal strength values of given AP over given metric cartesian
# coordinate system.

    fig = plt.figure()
    X, Y = np.meshgrid(rp_m[0,:,0], rp_m[0,:,1],indexing='ij')
    ZI = griddata((rp_m[0,:,0], rp_m[0,:,1]), rss_dB[:,ap], (X, Y))
    if pltype == 'plot3':
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(rp_m[0,:,0],rp_m[0,:,1], rss_dB[:,ap], c='k', marker='o')
    elif pltype == 'contour':
        ax = fig.add_subplot(111)
        c = ax.contour(X, Y, ZI, cmap='jet')
        fig.colorbar(c, ax=ax)
    elif pltype == 'contour3':
        ax = fig.add_subplot(111, projection='3d')
        c = ax.contour(X, Y, ZI, cmap='jet')
        fig.colorbar(c, ax=ax)
        ax.set_zlabel('RSS (dB)')
#    elif pltype == 'contourf': # requires too much mem
#        ax = fig.add_subplot(111, projection='3d')
#        c = ax.contourf(X, Y, ZI, cmap='jet')
#        fig.colorbar(c, ax=ax)
#        ax.set_zlabel('RSS (dB)')
#    elif pltype == 'surf':
#        ax = fig.add_subplot(111, projection='3d')
#        c = ax.plot_surface(X, Y, ZI, cmap='jet')
#        fig.colorbar(c, ax=ax)
#        ax.set_zlabel('RSS (dB)')
    else:
        print('Unknown plot type')

    ax.set_xlabel('easting (m)')
    ax.set_ylabel('northing (m)')
    ax.set_title('RSS of access point ' + repr(ap+1) + ' on the selected floor')

    plt.draw()
    return


# load data
rss_dB, rp_m = load_data(path_to_database)

print('Number of fingerprints: %i\nNumber of access points: %i' % (rss_dB.shape[0], rss_dB.shape[1]))

# set invalid RSS to NaN;
rss_dB[rss_dB==100] = np.nan


# plot some statistics
plot_stats(rss_dB)

# plot positions of reference points
plot_rp_grid(rp_m)

# plot fingerprints of a selected AP; in this example is 492
selAp =  492-1;
plot_fp_per_ap(rss_dB, rp_m, selAp)

# plot RSS of single access point for one floor
hgt = np.unique(rp_m[:,2])
#select the floor number for visualization; in this example it is 2
floor = 1; # choose between 0:4
idxRpFlX = np.where(rp_m[:,2] == hgt[floor])
# num. of valid FP must be high
plot_rss(rp_m[idxRpFlX, 0:2], rss_dB[idxRpFlX], selAp, 'contour')

plt.show()
