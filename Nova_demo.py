#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:41:16 2025

@author: ethanmckeever
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl

def main():
    tmin = 0.0
    tmax = 1.0
    t = tmin
    omega1 = 70
    omega2 = 0.98 * omega1
    steps = 1000.0
    t_array = np.zeros(0)
    t2_array = np.zeros(0)
    h1_array = np.zeros(0)
    h2_array = np.zeros(0)
    while t<tmax:
        h1 = np.cos(omega1 * (t-tmin))
        h1_array = np.append(h1_array, h1)
        t_array = np.append(t_array, t-tmin)
        t += (tmax-tmin)/steps
    
    t_0 = 0.35
    t = t_0
    while t<tmax:
        h2 = np.cos(omega2 * (t-tmin) + (omega1 - omega2)*(t_0-tmin))
        h2_array = np.append(h2_array, h2)
        t2_array = np.append(t2_array, t-tmin)
        t += (tmax-tmin)/steps
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(14,6))
    ax = plt.axes((0.07,0.15,0.88,0.78))
    plt.rc('font', size=16)

    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(which="both", labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(which="both", axis="both", direction="in")
    ax.tick_params(which="major", axis="both", length=8, width=1.87)
    #ax.tick_params(which="minor", axis="both", length=5.6, width=1.25)
    ax.tick_params(which="minor", axis="both", length=0.0, width=0.0)
    
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    
    plt.plot(t_array, h1_array, label = 'Original', linewidth=2)
    plt.plot(t2_array, h2_array, label = 'After a Nova Explosion', linewidth=2)
    plt.legend(loc='upper right', frameon=True, fontsize = 18)
    
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    
    plt.ylabel(r'Re[$h$] [arbitrary] ',fontsize=22)
    plt.xlabel(r'$t$ [arbitrary]', fontsize=22)
    
    plt.axvline(x = t_0-tmin, color = 'black', linestyle = 'dashed')
    
    filename = "h_t_nova.pdf"
    plt.savefig(filename)
    plt.close()
    
if __name__ == '__main__':
    main()