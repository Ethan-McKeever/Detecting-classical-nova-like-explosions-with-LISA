#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:02:54 2026

@author: ethanmckeever
"""

import numpy as np
import numpy.polynomial as poly
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator,FuncFormatter
import scipy.special

def integrate(params1, params2, simp3, simp5, Tobs):
    
    tmin = 0.0
    tmax = 1.0
    
    tol = 1.0e-4
    errmin = 1.0e-3
    
    hmax = (tmax-tmin)/4500.0
   
    h = hmax/10.0
    t = tmin
    
    I5 = np.zeros(5)
    I3 = np.zeros(3)
    
    IG = 0.0
    j = 0.0
    
    while (t < tmax  and j < 5000):
        while True:
            dh = h/4.0
            i = 0
            while i < 5:
                tt = t + float(i)*dh
                I5[i] = integrand0(tt, params1, params2, Tobs)
                i += 1
            
            #tt = np.linspace(t, t + 5*dh, 5)
            #print(tt)
            #burstflag1 = np.where(tt < params1[5], 0.0, 1.0)
            #burstflag2 = np.where(tt < params2[5], 0.0, 1.0)
            #I5 = integrand(tt, params1, params2, burstflag1, burstflag2, Tobs)
            
            #print(I5)
            
            I3 = I5[0::2]
            
            #3 point Simpson
            s3 = np.dot(I3, simp3)
            s3 = s3*h
            #5 point Simpson
            s5 = np.dot(I5, simp5)*h
            
            #absolute  error
            err = np.abs(s5-s3)/(15.0)
            #fractional  error
            ferr = np.abs(err/s5)
            
            #print(ferr, tol, err, errmin)
            
            if(ferr > tol and err > errmin):
                h /= 2.0
            
            if(ferr < tol or err < errmin):
                break
            
        
        t += h
        IG += s5
        
        #we try a larger step for the next iteration
        #this might then have to be shrunk
        h = h*2.0
        if(h > hmax):
            h = hmax
        j += 1
        
       #printf("%d %e\n", j, h);
        
    
    if(j >= 5000): #when the integrator needs many steps the likelihood won't be acceptable
        IG = -1.0e10
    else:
        
        #subtract the overshoot
        
        h = t-tmax
        t = tmax
        
        dh = h/4.0
        #i = 0
        #while i < 5:
        #    tt = t + float(i)*dh
        #    I5[i] = integrand(tt, params1, params2, Tobs)
        #    i += 1
            
        tt = np.linspace(t, t + 5*dh, 5)
        burstflag1 = np.where(tt < params1[5], 0.0, 1.0)
        burstflag2 = np.where(tt < params2[5], 0.0, 1.0)
        I5 = integrand(tt, params1, params2, burstflag1, burstflag2, Tobs)
            
        #5 point Simpson
        s5 = np.dot(I5, simp5)*h
        
        IG -= s5

    
    return IG
    


def integrand0(t, params1, params2, Tobs):
    
    if(t > params1[5]):
        phi1 = 2.0*np.pi*((params1[2])*t+0.5*(params1[3])*t*t+(params1[4])*(t-params1[5]))+(params1[1])  
    else:
        phi1 = 2.0*np.pi*((params1[2])*t+0.5*(params1[3])*t*t)+(params1[1])
    
    if(t > params2[5]):
        phi2 = 2.0*np.pi*((params2[2])*t+0.5*(params2[3])*t*t+(params2[4])*(t-params2[5]))+(params2[1])
    else:
        phi2 = 2.0*np.pi*((params2[2])*t+0.5*(params2[3])*t*t)+(params2[1])
    
    dphi = phi1-phi2
    
    ll = params1[0]*params2[0]*np.cos(dphi)
    
    return(ll)

def integrand(t, params1, params2, flag1, flag2, Tobs):
    
    phi1 = params1[1] + 2.0*np.pi*params1[2]*t + np.pi*params1[3]*t*t +  2.0*np.pi*params1[4] * (t-params1[5]) * flag1
    phi2 = params2[1] + 2.0*np.pi*params2[2]*t + np.pi*params2[3]*t*t +  2.0*np.pi*params2[4] * (t-params2[5]) * flag2
    
    dphi = phi1-phi2
    ll = params1[0]*params2[0]*np.cos(dphi)
    return ll
    
    
def wavematch(f_shift, t_b, params1, simp3, simp5, Tobs, length):
    
    paramsp = np.zeros(6)
    params2 = np.zeros(6)
    params3 = np.zeros(6)
    params4 = np.zeros(6)

    paramsp[0] = params1[0]
    paramsp[1] = params1[1]
    paramsp[2] = params1[2]
    paramsp[3] = params1[3]
    paramsp[4] = f_shift
    paramsp[5] = t_b
    
    tlist = np.linspace(0.0, 1.0, int(length))
    burstflag1 = np.where(tlist < paramsp[5], 0.0, 1.0)
    phaselist = phase(tlist, paramsp, Tobs, burstflag1)
    fit = poly.polynomial.Polynomial.fit(tlist, phaselist, 2, window=[0., 1.])
    
    params2[0] = params1[0]
    params2[1] = fit.coef[0]
    params2[2] = fit.coef[1]/(2*np.pi)
    params2[3] = fit.coef[2]/(np.pi)
    params2[4] = 0.0
    params2[5] = 0.0

    matchnum = integrate(params2, paramsp, simp3, simp5, Tobs)
    matchden = np.sqrt(integrate(params2, params2, simp3, simp5, Tobs) * integrate(paramsp, paramsp, simp3, simp5, Tobs))

    match = matchnum/matchden

    return match

def phase(t, params, Tobs, flag):
    phi = params[1] + 2.0*np.pi*params[2]*t + np.pi*params[3]*t*t +  2.0*np.pi*params[4] * (t-params[5]) * flag
    return phi

def MyFormatter(x,lim):
      if x == 0:
          return 0
      else:
        x = str(x).split("e")
        return x[0][0] + r"$\times 10^{" + x[1] + r"}$"
      # end if/else
    # end def

def main():

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(9,7))
    #ax = plt.axes((0.1,0.13,0.94,0.8))
    ax = plt.axes((0.15,0.17,0.8,0.75))

    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(which="both", labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=16)
    ax.tick_params(which="both", axis="both", direction="out")
    
    ax.tick_params(which="major", axis="both", length=12, width=3.0, labelsize=22)
    ax.tick_params(which="minor", axis="both", length=8.0, width=2.0)

    major_formatter = FuncFormatter(MyFormatter)

    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    
    tcomp_i = time.perf_counter()
    year = 3.15581498e7

    
    simp3 = np.zeros(3)
    simp5 = np.zeros(5)
    
    simp3[0] = 1.0/6.0
    simp3[1] = 4.0/6.0
    simp3[2] = 1.0/6.0
    
    simp5[0] = 1.0/12.0
    simp5[1] = 4.0/12.0
    simp5[2] = 2.0/12.0
    simp5[3] = 4.0/12.0
    simp5[4] = 1.0/12.0

    Tobs = 4.0 * year
    freq = 0.0097
    fdot = -3.1e-15
    t_b_min = 0.01
    t_b_max = 0.99
    f_shift_max = -freq*1.e-5*5.0*Tobs
    f_shift_min = -freq*1.e-7*Tobs
    t_steps = 100.0
    f_steps = 100.0
    t_stepsize = (t_b_max - t_b_min) / (t_steps - 1.0)
    f_stepsize = pow(f_shift_max/f_shift_min, 1.0/(f_steps-1.0))
    t_b = t_b_min
    
    length = freq * Tobs * 30 / 1000.0
    #print(length)

    params1 = np.zeros(6)
    
    params1[0] = 1.0
    params1[1] = 0.0
    params1[2] = freq*Tobs
    params1[3] = fdot*Tobs*Tobs
    params1[4] = 0.0
    params1[5] = 0.0
    
    t_blist = np.zeros(int(t_steps))
    gammalist = np.zeros(int(f_steps))
    SNRarray = np.zeros((int(t_steps), int(f_steps)))
    
    biasflag = 0 #0 for detection, 1 for beta bias

    i = 0
    while(t_b <= t_b_max+(t_stepsize/2.0)):
        f_shift = f_shift_min
        j = 0
        while(f_shift >= f_shift_max * 1.00005):
            if biasflag == 0:
                match = wavematch(f_shift, t_b, params1, simp3, simp5, Tobs, length);
                var1 = -(1-match**2.0) * np.sqrt(3.0) / (2.0 * np.pi * np.exp(0.5) * t_b**2.0 * (t_b - 1)**2.0 * (f_shift)**2.0 * 23.0259)
                var2 = (1.0 - 5.0 * t_b + 10.0 * t_b**2.0 - 10.0*t_b**3.0 + 5.0 * t_b**4.0)**0.5
                c = var1/var2
                a = -(1-match**2.0) / 2.0
                w = scipy.special.lambertw(c, k=-1)
                SNR = np.sqrt(w/a).real
            else:
                SNR = 1.0/(-np.pi * np.sqrt(5.0) * f_shift * t_b**2.0 * (1-t_b)**2.0)
            SNRarray[i][j] = SNR
            f_shift *= f_stepsize;
            if (i == int(t_steps) - 1):
                gammalist[j] = f_shift
            j += 1
        t_blist[i] = t_b
        i += 1
        t_b += t_stepsize;

    
    colors = plt.pcolormesh(-gammalist, t_blist, SNRarray, norm=mpl.colors.LogNorm(vmin = 1.0, vmax = 300.0), cmap="coolwarm")
    cbar = fig.colorbar(colors)
    
    cbar.ax.tick_params(which="major", length=12, width=2.5, labelsize=22)
    cbar.ax.tick_params(which="minor", length=8, width=2.0)
    if biasflag == 0:
        cbar.set_label(r'$\rho_{\rm{det}}$', size=30, weight='bold')
        filename = "Waveform_match_detection.pdf"
    else:
        cbar.set_label(r'$\rho_{\rm{sb}}$', size=30, weight='bold')
        filename = "Waveform_match_bias.pdf"
    
    plt.xscale('log')
    plt.xlabel(r'$-\gamma$', fontsize = 30)
    plt.ylabel(r'$t_b$', fontsize = 30)
    
    plt.savefig(filename)
    plt.close()
    
    tcomp_f = time.perf_counter()
    print(tcomp_f-tcomp_i)

if __name__ == '__main__':
    main()