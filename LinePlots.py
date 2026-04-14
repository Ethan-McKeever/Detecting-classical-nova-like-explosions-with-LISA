#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 08:13:43 2025

@author: ethanmckeever
"""

import numpy as np
import numpy.polynomial as poly
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator,FuncFormatter

def MyFormatter(x,lim):
     if x == 0:
         return 0
     else:
       x = str(x).split("e")
       return x[0][0] + r"$\times 10^{" + x[1] + r"}$"
     # end if/else
   # end def

def integrate(params1, params2, simp3, simp5, Tobs):
    
    tmin = 0.0
    tmax = 1.0
    
    tol = 1.0e-4
    errmin = 1.0e-3
    
    hmax = (tmax-tmin)/450.0
   
    h = hmax/10.0
    t = tmin
    
    I5 = np.zeros(5)
    I3 = np.zeros(3)
    
    IG = 0.0
    j = 0.0
    
    while (t < tmax  and j < 500):
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
        
    
    if(j >= 500): #when the integrator needs many steps the likelihood won't be acceptable
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
            
        #tt = np.linspace(t, t + 5*dh, 5)
        #burstflag1 = np.where(tt < params1[5], 0.0, 1.0)
        #burstflag2 = np.where(tt < params2[5], 0.0, 1.0)
        #I5 = integrand(tt, params1, params2, burstflag1, burstflag2, Tobs)
        
        i = 0
        while i < 5:
            tt = t + float(i)*dh
            I5[i] = integrand0(tt, params1, params2, Tobs)
            i += 1
            
        #5 point Simpson
        s5 = np.dot(I5, simp5)*h
        
        IG -= s5

    
    return IG
    


def integrand0(t, params1, params2, Tobs):    #integrand for use with a loop
    
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

def integrand(t, params1, params2, flag1, flag2, Tobs): #integrand for use with arrays
    
    phi1 = params1[1] + 2.0*np.pi*params1[2]*t + np.pi*params1[3]*t*t +  2.0*np.pi*params1[4] * (t-params1[5]) * flag1
    phi2 = params2[1] + 2.0*np.pi*params2[2]*t + np.pi*params2[3]*t*t +  2.0*np.pi*params2[4] * (t-params2[5]) * flag2
    
    dphi = phi1-phi2
    ll = params1[0]*params2[0]*np.cos(dphi)
    return ll

def phase(t, params, Tobs, flag):
    
    phi = params[1] + 2.0*np.pi*params[2]*t + np.pi*params[3]*t*t +  2.0*np.pi*params[4] * (t-params[5]) * flag
    return phi

def main():
    
    year = 3.15581498e7
    Tobs = 4.0*year
    
    simp3 = np.zeros(3)
    simp5 = np.zeros(5)
    params1_1 = np.zeros(6)
    params2_1 = np.zeros(6)
    params1_2 = np.zeros(6)
    params2_2 = np.zeros(6)
    params1_3 = np.zeros(6)
    params2_3 = np.zeros(6)
    
    simp3[0] = 1.0/6.0
    simp3[1] = 4.0/6.0
    simp3[2] = 1.0/6.0
    
    simp5[0] = 1.0/12.0
    simp5[1] = 4.0/12.0
    simp5[2] = 2.0/12.0
    simp5[3] = 4.0/12.0
    simp5[4] = 1.0/12.0
    
    f_steps = 40.0

    detlist1 = np.zeros(0)
    detlist2 = np.zeros(0)
    detlist3 = np.zeros(0)
    
    gammalist1 = np.zeros(0)
    gammalist2 = np.zeros(0)
    gammalist3 = np.zeros(0)
    
    freq = 0.01
    
    length = freq * Tobs * 30 / 1000.0
    
    freq = 0.0097
    fdot = -3.1e-15
    t_b1 = 0.1
    t_b2 = 0.3
    t_b3 = 0.5
    f_shift_min = -1.e-8*freq*Tobs
    f_shift_max = -5.e-5*freq*Tobs
    f_stepsize = pow(f_shift_max/f_shift_min, 1.0/(f_steps-1.0))
    f_shift = f_shift_min
    
    biasflag = 1 # 0 for detection, 1 for beta bias
    
    while f_shift > f_shift_max*1.0005:
        
        params1_1[0] = 1.0
        params1_1[1] = 0.0
        params1_1[2] = freq*Tobs
        params1_1[3] = fdot * Tobs**2.0 
        params1_1[4] = f_shift
        params1_1[5] = t_b1
        
        params1_2[0] = 1.0
        params1_2[1] = 0.0
        params1_2[2] = freq*Tobs
        params1_2[3] = fdot * Tobs**2.0 
        params1_2[4] = f_shift
        params1_2[5] = t_b2
        
        params1_3[0] = 1.0
        params1_3[1] = 0.0
        params1_3[2] = freq*Tobs
        params1_3[3] = fdot * Tobs**2.0 
        params1_3[4] = f_shift
        params1_3[5] = t_b3
        
        if biasflag == 0:
            tlist = np.linspace(0.0, 1.0, int(length))
            burstflag1 = np.where(tlist < params1_1[5], 0.0, 1.0)
            burstflag2 = np.where(tlist < params1_2[5], 0.0, 1.0)
            burstflag3 = np.where(tlist < params1_3[5], 0.0, 1.0)
        
            phaselist1 = phase(tlist, params1_1, Tobs, burstflag1)
            phaselist2 = phase(tlist, params1_2, Tobs, burstflag2)
            phaselist3 = phase(tlist, params1_3, Tobs, burstflag3)
    
            fit1 = poly.polynomial.Polynomial.fit(tlist, phaselist1, 2, window=[0., 1.])
            fit2 = poly.polynomial.Polynomial.fit(tlist, phaselist2, 2, window=[0., 1.])
            fit3 = poly.polynomial.Polynomial.fit(tlist, phaselist3, 2, window=[0., 1.])
    
            params2_1[0] = params1_1[0]
            params2_1[1] = fit1.coef[0]
            params2_1[2] = fit1.coef[1]/(2*np.pi)
            params2_1[3] = fit1.coef[2]/(np.pi)
        
            params2_2[0] = params1_2[0]
            params2_2[1] = fit2.coef[0]
            params2_2[2] = fit2.coef[1]/(2*np.pi)
            params2_2[3] = fit2.coef[2]/(np.pi)
        
            params2_3[0] = params1_3[0]
            params2_3[1] = fit3.coef[0]
            params2_3[2] = fit3.coef[1]/(2*np.pi)
            params2_3[3] = fit3.coef[2]/(np.pi)
    
            matchnum1 = integrate(params2_1, params1_1, simp3, simp5, Tobs)
            matchnum2 = integrate(params2_2, params1_2, simp3, simp5, Tobs)
            matchnum3 = integrate(params2_3, params1_3, simp3, simp5, Tobs)
            matchden1 = np.sqrt(integrate(params2_1, params2_1, simp3, simp5, Tobs) * integrate(params1_1, params1_1, simp3, simp5, Tobs))
            matchden2 = np.sqrt(integrate(params2_2, params2_2, simp3, simp5, Tobs) * integrate(params1_2, params1_2, simp3, simp5, Tobs))
            matchden3 = np.sqrt(integrate(params2_3, params2_3, simp3, simp5, Tobs) * integrate(params1_3, params1_3, simp3, simp5, Tobs))
    
            match1 = matchnum1/matchden1
            match2 = matchnum2/matchden1
            match3 = matchnum3/matchden1

            var1_1 = -(1-match1**2.0) * np.sqrt(3.0) / (2.0 * np.pi * np.exp(0.5) * t_b1**2.0 * (t_b1 - 1)**2.0 * (f_shift)**2.0 *23.0259)
            var2_1 = (1.0 - 5.0 * t_b1 + 10.0 * t_b1**2.0 - 10.0*t_b1**3.0 + 5.0 * t_b1**4.0)**0.5
            c1 = var1_1/var2_1
            a1 = -(1-match1**2.0) / 2.0
            w1 = scipy.special.lambertw(c1, k=-1)
            SNR1 = np.sqrt(w1/a1)
        
            var1_2 = -(1-match2**2.0) * np.sqrt(3.0) / (2.0 * np.pi * np.exp(0.5) * t_b2**2.0 * (t_b2 - 1)**2.0 * (f_shift)**2.0 *23.0259)
            var2_2 = (1.0 - 5.0 * t_b2 + 10.0 * t_b2**2.0 - 10.0*t_b2**3.0 + 5.0 * t_b2**4.0)**0.5
            c2 = var1_2/var2_2
            a2 = -(1-match2**2.0) / 2.0
            w2 = scipy.special.lambertw(c2, k=-1)
            SNR2 = np.sqrt(w2/a2)
        
            var1_3 = -(1-match3**2.0) * np.sqrt(3.0) / (2.0 * np.pi * np.exp(0.5) * t_b3**2.0 * (t_b3 - 1)**2.0 * (f_shift)**2.0 *23.0259)
            var2_3 = (1.0 - 5.0 * t_b3 + 10.0 * t_b3**2.0 - 10.0*t_b3**3.0 + 5.0 * t_b3**4.0)**0.5
            c3 = var1_3/var2_3
            a3 = -(1-match3**2.0) / 2.0
            w3 = scipy.special.lambertw(c3, k=-1)
            SNR3 = np.sqrt(w3/a3)
        else:
            SNR1 = 1.0 / (np.pi * np.sqrt(5.0) * np.abs(params1_1[4]) * t_b1**2.0 * (1.0 - t_b1)**2.0)
            SNR2 = 1.0 / (np.pi * np.sqrt(5.0) * np.abs(params1_2[4]) * t_b2**2.0 * (1.0 - t_b2)**2.0)
            SNR3 = 1.0 / (np.pi * np.sqrt(5.0) * np.abs(params1_3[4]) * t_b3**2.0 * (1.0 - t_b3)**2.0)

        detlist1 = np.append(detlist1, SNR1)
        gammalist1 = np.append(gammalist1, -f_shift)
        
        detlist2 = np.append(detlist2, SNR2)
        gammalist2 = np.append(gammalist2, -f_shift)
        
        detlist3 = np.append(detlist3, SNR3)
        gammalist3 = np.append(gammalist3, -f_shift)
        
        f_shift = f_shift*f_stepsize
    

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(9,7))
    #ax = plt.axes((0.1,0.13,0.94,0.8))
    ax = plt.axes((0.15,0.17,0.83,0.75))

    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(which="both", labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=16)
    ax.tick_params(which="both", axis="both", direction="out")

    ax.tick_params(which="major", axis="both", length=12, width=3.0, labelsize=22)
    ax.tick_params(which="minor", axis="both", length=8.0, width=2.0)

    major_formatter = FuncFormatter(MyFormatter)
   
    p1 = plt.plot(gammalist1, detlist1, linewidth = 3.0)
    p2 = plt.plot(gammalist2, detlist2, linewidth = 3.0)
    p3 = plt.plot(gammalist3, detlist3, linewidth = 3.0)
    
    plt.axhline(5.0, color='black', linestyle='dashed', linewidth = 3.0)

   
    ax.legend(handles=[p1[0],p2[0],p3[0]],
    labels=[r'$t_b = 0.1$',r'$t_b = 0.3$',r'$t_b = 0.5$'],
    loc='best',
    frameon=False,
    fontsize = 26)
    
    if biasflag == 0:
        filename = "DetectionLine.pdf"
        plt.ylabel(r'$\rho_{\rm{det}}$',fontsize=30)
    else:
        filename = "BiasLine.pdf"
        plt.ylabel(r'$\rho_{\rm{sb}}$',fontsize=30)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$|\gamma|$',fontsize=30)
    plt.ylim(2,6000)
    plt.xlim(0.01, 30)
    plt.savefig(filename)
    plt.close()

    
    
if __name__ == '__main__':
    main()