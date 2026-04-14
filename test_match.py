#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 08:13:43 2025

@author: ethanmckeever
"""

import numpy as np
import numpy.polynomial as poly
import matplotlib.pyplot as plt
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
    #ll = params1[0]*params2[0]*(1.0-0.5*dphi*dphi)
    
    return(ll)

def integrand(t, params1, params2, flag1, flag2, Tobs):
    
    phi1 = params1[1] + 2.0*np.pi*params1[2]*t + np.pi*params1[3]*t*t +  2.0*np.pi*params1[4] * (t-params1[5]) * flag1
    phi2 = params2[1] + 2.0*np.pi*params2[2]*t + np.pi*params2[3]*t*t +  2.0*np.pi*params2[4] * (t-params2[5]) * flag2
    
    dphi = phi1-phi2
    ll = params1[0]*params2[0]*np.cos(dphi)
    return ll

def phase(t, params, Tobs, flag, biasflag):
    if biasflag == 1:
        phi = params[1] + 2.0*np.pi*params[2]*t + 2.0*np.pi*params[4] * (t-params[5]) * flag
    else:
        phi = params[1] + 2.0*np.pi*params[2]*t + np.pi*params[3]*t*t +  2.0*np.pi*params[4] * (t-params[5]) * flag
    return phi

def main():
    
    year = 3.15581498e7
    Tobs = 4.0*year
    
    simp3 = np.zeros(3)
    simp5 = np.zeros(5)
    params1 = np.zeros(6)
    params2 = np.zeros(6)
    params3 = np.zeros(6)
    params4 = np.zeros(6)
    
    simp3[0] = 1.0/6.0
    simp3[1] = 4.0/6.0
    simp3[2] = 1.0/6.0
    
    simp5[0] = 1.0/12.0
    simp5[1] = 4.0/12.0
    simp5[2] = 2.0/12.0
    simp5[3] = 4.0/12.0
    simp5[4] = 1.0/12.0
    
    freq = 0.0097
    fdot = -1.0e-15
    
    t_b=0.33  #set time of burst and gamma here
    gamma = -1.5
    
    

    params1[0] = 1.0
    params1[1] = 0.0
    params1[2] = freq*Tobs
    params1[3] = fdot * Tobs**2.0 
    params1[4] = gamma
    params1[5] = t_b
    
    length = freq * Tobs * 30 / 100.0
    
    tlist = np.linspace(0.0, 1.0, int(length))
    burstflag1 = np.where(tlist < params1[5], 0.0, 1.0)
    phaselist = phase(tlist, params1, Tobs, burstflag1, 0)
    fit = poly.polynomial.Polynomial.fit(tlist, phaselist, 2, window=[0., 1.])
    
    params2[0] = params1[0]
    params2[1] = fit.coef[0]
    params2[2] = fit.coef[1]/(2*np.pi)
    params2[3] = fit.coef[2]/(np.pi)
    params2[4] = 0.0
    params2[5] = 0.0
    
    #print(params2-params1)
    
    params3[0] = params1[0]
    params3[1] = params1[1]
    params3[2] = params1[2]
    params3[3] = params1[3]
    params3[4] = 0.0
    params3[5] = 0.0

    matchnum = integrate(params2, params1, simp3, simp5, Tobs)
    matchden = np.sqrt(integrate(params2, params2, simp3, simp5, Tobs) * integrate(params1, params1, simp3, simp5, Tobs))
    
    match = matchnum/matchden
    biasSNR = 1.0/(np.pi * np.sqrt(5) * np.abs(gamma) * t_b**2.0 * (1.0 - t_b)**2.0)

    C = 1.0
    
    var1 = -(1-match**2.0) * np.sqrt(3.0) / (2.0 * np.pi * np.exp(0.5) * t_b**2.0 * (t_b - 1)**2.0 * (gamma)**2.0 * 23.0259/C)
    var2 = (1.0 - 5.0 * t_b + 10.0 * t_b**2.0 - 10.0*t_b**3.0 + 5.0 * t_b**4.0)**0.5
    c = var1/var2
    a = -(1-match**2.0) / 2.0
    w = scipy.special.lambertw(c, k=-1)
    detSNR = np.sqrt(w/a).real
    
    print("gamma:", gamma)
    print("Time of Burst:", t_b)
    print("Fractional Shift:", gamma/Tobs/freq)
    print("Fitting Factor:", match)
    print("Bias SNR:", biasSNR)
    print("Detection SNR:", detSNR)
    
if __name__ == '__main__':
    main()