#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 20:39:51 2025

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
    
    tol = 1.0e-5
    errmin = 1.0e-4
    
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
        #print(s5)
        
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
    #print(np.cos(dphi), t, dphi)
    #print(params2)
    
    return(ll)

def integrand(t, params1, params2, flag1, flag2, Tobs):
    
    phi1 = params1[1] + 2.0*np.pi*params1[2]*t + np.pi*params1[3]*t*t +  2.0*np.pi*params1[4] * (t-params1[5]) * flag1
    phi2 = params2[1] + 2.0*np.pi*params2[2]*t + np.pi*params2[3]*t*t +  2.0*np.pi*params2[4] * (t-params2[5]) * flag2
    
    dphi = phi1-phi2
    ll = params1[0]*params2[0]*np.cos(dphi)
    return ll
    
    
def wavematch(params1, params2, simp3, simp5, Tobs, length):

    matchnum = integrate(params2, params1, simp3, simp5, Tobs)
    #matchden = np.sqrt(integrate(params2, params2, simp3, simp5, Tobs) * integrate(params1, params1, simp3, simp5, Tobs))
    matchden = np.sqrt(params2[0]*params1[0])

    match = matchnum/matchden

    return match

def wavelikelihood(params1, params2, simp3, simp5, Tobs, length):
    num= integrate(params2, params1, simp3, simp5, Tobs)
    logly = -0.5 * (params1[0]**2.0 + params2[0]**2.0) + num
    return logly

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

    fig = plt.figure(figsize=(9,3))
    #ax = plt.axes((0.1,0.13,0.94,0.8))
    ax = plt.axes((0.1,0.17,0.88,0.8))

    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(which="both", labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=16)
    ax.tick_params(which="both", axis="both", direction="in")
    #ax.tick_params(which="major", axis="both", length=10, width=2.0, labelsize=16)
    #ax.tick_params(which="minor", axis="both", length=6.0, width=1.4)

    ax.tick_params(which="major", axis="both", length=10, width=2.0, labelsize=16)
    ax.tick_params(which="minor", axis="both", length=6.0, width=1.5)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    major_formatter = FuncFormatter(MyFormatter)
    
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


    params0 = np.zeros(6)
    params1 = np.zeros(6)
    paramsp = np.zeros(6)
    paramsp2 = np.zeros(6)
    params2 = np.zeros(6)
    
    Tobs = 4.0 * year
    freq = 0.0097
    fdot = -3.1e-15
    t_b1 = 0.25
    t_b2 = 0.5
    length = freq * Tobs * 30 / 1000.0
    
    #Use this block to make Fig 6.
    
    gamma = -1.0
    
    paramsp[0] = 1.0
    paramsp[1] = 0.0
    paramsp[2] = freq*Tobs
    paramsp[3] = fdot*Tobs*Tobs
    paramsp[4] = gamma
    paramsp[5] = t_b1
    
    paramsp2[0] = 1.0
    paramsp2[1] = 0.0
    paramsp2[2] = freq*Tobs
    paramsp2[3] = fdot*Tobs*Tobs
    paramsp2[4] = gamma
    paramsp2[5] = t_b2
    
    tlist1 = np.linspace(0.0, 1.0, int(length))
    burstflag1 = np.where(tlist1 < paramsp[5], 0.0, 1.0)
    phaselist1 = phase(tlist1, paramsp, Tobs, burstflag1)
    fit1 = poly.polynomial.Polynomial.fit(tlist1, phaselist1, 2, window=[0., 1.])
    
    tlist2 = np.linspace(0.0, 1.0, int(length))
    burstflag2 = np.where(tlist2 < paramsp2[5], 0.0, 1.0)
    phaselist2 = phase(tlist2, paramsp2, Tobs, burstflag2)
    fit2 = poly.polynomial.Polynomial.fit(tlist2, phaselist2, 2, window=[0., 1.])
    
    params0[0] = 1.0
    params0[1] = 0.0
    params0[2] = freq*Tobs
    params0[3] = fdot*Tobs*Tobs
    params0[4] = 0.0
    params0[5] = 0.0
    
    params1[0] = 1.0
    params1[1] = fit1.coef[0]
    params1[2] = fit1.coef[1]/(2*np.pi)
    params1[3] = fit1.coef[2]/(np.pi)
    params1[4] = 0.0
    params1[5] = 0.0
    
    params2[0] = 1.0
    params2[1] = fit2.coef[0]
    params2[2] = fit2.coef[1]/(2*np.pi)
    params2[3] = fit2.coef[2]/(np.pi)
    params2[4] = 0.0
    params2[5] = 0.0
    
    dbeta1 = fit1.coef[2]/(np.pi) - fdot*Tobs*Tobs
    dbeta2 = fit2.coef[2]/(np.pi) - fdot*Tobs*Tobs

    match1 = wavematch(paramsp, params1, simp3, simp5, Tobs, length);
    match2 = wavematch(paramsp2, params2, simp3, simp5, Tobs, length);
    
    likelihood1 = wavelikelihood(paramsp, params1, simp3, simp5, Tobs, length);
    likelihood2 = wavelikelihood(paramsp2, params2, simp3, simp5, Tobs, length);
    
    print("Likelihoods: ", likelihood1, likelihood2)
    
    print("Matches:", match1, match2)
    
    burstflag1 = np.where(tlist1 < params1[5], 0.0, 1.0)
    burstflag2 = np.where(tlist2 < params2[5], 0.0, 1.0)
    phase1 = phase(tlist1, params1, Tobs, burstflag1) - phaselist1
    phase2 = phase(tlist2, params2, Tobs, burstflag2) - phaselist2

    p1 = plt.plot(tlist1, phase1, linewidth=3.0, color="blue")
    p2 = plt.plot(tlist2, phase2, linewidth=3.0, color="red")
    
    plt.axhline(0.0, color='black', linestyle='dashed', linewidth = 2.0)
    
    ax.legend(handles=[p1[0],p2[0],],
    labels=[r"$t_b = 0.25$, "+ r'$\Delta \beta=$' + str(np.round(dbeta1, 3)) + "           ", r"$t_b = 0.50$, " + r'$\Delta \beta=$' + str(np.round(dbeta2, 3))],
    loc='best',
    frameon=False,
    fontsize = 15)
    
    plt.xlabel(r'$t$', fontsize = 18)
    plt.ylabel(r'$\phi(t) - \phi_{\rm{tr}}(t)$', fontsize = 18)

    filename = "phasechecks2.pdf"
    
    
    #Use this block to make Fig 3.
    '''
    gamma = -1.5
    params1 = np.zeros(6)
    paramsp = np.zeros(6)
    params2 = np.zeros(6)
    params3 = np.zeros(6)
    
    paramsp[0] = 1.0
    paramsp[1] = 0.0
    paramsp[2] = freq*Tobs
    paramsp[3] = fdot*Tobs*Tobs
    paramsp[4] = gamma
    paramsp[5] = t_b
    
    tlist = np.linspace(0.0, 1.0, int(length))
    burstflag = np.where(tlist < paramsp[5], 0.0, 1.0)
    phaselist = phase(tlist, paramsp, Tobs, burstflag)
    fit = poly.polynomial.Polynomial.fit(tlist, phaselist, 2, window=[0., 1.])
    
    params1[0] = 1.0
    params1[1] = 0.0
    params1[2] = freq*Tobs
    params1[3] = fdot*Tobs*Tobs
    params1[4] = 0.0
    params1[5] = 0.0
    
    params2[0] = params1[0]
    params2[1] = fit.coef[0]
    params2[2] = fit.coef[1]/(2*np.pi)
    params2[3] = fit.coef[2]/(np.pi)
    params2[4] = -freq*1.e-8*Tobs
    params2[5] = t_b
    
    params3[0] = 1.0
    params3[1] = -1.56952423e-02
    params3[2] = 1.22445658e+06
    params3[3] = -5.27297858e+01
    params3[4] = 9.63090908e-01
    params3[5] = 7.05096651e-01

    match1 = wavematch(paramsp, params1, simp3, simp5, Tobs, length);
    match2 = wavematch(paramsp, params2, simp3, simp5, Tobs, length);
    match3 = wavematch(paramsp, params3, simp3, simp5, Tobs, length);
    
    likelihood2 = wavelikelihood(paramsp, params2, simp3, simp5, Tobs, length);
    likelihood3 = wavelikelihood(paramsp, params3, simp3, simp5, Tobs, length);
    
    print(likelihood2, likelihood3)
    
    print("matches: ", match1, match2, match3)
    
    burstflag1 = np.where(tlist < params1[5], 0.0, 1.0)
    burstflag2 = np.where(tlist < params2[5], 0.0, 1.0)
    burstflag3 = np.where(tlist < params3[5], 0.0, 1.0)
    phase1 = phase(tlist, params1, Tobs, burstflag1) - phaselist
    phase2 = phase(tlist, params2, Tobs, burstflag2) - phaselist
    phase3 = phase(tlist, params3, Tobs, burstflag3) - phaselist

    
    p1 = plt.plot(tlist, phase2, linewidth=3.0, color="mediumorchid")
    p2 = plt.plot(tlist, phase3, linewidth=3.0, color="red")
    
    plt.axhline(0.0, color='black', linestyle='dashed', linewidth = 2.0)
    
    ax.legend(handles=[p1[0],p2[0]],
    labels=[r"$\gamma \ll \gamma_{\rm{tr}}$, "+ r'$M=$' + str(np.round(match2, 3)) + "                            ", r"$\gamma/\gamma_{\rm{tr}} < 0$ , $t_b=0.7$, " + r'$M=$' + str(np.round(match3, 3))],
    loc='best',
    frameon=False,
    fontsize = 15)
    
    plt.xlabel(r'$t$', fontsize = 18)
    plt.ylabel(r'$\phi(t) - \phi_{\rm{tr}}(t)$', fontsize = 18)

    filename = "phasechecks.pdf"
    '''
    
    

    plt.savefig(filename)
    plt.close()
    
    tcomp_f = time.perf_counter()
    print(tcomp_f-tcomp_i)

if __name__ == '__main__':
    main()