#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 15:28:22 2026

@author: ethanmckeever
"""
import numpy as np
import scipy.special as spc

def FisherFull(t_b, gamma):
    
    tau1 = 1 - t_b
    tau2 = 1 - t_b**2.0
    tau3 = 1 - t_b**3.0
    tau4 = 1 - t_b**4.0
    
    v2 = t_b**2.0 - 2.0*t_b + 1.0
    v3 = t_b**3.0 - 3.0*t_b + 2.0
    v4 = t_b**4.0 - 4.0*t_b + 3.0
    
    Fisher = np.zeros((6,6))
    Fisher[0][0] = 1.0
    Fisher[1][1] = 1.0
    Fisher[1][2] = np.pi
    Fisher[1][3] = (1.0/3.0)*np.pi
    Fisher[1][4] = np.pi * v2
    Fisher[1][5] = -2.0 * np.pi * gamma * tau1
    Fisher[2][2] = (4.0 / 3.0) * np.pi * np.pi
    Fisher[2][3] = (1.0 / 2.0) * np.pi * np.pi
    Fisher[2][4] = (2.0 / 3.0) * np.pi * np.pi * v3
    Fisher[2][5] = -2.0 * np.pi * np.pi * gamma * tau2
    Fisher[3][3] = (1.0 / 5.0) * np.pi * np.pi
    Fisher[3][4] = (1.0 / 6.0) * np.pi * np.pi * v4
    Fisher[3][5] = -(2.0 / 3.0) * np.pi * np.pi * gamma * tau3
    Fisher[4][4] = (4.0 / 3.0) * np.pi * np.pi * (3.0*v2 - v3)
    Fisher[4][5] = -2.0 * np.pi * np.pi * gamma * v2
    Fisher[5][5] = 4.0 * np.pi * np.pi * gamma * gamma * tau1
    
    i = 1
    j = 1
    while(i < 6):
        j = 1;
        while(j < i):
            Fisher[i][j] = Fisher[j][i];
            j +=1;
        i+=1;

    return Fisher

def FisherInv():
    FishernbInv = np.zeros((4,4))
    FishernbInv[0][0] = 1.0
    FishernbInv[1][1] = 9.0
    FishernbInv[1][2] = -18.0/np.pi
    FishernbInv[1][3] = 30.0/np.pi
    FishernbInv[2][1] = -18.0/np.pi
    FishernbInv[2][2] = 48.0/np.pi**2.0
    FishernbInv[2][3] = -90.0/np.pi**2.0
    FishernbInv[3][1] = 30.0/np.pi
    FishernbInv[3][2] = -90.0/np.pi**2.0
    FishernbInv[3][3] = 180.0/np.pi**2.0    
    return FishernbInv

def DeltaParams(Fisher, FishernbInv, gamma, t_b):
    Deltamu = np.zeros(4)
    DeltaLamb = np.zeros(2)
    DeltaLamb[0] = -gamma
    DeltaLamb[1] = 0.0
    j = 0
    while j < 4:
        i = 0
        while i < 4:
            k = 0
            while k < 1:
                Deltamu[j] += -Fisher[i][k+4] * DeltaLamb[k] * FishernbInv[i][j]
                #print(j, -Fisher[i][k+4] * DeltaLamb[k] * FishernbInv[i][j])
                k += 1
            i += 1
        j += 1
    return Deltamu

def metric2func(Deltavar, t_b):
    i = 1
    metric2 = 0.0
    while i < 5:
        j = 1
        while j < 5:
            k = 1
            while k < 5:
                l = 1
                while l < 5:
                    indexlist = np.array([i, j, k, l])
                    #print(indexlist)
                    n1 = np.sum(indexlist == 1)
                    n2 = np.sum(indexlist == 2)
                    n3 = np.sum(indexlist == 3)
                    n4 = np.sum(indexlist == 4)
                    b = n2 + 2 * n3
                    piterm = (2.0 * np.pi) ** (n2 + n4) * np.pi ** n3
                    if n4 == 0:
                        inner = piterm * 1.0/(1.0+b)
                    if n4 == 1:
                        inner = piterm * (1 + t_b**(2+b) + b - t_b * (2+b)) / ((1.0+b) * (2.0+b))
                    if n4 == 2:
                        inner = piterm * ((t_b**2.0)/(1.0+b) - (2.0*t_b)/(2.0+b) + 1.0/(3.0+b) - (2.0*(t_b)**(3+b))/(6.0+11.0*b+6.0*b**2.0+b**3.0))
                    if n4 == 3:
                        if b == 0:
                            inner = piterm * ((1.0-t_b)**4.0) / 4.0
                        if b == 1:
                            inner = piterm * (1.0/20.0) * (1.0 - t_b)**4.0 * (4.0 + t_b)
                        if b == 2:
                            inner = piterm * (1.0/60.0) * (1.0 - t_b)**4.0 * (10.0 + t_b * (4.0 + t_b))
                    if n4 == 4:
                        if b == 0:
                            inner = piterm * ((1.0-t_b)**5.0) / 5.0
                    metric2 += inner * Deltavar[i] * Deltavar[j] * Deltavar[k] * Deltavar[l]
                    l +=1
                k += 1
            j += 1
        i += 1
    metric2 = -metric2 / 8.0
    return metric2

def FFfunc(Fisher, Deltavar, t_b):
    i = 0
    metric = 0.0
    metric2 = 0.0
    while i < 6:
        j = 0
        while j < 6:
            metric += 0.5 * (Fisher[i][j]) * Deltavar[i] * Deltavar[j]
            j += 1
        i += 1
    metric2 = metric2func(Deltavar, t_b)
    FForder4 = 1 - metric + 3.0/2.0 * metric**2.0 + metric2
    FForder2 = 1 - metric
    FF = FForder4
    print(FForder4, FForder2, -metric, 3.0/2.0 * metric**2.0, metric2)
    return FF
        
def main():
    year = 3.15581498e7
    Tobs = 4.0 * year
    freq = 0.0097
    f_shift = -freq * 7.e-6
    gamma = f_shift * Tobs
    gamma = -0.2
    t_b = 0.33
    Fisher = FisherFull(t_b, gamma)
    FishernbInv = FisherInv()
    Deltamu = DeltaParams(Fisher, FishernbInv, gamma, t_b)
    biasSNR = 1.0/(-np.pi * np.sqrt(5.0) * gamma * t_b**2.0 * (1-t_b)**2.0)
    print(biasSNR)
    
    Deltavar = np.append(Deltamu, gamma)
    Deltavar = np.append(Deltavar, 0.0)
    print(Deltavar)
    
    FF = FFfunc(Fisher, Deltavar, t_b)
    bias = 1.0 / np.sqrt(2.0*(1-FF))
    print(FF, bias)

if __name__ == '__main__':
    main()