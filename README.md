# Detecting-classical-nova-like-explosions-with-LISA
This repository includes all relevant programs used to prepare the paper of the same name.
This Document contains the software license for this repository as well as descriptions of each program.

License:
Copyright (c) 2026 Neil Cornish & Ethan McKeever

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

See <http://www.gnu.org/licenses/> for a copy of the GNU General Public License


Program List:

Nova_burst.c is the main RJMCMC used for parameter estimation in this work.

Nova_burst_scan.c is the RJMCMC set up to scan across one parameter such as burst magnitude, time of burst, or SNR. This program prints Bayes factors to bayes_nova.dat.

Both programs above compile to Nova_burst.

The sampling for both programs is done using a combination of jumps along the eigenvectors of the Fisher information matrix, independent gaussian jumps for each parameter, and model specific differential evolution. The jumps between models are calculated on the same level at the same time as jumps within each model. 

The code uses a noise free likelihood. The output of the RJMCMC gets written to the file "chain.dat". Since the code is zero noise the likelihood can be computed using a fast adaptive integrator in the time domain. The subroutines "integrandF" and "integrateF" are used in computing the likelihood. The waveform is specified using a Taylor expansion in the frequency derivatives, and "pmapping" subroutines map from the physics models to the terms in the Taylor expansion.

Both codes used parallel processing via OpenMP. We installed GSL and OpenMP via Homebrew on our Macs, which require the compiler to point to where the libraries are placed, hence the ugly looking compile lines found at the top of each program.

corner_nova.py plots a corner plot of all 6 waveform parameters and overlays an ellipse for each plot from the Fisher matrix 1-sigma prediction. This can be modified to also use the log likelihood as a parameter as was done in Fig. 2. The log likelihood is the first data point on each line in the chain file.

corner_bias.py plots paired corner plots from two distributions and was used in the making of Fig. 10 in the paper and was slightly adjusted to make Fig. 9.


Waveform_match.py scans for detection and bias SNRs over the size and time of burst. This program was used for the left panels of Figs. 5 and 8.

LinePlots.py scans for detection and bias SNRs over the size of burst. This program was used for the right panels of Figs. 5 and 8.


ScanSNR.py calculates detection SNRs for a variety of SNRs and loads in points from bayes_nova.dat for comparison. This program was used for Figs. 4 and A1.

Phaseplots.py is used to make plots showing phase deviation from true values and was used to create Figs. 3 and 6.


FF_Fisher calculates the fitting factor at 2nd or 4th order in given parameter shifts and calculates the parameter shifts from not including the effects of the burst at the 2nd order level.


test_match.py quickly calculates the fractional shift in frequency, fitting factor, detection SNR, and bias SNR for a given pair of burst parameters.
