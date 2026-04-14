import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator,FuncFormatter
import statistics
from pathlib import Path

def MyFormatter(x,lim):
     if x == 0:
         return 0
     else:
       x = str(x).split("e")
       return x[0][0] + r"$\times 10^{" + x[1] + r"}$"
     # end if/else
   # end def


mpl.rcParams['pdf.fonttype'] = 42   #this block is all plot formatting
mpl.rcParams['ps.fonttype'] = 42 
mpl.rcParams['font.family'] = 'Arial'

fig = plt.figure(figsize=(9,7))
ax = plt.axes((0.08,0.1,0.88,0.8))

ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(which="both", labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=16)
ax.tick_params(which="both", axis="both", direction="in")

ax.tick_params(which="major", axis="both", length=12, width=3.0, labelsize=22)
ax.tick_params(which="minor", axis="both", length=8.0, width=2.0)

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))

major_formatter = FuncFormatter(MyFormatter)
filename = "SNRplot1.pdf"



bayesfilename = "bayes_nova.dat"   #loading in Bayes Factor data for red triangle points
SNRlist1 = np.loadtxt(bayesfilename, usecols = 0)
lnBFlist = np.loadtxt(bayesfilename, usecols = 2)

gamma = -1.5
t_b=0.33
FF = 0.9729
slope = (1-FF**2.0)
C = 2.5

var1_1 = 0.5 * np.sqrt(3.0) / (np.pi**2.0 * t_b**2.0 * (t_b - 1)**2.0 * (gamma)*23.0259) #Calculating Occam Factors
var1_2 = 0.5 * np.sqrt(3.0) / (np.pi**2.0 * t_b**2.0 * (t_b - 1)**2.0 * (gamma)*23.0259/C)
var2 = (1.0 - 5.0 * t_b + 10.0 * t_b**2.0 - 10.0*t_b**3.0 + 5.0 * t_b**4.0)**0.5
O1_1 = var1_1/var2 * 2.0 * np.pi / gamma
O1_2 = var1_2/var2 * 2.0 * np.pi / gamma

SNRlist2 = np.logspace(0, 2, 1000)
EstimateBF3_1 = 0.5*slope * SNRlist2**2.0 + np.log(O1_1) - 2.0 * np.log(SNRlist2)  #Calculating Bayes factors 
EstimateBF3_2 = 0.5*slope * SNRlist2**2.0 + np.log(O1_2) - 2.0 * np.log(SNRlist2)



p1 = plt.plot(SNRlist2, (EstimateBF3_1), linewidth=4.0, color="orange")
p2 = plt.plot(SNRlist2, (EstimateBF3_2), linewidth=4.0, color="blue")
p3 = plt.plot(SNRlist2, (0.5 * slope * SNRlist2**2.0), linewidth=4.0, color="green")
p4 = plt.plot(SNRlist1, (lnBFlist), 'r^', markersize=13)
plt.axhline(0.5, color='black', linestyle='dashed', linewidth = 4.0)

ax.legend(handles=[p1[0],p2[0],p3[0],p4[0]],
labels=[r'$\ln \mathcal{B}_{21}, C = 1$',r'$\ln \mathcal{B}_{21}, C = 2.5$',r'$\ln \mathcal{B}_{21}, \Delta \ln \mathcal{O}=0$',r'$\ln \mathcal{B}_{21}, {RJMCMC}$'],
loc='upper left',
frameon=False,
fontsize = 22)

fig.suptitle(r'$\gamma = -1.5, t_b = 0.33$', fontsize=28)#, horizontalalignment = "right")

plt.yscale('log')
plt.xlabel(r'$SNR$',fontsize=26)
#plt.ylabel(r'$\ln \mathcal{B}_{21}$',fontsize=26)
plt.ylim(0.3, 30)
plt.xlim(10, 21.7)
plt.savefig(filename)
plt.close()