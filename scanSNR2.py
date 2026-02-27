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


filename = "SNRplot1.pdf"        #Input key values here
bayesfilename = "bayes_nova.dat"
Tobs = 4.0 * 3.15581498e7
freq = 0.0097
f_shift = -1.5/Tobs
t_b = 0.33
FF = 0.9729


SNRlist = np.zeros(0)
proplist = np.zeros(0)
SNRlist3 = np.zeros(0)
proplist3 = np.zeros(0)
i = 0


mpl.rcParams['pdf.fonttype'] = 42
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

SNRlist = np.loadtxt(bayesfilename, usecols = 0)
BFlist = np.loadtxt(bayesfilename, usecols = 1)
lnBFlist = np.loadtxt(bayesfilename, usecols = 2)
lnBFerrlist = np.loadtxt(bayesfilename, usecols = 3)
BFerrlist = np.loadtxt(bayesfilename, usecols = 4)

var1 = 0.5 * np.sqrt(3.0) / (np.pi**2.0 * t_b**2.0 * (t_b - 1)**2.0 * (f_shift*Tobs)*23.0259)
var2 = (1.0 - 5.0 * t_b + 10.0 * t_b**2.0 - 10.0*t_b**3.0 + 5.0 * t_b**4.0)**0.5

var3 = np.sqrt(3.0) / (f_shift*Tobs*23.0259)
var4 = (4.0 * np.pi**4.0 * (1.0 - t_b)**4.0 * t_b**4.0 * (1.0 - 5.0 * t_b + 10.0 * t_b**2.0 - 10.0*t_b**3.0 + 5.0 * t_b**4.0)) * SNRlist**4.0 * (f_shift*Tobs)
var5 = 12.0 * np.pi**2.0 * (t_b - 1) * t_b * (1.0 + 4.0 * (t_b - 1.0) * t_b * (2.0 + 5.0 * (t_b - 1.0) * t_b)) * SNRlist**2.0

O1 = var1/var2 * 2.0 * np.pi / (f_shift*Tobs)
O2 = var3/(np.sqrt(np.abs(var4 + var5))) * 2.0 * np.pi

SNRlist1 = np.logspace(0, 2, 1000)

slope = (1-FF**2.0)
EstimateBF3 = 0.5*slope * SNRlist1**2.0 + np.log(O1) - 2.0 * np.log(SNRlist1)

print(np.log(O1), slope)

p1 = plt.plot(SNRlist1, (EstimateBF3), linewidth=3.0)
p2 = plt.plot(SNRlist1, (0.5 * slope * SNRlist1**2.0), linewidth=3.0)
p3 = plt.plot(SNRlist, (lnBFlist), linewidth=3.0)
plt.axhline(0.5, color='black', linestyle='dashed', linewidth = 3.0)

ax.legend(handles=[p1[0],p2[0],p3[0]],
labels=[r'$\ln \mathcal{B}_{21}$',r'$\ln \mathcal{B}_{21}, \Delta \ln \mathcal{O}=0$',r'$\ln \mathcal{B}_{21}, {RJMCMC}$'],
loc='best',
frameon=False,
fontsize = 22)

fig.suptitle(r'$\gamma = -1.5, t_b = 0.33$', fontsize=28)#, horizontalalignment = "right")

filename = "SNRplot1.pdf"
plt.yscale('log')
plt.xlabel(r'$SNR$',fontsize=26)
plt.ylim(0.3, 30)
plt.xlim(10, 24)
plt.savefig(filename)
plt.close()
