import argparse
import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import corner
import numpy as np

#data = np.loadtxt("chain_nova3.dat", usecols=(3, 4, 5, 6))
#tv = np.loadtxt("truth_nova3.txt", usecols=(1, 2, 3))
#mask = data[:, 3] == 0.0
#filtered_data = data[mask]
#data2 = filtered_data[:, 0:3]
#figure = corner.corner(data2, labels=[r'$\phi_0$', r'$\alpha$', r'$\beta$'], truths=tv, bins=40, quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 14}, label_kwargs={"fontsize": 18}, show_titles=True, range = ((-0.15, 0.15), (631160+2.95, 631160+3.1), (-16, -15.5)))

datavals1 = "chain_novahalf1.dat"
datavals2 = "chain_novahalf2.dat"
truth = "truth_novahalf.txt"

data1 = np.loadtxt(datavals1, usecols=(4, 5, 6))
data2 = np.loadtxt(datavals2, usecols=(4, 5, 6))
tv = np.loadtxt(truth, usecols=(2, 3))#*16
gammatrue = np.loadtxt(truth, usecols=(4))
betatrue = np.loadtxt(truth, usecols=(3))
print(tv)

mask1 = data1[:, 2] == 0.0     #filters out points in the nova model
filtered_data1 = data1[mask1]
data1 = filtered_data1[:, 0:2]

mask2 = data2[:, 2] == 0.0     #filters out points in the nova model
filtered_data2 = data2[mask2]
data2 = filtered_data2[:, 0:2]

alphalist1 = filtered_data1[:, 0:1]
betalist1 = filtered_data1[:, 1:2]
alphalist1 = alphalist1 + betalist1
filtered_data1[:, 0:1] = alphalist1

betalist2 = filtered_data2[:, 1:2]
med0 = np.median(betalist2)
med1 = np.median(betalist1)
std0 = np.std(betalist2)
std1 = np.std(betalist1)

alphalist2 = filtered_data2[:, 0:1]
med2 = np.median(alphalist2)
med3 = np.median(alphalist1)
std2 = np.std(alphalist2)
std3 = np.std(alphalist1)

print(tv[0], tv[1], med1, std0, np.abs(med1-tv[1])/std1, np.abs(med2-tv[0])/std2)
print(np.abs(med2-med3)/std2, np.abs(med2-med3)/std3)
print(np.abs(med0-med1)/std1, np.abs(med0-med1)/std0)
print(std0, std1, std2, std3)

figure = corner.corner(data2, labels=[r'$\alpha$', r'$\beta$'], color = 'red', bins=40, quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 14}, label_kwargs={"fontsize": 18}, show_titles=False)#, range = ((612210+2.5, 612210+6.5), (-14.7, -10.6)))
figure = corner.corner(data1, labels=[r'$\alpha$', r'$\beta$'], color = 'black', bins=40, quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 14}, label_kwargs={"fontsize": 18}, show_titles=False, fig = figure)#, range = ((612210+2.5, 612210+6.5), (-14.7, -10.6)))

figure.suptitle(r'$\gamma = -2, t_b = 0.45, SNR = 12.7, \mathcal{B}_{21} = 0.67, \Delta\alpha = 3.3$', fontsize=14)#, horizontalalignment = "right")

#plt.subplots_adjust(bottom=0.16, right=0.9, top=0.92)
plt.subplots_adjust(bottom=0.16, right=0.9, top=0.87)

figure.savefig("nova_bias.png", dpi=300)
plt.close(figure)

