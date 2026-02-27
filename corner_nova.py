import argparse
import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import corner
import numpy as np

def get_prob_contour(x0, w, v):
    """
    x0 is the true value (center of the ellipse)
    w is the eigenvalue  (semi-major/minor axes)
    v is the matrix formed by eigenvectors (for rotation)
    """
    nPt=100
    vT=v.transpose()
    y0=np.dot(vT, x0)
    y1=np.linspace(y0[0]-np.sqrt(w[0]), y0[0]+np.sqrt(w[0]), nPt)
    y2=np.zeros([2, nPt], dtype=np.complex128)
    y2[0, :]=y0[1]+np.sqrt(w[1])*np.sqrt(1.+0j-(y1-y0[0])**2./(w[0]))
    y2[1, :]=y0[1]-np.sqrt(w[1])*np.sqrt(1.+0j-(y1-y0[0])**2./(w[0]))
    
    x1=np.zeros([2, nPt])
    x2=np.zeros([2, nPt])
    for i in range(nPt):
        y_temp=np.array([y1[i], np.real(y2[0, i])])
        x_temp=np.dot(v, y_temp)
        x1[0, i]=x_temp[0]
        x2[0, i]=x_temp[1]
        
        y_temp=np.array([y1[i], np.real(y2[1, i])])
        x_temp=np.dot(v, y_temp)
        x1[1, i]=x_temp[0]
        x2[1, i]=x_temp[1]
    return x1, x2
    
def plot_prob_contour(fisher, tv, i, j, \
                      ax=None, label='', **ax_kwargs):
    err_mtrx = np.zeros((2,2))
    
    err_mtrx[0][0] = fisher[i][i]
    err_mtrx[0][1] = fisher[i][j]
    err_mtrx[1][0] = fisher[j][i]
    err_mtrx[1][1] = fisher[j][j]
    w,v=np.linalg.eigh(err_mtrx)
    
    '''
    w = np.zeros(2)
    v = np.zeros((2,2))
    w2, v2 = np.linalg.eigh(fisher)
    w[0] = w2[i]
    w[1] = w2[j]
    v[0][0] = v2[i][i]
    v[0][1] = v2[i][j]
    v[1][0] = v2[j][i]
    v[1][1] = v2[j][j]
    '''
    x1, x2=get_prob_contour(np.array([tv[i], tv[j]]), w, v)
    #if ax==None:
    #    fig=plt.figure(figsize=(6,5))
    #    ax=fig.add_subplot(111)
    #ax.plot(x1[0, :], x2[0, :], label=label, **ax_kwargs)
    #ax.plot(x1[1, :], x2[1, :], **ax_kwargs)
    #fig.savefig("fishertest.png", dpi=300)
    return x1, x2  

data = np.loadtxt("chain.dat", usecols=(2, 3, 4, 5, 6, 7))
tv = np.loadtxt("truth.txt", usecols=(0, 1, 2, 3, 4, 5))
fisher = np.loadtxt("fisher_inv.dat")

mask = data[:, 4] != 0.0  #selects only data points in the burst model, change to == to select only points in the nonburst model
filtered_data = data[mask]

figure = corner.corner(filtered_data, labels=[r'$A$', r'$\phi_0$', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$t_b$'], truths=tv, bins=40, title_kwargs={"fontsize": 14}, label_kwargs={"fontsize": 18}, show_titles=True, title_fmt=".3f", range=[(14,21),(-0.7,0.7),(1224450+5.5,1224450+7.2),(-55,-47),(-3,3),(0,1)])
#corner.overplot_points(figure, x1[0])
#figure.add_subplot((6,6,1))
#ax1 = plt.subplot(442)
#ax1.plot(x1[0,:], x2[0,:])
#ax1.plot(x1[1,:], x2[1,:])

#Plots a single 1sigma distribution from the Fisher matrix
#i = 2
#j = 5
#x1, x2 = plot_prob_contour(fisher, tv, i , j)
#plt.subplot(6,6,i+6*j+1)
#plt.plot(x1[0, :], x2[0, :], color="blue")
#plt.plot(x1[1, :], x2[1, :], color="blue")

print(tv)

i = 0
fish_errors = np.zeros(6)
while i < 6:
    fish_errors[i] = np.sqrt(fisher[i][i])
    i += 1

#Overplots the 1sigma spreads expected from the Fisher matrix as green lines
#print(fish_errors)
#corner.overplot_lines(figure, tv - fish_errors, color = 'green')
#corner.overplot_lines(figure, tv + fish_errors, color = 'green')

#Plots the expected 1sigma distributions from the Fisher matrix
i = 0
while i < 6:
    j = i+1
    while j < 6:
        x1, x2 = plot_prob_contour(fisher, tv, i , j)
        plt.subplot(6,6,i+6*j+1)
        plt.plot(x1[0, :], x2[0, :], color="red")
        plt.plot(x1[1, :], x2[1, :], color="red")
        j += 1
    i += 1

figure.savefig("nova_corner.png", dpi=300)
plt.close()
