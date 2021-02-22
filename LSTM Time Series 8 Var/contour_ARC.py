
import pandas as pd
import pandas
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
plt.switch_backend('agg')




df = pd.read_csv(open('contour2.csv', newline=''), sep=',')
x = np.array(df.iloc[:,1].values.tolist())
y = np.array(df.iloc[:,2].values.tolist())
z0 = np.array(df.iloc[:,3].values.tolist())
z1 = np.array(df.iloc[:,4].values.tolist())
z2 = np.array(df.iloc[:,5].values.tolist())
z5 = np.array(df.iloc[:,6].values.tolist())
z10 = np.array(df.iloc[:,7].values.tolist())
# print (z0,z1,z2,z5,z10)
def plot_contour(x,y,z,resolution = 50,contour_method='linear', ax=None):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(x):max(x):complex(resolution),   min(y):max(y):complex(resolution)]
    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method)
    return X, Y, Z

X,Y,Z_0 = plot_contour(x,y,z0,resolution = 50,contour_method='linear')
X,Y,Z_1 = plot_contour(x,y,z1,resolution = 50,contour_method='linear')
X,Y,Z_2 = plot_contour(x,y,z2,resolution = 50,contour_method='linear')
X,Y,Z_5 = plot_contour(x,y,z5,resolution = 50,contour_method='linear')
X,Y,Z_10 = plot_contour(x,y,z10,resolution = 50,contour_method='linear')

cmap1 = plt.get_cmap('binary')

fig, _axs = plt.subplots(2, 3, constrained_layout=True)
# fig.subplots_adjust(hspace=0.3)
axs = _axs.flatten()


cs1=axs[0].contourf(X, Y, Z_0, cmap=cmap1)
axs[0].set_title('Base Year')
axs[0].set_ylabel('Median household income \n (2018 inflation adjusted dollars)$',fontsize=10)
axs[0].set_xlabel('Percent Damage (%)',fontsize=10)
axs[0].tick_params(labelsize=10)
# axs[0].set_aspect('equal')
fig.colorbar(cs1,ax=axs[1],label='Percent change in \n housing units(%)')
#
#
cs2=axs[1].contourf(X,Y,Z_1, cmap=cmap1)
axs[1].set_title("Year 1")
axs[1].set_xlabel('Percent Damage (%)',fontsize=10)
# axs[1].set_aspect('equal')
# fig.colorbar(cs2,ax=axs[1],label='Percent change in \n housing units(%)')

cs3=axs[2].contourf(X,Y,Z_2, cmap=cmap1)
axs[2].set_title("Year 2")
# axs[2].set_ylabel(r'Median household income\(2018 inflation adjusted dollars)$',fontsize=10)
axs[2].set_xlabel('Percent Damage (%)',fontsize=10)
axs[2].tick_params(labelsize=10)
# axs[2].set_aspect('equal')
# fig.colorbar(cs3,ax=axs[2],label='Percent change in \n housing units(%)')


cs4=axs[3].contourf(X,Y,Z_5, cmap=cmap1)
axs[3].set_title("Year5")
axs[3].set_ylabel('Median household income \n (2018 inflation adjusted dollars)$',fontsize=10)
axs[3].set_xlabel('Percent Damage (%)',fontsize=10)
# axs[3].set_aspect('equal')
# fig.colorbar(cs4,label='Percent change in \n housing units(%)')


cs5=axs[4].contourf(X,Y,Z_10, cmap=cmap1)
axs[4].set_title("Year 10")
# axs[4].set_ylabel(r'Median household income\(2018 inflation adjusted dollars) $',fontsize=10)
axs[4].set_xlabel('Percent Damage (%)',fontsize=10)
axs[4].tick_params(labelsize=10)
# axs[4].set_aspect('equal')
# fig.colorbar(cs5,ax=axs[4],label= 'Percent change in \n housing units(%)')

plt.tight_layout()
plt.savefig('contour7.png', dpi=300)
plt.show()
