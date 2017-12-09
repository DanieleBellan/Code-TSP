#Read from file and generate the histogram plot and the scatter plot.

import random
import re
import time
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import math
import itertools
from random import randint
from myFunction import euclTsp
from myFunction import distance
from myFunction import remove
from myFunction import greedyPath
from myFunction import optPath
from myFunction import randPath
from myFunction import printPath
from myFunction import printTwoPaths
from myFunction import findMinRow
from myFunction import buildData
from myFunction import optPdp
from math import hypot



X=[]
Y=[]

X=np.r_[X,buildData("X_opt.txt")]
Y=np.r_[Y,buildData("Y_opt.txt")]

m=Y.size
print"size Y", m

m=X.size
print"size X", m

n=np.amax(X) #select which is the maximum number of points for which we have data
n=int(n)

'''
Y=np.reshape(Y,(m, n-1))
X=np.reshape(X,(m, n-1))
'''

x_jitter=np.random.normal(0.1, 0.1, len(X)), #we add jitter to have a better visualization of the scatter plot
        
Xplot=X+x_jitter


meanY=[]
varY=[]
stdY=[]
bins = np.linspace(0, 4, 100)


for i in range(2,n+1):
    indices=np.where(X==i)
    Y_plot=Y[indices]
    #print "Size ", i, " is", Y_plot.size
    meanY.append(np.mean(Y_plot)) e
    varY.append(np.var(Y_plot, dtype=np.float64)) 
    stdY.append(np.std(Y_plot, dtype=np.float64))
    plt.hist(Y_plot, bins, alpha=0.5, label='n=%s'%(i))

plt.xlabel('Cost_values')
plt.ylabel('Number of Occurrences')
plt.title('TSP with ottimal solver')
plt.legend(loc='upper right')
plt.show()

x=np.arange(n+1)
x=x[2:n+1]
x=np.reshape(x,(len(x), -1))
meanY=np.reshape(meanY,(len(meanY), -1))
plt.scatter(Xplot,Y, s=10, c='b', marker='o', alpha=.1) #scatter of the cost read from the file
plt.xlabel('number of points')
plt.ylabel('cost')
for i in np.arange(0,len(meanY),1):
    plt.plot(x[i:i+2],meanY[i:i+2],'k-') #plot of the lines which connects the means for all n
    plt.plot(x[i:i+2],meanY[i:i+2] + stdY[i:i+2],'r--')  #plot of the lines which connects the means+ std deviation for all n
    plt.plot(x[i:i+2],meanY[i:i+2] - stdY[i:i+2],'r--')

black_patch = mpatches.Patch(color='black', label='mean')
red_patch = mpatches.Patch(color='red', label='std deviation')
plt.legend(handles=[black_patch, red_patch])
plt.show()




