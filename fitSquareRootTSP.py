import random
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
from optProblem import TSP
from math import hypot
from geometricFunction import minimum_bounding_rectangle
from geometricFunction import area


#scikit functions
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sknn.mlp import Regressor, Layer
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.svm import SVR
from sklearn.externals import joblib

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import scipy 
from scipy import optimize
from scipy.optimize import curve_fit

X=[]
Y=[]
X=np.r_[X,buildData("X_opt.txt")]
Y=np.r_[Y,buildData("Y_opt.txt")]
'''
indices=np.where(X==12)
X = np.delete(X, indices)
Y = np.delete(Y, indices)

X = np.delete(X, indices)
Y = np.delete(Y, indices)
'''
'''
indices=np.where(X==13)
X12=X[indices]
Y12=Y[indices]
X = np.delete(X, indices)
Y = np.delete(Y, indices)

'''
m=Y.size
X=np.reshape(X, (m, -1))

Y=np.reshape(Y, (m, -1))


X_train=X
Y_train=Y
meanY_trainData=[]
varY_trainData=[]
stdY_trainData=[]
n=np.amax(X)
n=int(n)

def vaf(y_real, y_pred):
    N=y_real.size
    
    for i in range(0,N):
        sum1=(y_real[i]-y_pred[i])*(y_real[i]-y_pred[i])
        sum2=(y_real[i])*(y_real[i])
    return max(0, (1- (sum1/N)/(sum2/N))*100)
    

def func(x, a, b): #square root function
    return a * np.sqrt(x) + b
'''
yn=func(X, 0.5,0.2)
modelsr=curve_fit(func, X, Y.ravel())#
'''

popt, pcov = curve_fit(func, X.ravel(), Y.ravel())
#popt, pcov =scipy.optimize.curve_fit(lambda a,b: a+b*np.sqrt(X),  X.ravel(),  Y.ravel())

#Evaluating of the goodness of the fitness
m=700
X=[]
Y=[]
y_pred=[]
for i in range(0,m):
    j=randint(2, 12)
    length_path=TSP(j)
    X=np.r_[X,j]
    #t=np.reshape(mat, (e,-1))
    Y=np.r_[Y, (length_path)]
    y_pred=np.r_[y_pred, func(j, *popt)]

meanY=[]
varY=[]
stdY=[]
n=np.amax(X)
n=int(n)
for i in range(2,n+1):
    indices=np.where(X_train==i)
    Y_plot=Y_train[indices]
    #print "Size ", i, " is", Y_plot.size
    meanY_trainData.append(np.mean(Y_plot, dtype=np.float64)) 
    varY_trainData.append(np.var(Y_plot, dtype=np.float64))
    stdY_trainData.append(np.std(Y_plot, dtype=np.float64))

meanY_trainData=np.reshape(meanY_trainData,(len(meanY_trainData), -1))
for i in range(2,n+1):
    indices=np.where(X==i)
    Y_plot=Y[indices]
    #print "Size ", i, " is", Y_plot.size
    meanY.append(np.mean(Y_plot, dtype=np.float64)) #facciamo un vettore di medie e un vettore di varianze
    varY.append(np.var(Y_plot, dtype=np.float64)) 
    stdY.append(np.std(Y_plot, dtype=np.float64))


print " VAF SR",vaf(Y,y_pred)


'''
Y=np.reshape(Y,(m, n-1))
X=np.reshape(X,(m, n-1))
'''

x_jitter=np.random.normal(0.1, 0.1, len(X)), 
        
Xplot=X+x_jitter


x=np.arange(n+1)
x=x[2:n+1]
x=np.reshape(x,(len(x), -1))
meanY=np.reshape(meanY,(len(meanY), -1))
print "size x ", x.size,"size y", meanY_trainData.size

plt.scatter(Xplot,Y, s=10, c='b', marker='o', alpha=.1, label='true value')
plt.scatter(X,y_pred, s=10, c='g', marker='o', alpha=.4,  label='predicted square root value')
first_legend = plt.legend( loc='upper_left')

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
plt.xlabel('number of points')
plt.ylabel('cost')
#plt.legend(loc='upper_left')
for i in np.arange(0,len(meanY),1):
    plt.plot(x[i:i+2],meanY[i:i+2],'k:')
    plt.plot(x[i:i+2],meanY_trainData[i:i+2],'m-.')
    plt.plot(x[i:i+2],meanY[i:i+2] + stdY[i:i+2],'r--')
    plt.plot(x[i:i+2],meanY[i:i+2] - stdY[i:i+2],'r--')
#for i in range(1,n-3):
 #   plt.plot(x[i], meanY[i], x[i+1], meanY[i+1], 'ro-')


mag_patch = mpatches.Patch(color='magenta', label='mean training data')
red_patch = mpatches.Patch(color='red', label='mean test data +/- std dev')
black_patch = mpatches.Patch(color='black', alpha=.4, label='mean testing data ')
plt.legend(loc=4,handles=[ black_patch, red_patch,mag_patch])
plt.show()

