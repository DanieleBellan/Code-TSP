import random
import numpy as np
import matplotlib.pyplot as plt 
import math
from myFunction import euclTsp
from myFunction import distance
from myFunction import remove
from myFunction import greedyPath
from myFunction import optPath
from myFunction import printPath
from myFunction import printTwoPaths
from myFunction import findMinRow
from math import hypot
import itertools
#Variables
e=10;#events: number of points

mat=np.random.poisson(6,2) #We generete the set of points
for x in range(e):
    t=np.random.poisson(7,2)
    mat=np.r_[mat,t]  #concatenate matrix

mat=np.reshape(mat, (e+1,-1)) #reshape the matrix as Nx2
print mat;
New=euclTsp(mat)
size=np.shape(mat);
print "These are the distances \n", New

[path,route]=greedyPath(mat)
print "The greedy path is long", path,"with route ",route
[opt_path,route_opt]=optPath(mat)
print "The shortest path is long", opt_path,"with route ", route_opt
#printPath(mat, route)
#printPath(mat,route_opt)
printTwoPaths(mat,route,route_opt)



