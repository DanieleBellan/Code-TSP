#With this script we generate the data, and we store it in a file.

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
from optProblem import TSP
from math import hypot



m=40 #size of data set
n=14


#Generate Input
X=[]
Y=[]

for j in range(12,n):
    for i in range(0,m):
        length_path=TSP(j)
        X.append(j)
        Y.append(length_path)

file = open("X_opt.txt","a")
file.write('%s' %X) 
file.close
file = open("Y_opt.txt","a")
file.write('%s' %Y) 
file.close


