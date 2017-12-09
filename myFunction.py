import random
import re
import time
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import math
import itertools
from random import randint

from math import hypot
from geometricFunction import minimum_bounding_rectangle
from geometricFunction import area
def distance(p1,p2): #compute the euclidean distance between two points
    return math.hypot(p1[0]-p2[0],p2[1]-p1[1]) 

def euclTsp(Points): #Compute the euclidean distance given a certain number of points: Build the Adjacency matrix
    size=np.shape(Points)  
    Dist=[] 
    for i in range(0,max(size)):
        temp=np.delete(Points,i,axis=0)
        for j in range(0,max(size)):    
	    Dist.append(np.matrix(distance(Points[i,:], Points[j,:]))) #this is a symmetric adjacency matrix, with zero on diagonal.
    Dist=np.reshape(Dist, (max(size), max(size)))
    return Dist   

def remove(Vector, element): # remove the element "element" from the vector "Vector"
    Vector=np.delete(Vector, np.where(Vector == element))
    #Vector=np.array(list(itertools.compress(Vector, [i!=element for i in range(len(Vector))])))
    return Vector

def findMinRow(Matrix, row): #Given a matrix and the row where you want to find the minimum, it returns the value of the minimum and the position along the row. IT DOES NOT CONSIDER ALL THE EQUAL VALUES. IT TAKES THE FIRST ENCOUNTERED MINIMUM
    min_row=Matrix.min(axis=1)
    pos_min_row=Matrix[row,:].argmin()
    return min_row[row], pos_min_row

def greedyPath(Points): #very greedy approach: select the shortest distance from each point.
    n = max(np.shape(Points))
    Adj=euclTsp(Points)
    Q=np.arange(n) #generate a vector with the nodes 1,2,...n
    shortest = 100000 #np.finfo(np.float128)
    path=0
    j=0
    route=[]
    Adj_mod=Adj+100*np.identity(n) #Modify the adjacency matrix due to the zero on the diagonal
    #print Adj_mod
    for i in range(0,n):
        pos=[]
        pos.append(i)
	Q_temp=remove(Q,i)
	j=i
	while j>=0 and j<n and len(Q_temp)!=0:              
              Adj_mod[:,j]=100*np.full((1,n),1) # weight extensively the column j: it won't be considered in the next iterations
	      [min_row,pos_min]=findMinRow(Adj_mod,j)
	      pos.append(pos_min)
              path=path+min_row
	      j=pos_min
	      Q_temp=remove(Q_temp,j)
              if path>shortest: #if for some reason the path we are building is bigger than the best found until now, we exits from the cycle
                 j=n+1 
              if len(Q_temp)==1 and j<n: #last iteration
		 [min_row,pos_min]=findMinRow(Adj_mod,j)
	      	 pos.append(pos_min)
                 path=path+min_row
                 j=n+1 #exit from the cycle
              #print "temporaty path", path, "temporaty shortest", shortest
	if path<shortest: 
           route=[]
    	   for i in range(0,n):
               route.append(Points[pos[i],:])
           route=np.reshape(route, (n,-1))
	   #route=pos
    	shortest = min(shortest, path)
        path=0
        Adj_mod=Adj+1000000000*np.identity(n) 
    return shortest, route

def optPath(Points): #find the optimal path. DO NOT USE WITH GREAT(>10) number of points
    [greedy_cost,greedy_route]=greedyPath(Points) #we use the greedy result as the starting points
    n = max(np.shape(Points))
    Adj=euclTsp(Points)
    Q=np.arange(n) #generate a vector with the nodes 1,2,...n
    shortest = greedy_cost
    path=0
    j=0
    route=greedy_route
    perm=itertools.permutations(Q)
    for p in perm:
        for i in range(0,n-1):
	    path=path+Adj[p[i],p[i+1]]
            if path>shortest:
               i=n+1
	if path<shortest:
           shortest=path 
           route=[]
	   for i in range(0,n):
               route.append(Points[p[i],:])
           route=np.reshape(route, (n,-1))#route.append(p)
           #route=np.reshape(route, (n,-1))
        path=0
    return shortest, route

def randPath(Points): #find the optimal path using random approach
    [greedy_cost,greedy_route]=greedyPath(Points) #we use the greedy result as the starting points
    n = max(np.shape(Points))
    #Adj=euclTsp(Points)
    shortest = greedy_cost
    path=0
    j=0
    route=greedy_route
    start_time = time.time()
   # Already_had_permutations=[]
    while (time.time()-start_time)<30 :
              #path=path+distance(Points[random_permutation[i]],Points[random_permutation[i+1]])
              #p= np.delete(p, indexes, axis=0) QUalcosa per evitare le ripetizioni
    	  np.random.shuffle(Points)
	  for i in range(0,n-1):
	      path=path+distance(Points[i],Points[i+1])
              if path>shortest:
                 i=n+1
	  if path<shortest:
             shortest=path
             route=[]
 	     route.append(Points)
             route=np.reshape(route, (n,-1))
          path=0
    return shortest, route          

def printPath(Points,route):
    x = Points[:,0]
    y = Points[:,1]
    n=len(x)
    plt.scatter(x,y)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    for i,j in zip(x,y):
        plt.annotate((i,j), xy=(i,j) )
    print "Route is",
    for i in range(0,len(route)):
        print route[i], 
        if i!=len(route)-1: 
           print "->",
    print "\n"
    '''
    RoutePoints=[] #Build the route as a succession of points
    for i in range(0,len(route)):
        RoutePoints.append([x[route[i]],y[route[i]]])
    RoutePoints=np.reshape(RoutePoints, (n,-1)) #reshape the matrix as Nx2
    print RoutePoints
    '''
    x=route[:,0]
    y=route[:,1]
    k=0
    temp=0
    for i,j in zip(x,y):
         if k<len(x) and k!=0:
     	    plt.annotate('', xytext=temp, xy=(i,j), arrowprops=dict(facecolor='blue', width=0.1, headwidth=5),)
         k=k+1
         temp=(i,j)
    plt.show() 

def printTwoPaths(Points,route1,route2):
    x = Points[:,0]
    y = Points[:,1]
    n=len(x)
    plt.scatter(x,y)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    for i,j in zip(x,y):#range(0,len(x)):
        plt.annotate((i,j), xy=(i,j) )
    x= route1[:,0]
    y=route1[:,1]
    k=0
    temp=0
    for i,j in zip(x,y):
         if k<len(x) and k!=0: 
     	    plt.annotate('', xytext=temp, xy=(i,j), arrowprops=dict(facecolor='blue', width=0.1, headwidth=5, linestyle='dashed', color='blue'),)
         k=k+1
         temp=(i,j)
    x = Points[:,0]
    y = Points[:,1]
    n=len(x)
    plt.scatter(x,y)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    for i,j in zip(x,y):#range(0,len(x)):
        plt.annotate((i,j), xy=(i,j) )
    x= route2[:,0]
    y=route2[:,1]
    k=0
    temp=0
    for i,j in zip(x,y):
         if k<len(x) and k!=0:
     	    plt.annotate('', xytext=temp, xy=(i,j), arrowprops=dict(facecolor='red', width=0.08, headwidth=5, color='red'),)
         k=k+1
         temp=(i,j)
    red_patch = mpatches.Patch(color='red', label='Optimal path')
    blue_patch = mpatches.Patch(color='blue', label='Greedy path')
    plt.legend(handles=[red_patch,blue_patch])
    plt.show() 

def buildData(filename): #read the files and build the vector
    X=[]
    int_list=[]
    file = open(filename, "r") 
    f= file.read()
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", f)
    int_list= [float(i) for i in numbers]
    X = np.r_[X,int_list]
    file.close
    return X

def optPdp(Points): 
    n = len(Points) #Points supposed to be nx4 Points[0:1] origin, Points [2:3] destination. 
    origin=Points[:,0:2]
    destination=Points [:,2:4]
    Adj=euclTsp(np.concatenate((origin,destination),axis=0))
    shortest = 100000 #np.finfo(np.float128)
    path=0
    j=0
    route=[]
    Q=np.arange(2*n)
    perm=itertools.permutations(Q)
    for p in perm:
    	if p[0]>n-1 :
		break;
	for i in range(0, 2*n-1):
        	if (p[i]>n and i<p.index(p[i]-n)) or (p[i]==n and i<p.index(p[i]-n)):
               		i=2*n+1
			path=10000
			
			
                else:   
			path=path+Adj[p[i],p[i+1]]
		        if path>shortest:
               		   i=2*n+1
	if path<shortest and path>0:
           	shortest=path 
           	route=p
        path=0
    return shortest, route



   
