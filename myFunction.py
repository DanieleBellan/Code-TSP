import random
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import math
import itertools
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
	   route=pos
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
	   route=p
        path=0
    return shortest, route

def printPath(Points,route):

    x = Points[:,0]
    y = Points[:,1]
    n=len(x)
#ax.set_ylim(0,10)
    plt.scatter(x,y)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    for i,j in zip(x,y):#range(0,len(x)):
        plt.annotate((i,j), xy=(i,j) )
#plt.show()
    print "Route is",
    for i in range(0,len(route)):
        print route[i], 
        if i!=len(route)-1: 
           print "->",
    print "\n"
    RoutePoints=[] #Build the route as a succession of points
    for i in range(0,len(route)):
        RoutePoints.append([x[route[i]],y[route[i]]])
    RoutePoints=np.reshape(RoutePoints, (n,-1)) #reshape the matrix as Nx2
    print RoutePoints
    x= RoutePoints[:,0]
    y=RoutePoints[:,1]
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
    RoutePoints=[] #Build the route as a succession of points
    for i in range(0,len(route1)):
        RoutePoints.append([x[route1[i]],y[route1[i]]])
    RoutePoints=np.reshape(RoutePoints, (n,-1)) #reshape the matrix as Nx2
    x= RoutePoints[:,0]
    y=RoutePoints[:,1]
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
    RoutePoints=[] #Build the route as a succession of points
    for i in range(0,len(route2)):
        RoutePoints.append([x[route2[i]],y[route2[i]]])
    RoutePoints=np.reshape(RoutePoints, (n,-1)) #reshape the matrix as Nx2
    x= RoutePoints[:,0]
    y=RoutePoints[:,1]
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





   
