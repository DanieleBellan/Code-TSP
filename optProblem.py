#Optimization problems

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
from math import hypot

from pulp import *

def TSP(n):
	Points=np.random.rand(n,2)
	a=[]#np.zeros(shape=(n,n))
	prob = LpProblem("The TSP",LpMinimize)
	a = {}
	for i in range(n+1):
		for j in range(n+1):
            		lowerBound = 0
            		upperBound = 1

            # Forbid loops
            		if i == j:
                		upperBound = 0
            		a[i,j] = pulp.LpVariable('a' + str(i) + '_' + str(j), lowerBound, upperBound, LpInteger ) #pulp.LpBinary
            # x[j,i] = x[i,j]

	Adj=euclTsp(Points) #build the adjacency matrix

#Remove the zero (i.e. the element on the diagonal) of the Adj matrix
	#A_cost=np.reshape(Adj,(n*n,1))
	#A_cost=[elem for elem in A_cost if elem!=0]
	#A_cost=np.asarray(A_cost)


	prob +=pulp.lpSum([Adj[i][j] * a[i,j] for i in range(0,n) for j in range(0,n)]) #objective function


# constraints

	for i in range(0,n):
		prob += pulp.lpSum([a[i,j] for j in range(n)]) <= 1  #each row must have at maximum one element equal to one
	for j in range(0,n):
		prob += pulp.lpSum([a[i,j] for i in range(n)]) <=1 #each row must have at maximum one element equal to one

	for i in range(n):
		prob += pulp.lpSum([a[i,j] for j in range(0,n)]) + pulp.lpSum([a[j,i] for j in range(0,n)]) >= 1 #at least one element on the row and the column of the node i must be 1

	sum_along_columns=0
	for i in range(0,n):
		for j in range(n):
			sum_along_columns = sum_along_columns+a[i,j]
	prob += sum_along_columns == n-1 #The sum of the elements column by column must be n-1
	sum_along_rows=0
	for j in range(0,n):
		for i in range(n):
			sum_along_rows = sum_along_rows+a[i,j]
	prob += sum_along_rows == n-1 #The sum of the elements row by row must be n-1

	for i in range(0,n):
		for j in range(n):
			prob += a[j,i]+a[i,j] <=1 #only one direction is allowed
	Total=0
	for i in range(0,n):
		for j in range(n):
			Total= Total+a[i,j]   
	prob += Total == n-1 #the maximum number of connecting edges must be n-1


	u = []

	for i in range(n):
    		u.append(pulp.LpVariable('u_' + str(i), cat='Integer'))

	for i in range(n):
    		for j in range(0, n):
        		prob += pulp.lpSum([ u[i] - u[j] + n*a[i,j]]) <= n-1 #avoid subtours.

	for i in range(n):
		for j in range(n):
			for k in range(n):
				prob += a[i,j]+a[j,k]+a[k,i]<=2 #avoid cyclic


	prob.writeLP("TSP.lp")

	prob.solve()
	'''
	print("Status:", LpStatus[prob.status])

	for v in prob.variables():
    		print(v.name, "=", v.varValue)

	print "Optimal with my function", optPath(Points)[0]

	print("Optimal with the solver = ", value(prob.objective))

	'''
	return value(prob.objective)

'''
n=5
Points=np.random.rand(n,2)
a=[]#np.zeros(shape=(n,n))
prob = LpProblem("The TSP",LpMinimize)
a = {}
for i in range(n+1):
	for j in range(n+1):
            lowerBound = 0
            upperBound = 1

            # Forbid loops
            if i == j:
                upperBound = 0
                # print i,i
            a[i,j] = pulp.LpVariable('a' + str(i) + '_' + str(j), lowerBound, upperBound, LpInteger ) #pulp.LpBinary
            # x[j,i] = x[i,j]
#a = np.asarray(a)
#a=np.reshape(a, (1,len(a)))
Adj=euclTsp(Points)

#Remove the zero (i.e. the element on the diagonal) of the Adj matrix
#A_cost=np.reshape(Adj,(n*n,1))
#A_cost=[elem for elem in A_cost if elem!=0]
#A_cost=np.asarray(A_cost)


prob +=pulp.lpSum([Adj[i][j] * a[i,j] for i in range(0,n) for j in range(0,n)]) #sum(a[0,i]*A_cost[i] for i in range(1,n*n-n))#cost #objective function


# constraints

for i in range(0,n):
	prob += pulp.lpSum([a[i,j] for j in range(n)]) <= 1 # sum(a[0,i]for i in range(i,i+n)) ==1, ' sum along ' + str(i) 
for j in range(0,n):
	prob += pulp.lpSum([a[i,j] for i in range(n)]) <=1 # sum(a[0,i]for i in range(i,i+n)) ==1, ' sum along ' + str(i) 

sum_along_columns=0
for i in range(0,n):
	for j in range(n):
		sum_along_columns = sum_along_columns+a[i,j]
prob += sum_along_columns == n-1
sum_along_rows=0
for j in range(0,n):
	for i in range(n):
		sum_along_rows = sum_along_rows+a[i,j]
prob += sum_along_rows == n-1
for i in range(0,n):
	for j in range(n):
		prob += a[j,i]+a[i,j] <=1
Total=0
for i in range(0,n):
	for j in range(n):
		Total= Total+a[i,j]   # sum(a[0,i]for i in range(i,i+n)) ==1, ' sum along ' + str(i) 
prob += Total == n-1
s=[]
t=0
for i in range(n):
	prob += pulp.lpSum([a[i,j] for j in range(0,n)]) + pulp.lpSum([a[j,i] for j in range(0,n)]) >= 1

u = []

for i in range(n):
    u.append(pulp.LpVariable('u_' + str(i), cat='Integer'))

for i in range(n):
    for j in range(0, n):
        prob += pulp.lpSum([ u[i] - u[j] + n*a[i,j]]) <= n-1 #Cosi non fa cerchi ( o subtour) ma non so perche'.

for i in range(n):
	for j in range(n):
		for k in range(n):
			prob += a[i,j]+a[j,k]+a[k,i]<=2


prob.writeLP("TSP.lp")

prob.solve()

print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print "Ottimo col mio metodo", optPath(Points)[0]

print("ottimo solver = ", value(prob.objective))

#return value(prob.objective)


'''
