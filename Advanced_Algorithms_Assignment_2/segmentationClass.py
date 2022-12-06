#Max Casteel
#11/25/2022
#Software Assignment 2
#Advanced Algorithms - Dr. Jordan Malof - Fall 2022
#implements Ford-Fulkerson algorithm to find max flow in a graph

import sys
import math
import time
import random
import copy
import numpy as np
import timeit
import time
import os
import csv
import statistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import segmentationClass as sc

#Prompt - Implement (i) Ford-Fulkerson algorithm, and then use it in (ii) an implementation
#of the segmentation algorith described in section 7.10 of the kleinberg and Tardos book.
#For full credit, you must implement the algorithm as described in Section 7.10 of the Kleinberg and Tardos book.

#Let 'x' \in\mathbb{N^2} be row/column coordinates of a pixel, and I(x) \in\mathbb{N^3} be RGB values of a pixel at x.
#Per the implementation in section 7.10, each pixel is treated as a graph and must have weights for (i), the likelihood of that pixel being in the 
#background, denoted by b(x).  (ii), the likelihood of that pixel being in the forground
#denoted here by a(x).  (iii), and the weights being neighboring pixels, denoted p(x_i,x_j), where x_i and x_j are the coordinates
#of two pixels, i and j.  You must set these values in the following way:
#a(x) = 442 - d(x,x_a)
#b(x) = 442 - d(x,x_b)
#p(x_i,x_j) = p_0, x(x_i,x_j) < 2 
#p(x_i,x_j) = 0, otherwise

#In these equations d(x,y) is the Euclidean distance between two input vectors,
#and the parameters x_b, x_a \in\mathbb{N^2} are coordinates of one background andm one foreground pixel,
#respectively.  (i.e. imagine a user manually choosing one point in the foreground and one in the background.)

#You can only utilize the following primitive python data structures: python lists, python dictionaries, and numpy arrays.
#You can only utilize primitive python and numpy functions, such as sum, multiplication,
#division, exponentation, logarrithms, etc.  One exception is that you may use someone elses implementation
#of breadth-first search or depth-first search for use in the Ford-Fulkerson Alorithm.

#Your class must have the following specifications that can be set by the user:
#segmentationClass.p0 - an integer value greater than one.  This parameter will be used as p_0 in equation (iii).
#segmentationClass.x_a - a 1x2 numpy array specifying the coordinate (roa and column)
#segmentationClass.x_b - a 1x2 numpy array specufying the coordinate (row and column) of a location in the image representing the background.

#Your class must have the following methods:
#obj = segmentationClass()
#L = segmentImage(I)

#obj = segmentationClass()
    #Input: None
    #Output: An object instantiated from your class.  Your class should have the name as segmentationClass.
class segmentationClass:

    def __init__(self):
        print("Initializing segmentationClass object")

    

    #L = segmentImage(I)
    #Input: I is an NxNx3 numpy array representing a color (RGB) image.
    #Each pixel intensity will be an integer value between 0 and 255.
    #Output: L is an NxNx1 numpy array of binary values representing whether each pixel 
    #in the image is in the foreground (1) or background (0).  So if pixel
    #at row (i,j) is in the foreground then L[i,j] = 1, otherwise L[i,j] = 0.
    def segmentImage(self,I):
        print("Segmenting image")
        
        #initialize variables
        self.I = I
        self.N = I.shape[0]
        self.p0 = 100
        self.x_a = np.array([0,0])
        self.x_b = np.array([self.N-1,self.N-1])
        self.a = np.zeros((self.N,self.N))
        self.b = np.zeros((self.N,self.N))
        self.p = np.zeros((self.N,self.N))
        self.L = np.zeros((self.N,self.N))
        self.G = np.zeros((self.N,self.N))
        self.G = self.G.astype(int)
        self.G = self.G.tolist()
        self.G = np.array(self.G)
        self.G = self.G.astype(int)
        self.G = self.G.tolist()

        #set a(x) and b(x) values
        for i in range(self.N):
            for j in range(self.N):
                self.a[i,j] = 442 - self.distance(self.x_a,np.array([i,j]))
                self.b[i,j] = 442 - self.distance(self.x_b,np.array([i,j]))

        #set p(x_i,x_j) values
        for i in range(self.N):
            for j in range(self.N):
                if self.distance(np.array([i,j]),self.x_a) < 2 or self.distance(np.array([i,j]),self.x_b) < 2:
                    self.p[i,j] = self.p0
                else:
                    self.p[i,j] = 0

        #set G values
        for i in range(self.N):
            for j in range(self.N):
                if i == 0:
                    self.G[i,j] = self.a[i,j]
                elif i == self.N-1:
                    self.G[i,j] = self.b[i,j]
                else:
                    self.G[i,j] = self.p[i,j]

        #set source and sink values
        self.source = self.N*self.N
        self.sink = self.N*self.N + 1

        
        #set up graph
        self.G = np.append(self.G,np.zeros((self.N,2)),axis=1)
        self.G = np.append(self.G,np.zeros((2,self.N+2)),axis=0)
        self.G[self.source,0:self.N] = self.a[0,:]
        self.G[self.N-1,self.N:self.sink] = self.b[self.N-1,:]
        self.G[self.source,self.sink] = 1000000
        self.G[self.sink,self.source] = 1000000

        #run Ford-Fulkerson algorithm
        self.maxFlow = self.fordFulkerson(self.G,self.source,self.sink)

        #set L values
        for i in range(self.N):
            for j in range(self.N):
                if self.G[i,j] == 0:
                    self.L[i,j] = 1
                else:
                    self.L[i,j] = 0

        return self.L


    #distance function
    def distance(self,x,y):
        return np.sqrt(np.sum((x-y)**2))

    #breadth-first search
    def breadthFirstSearch(self,rGraph,s,t,path):
        visited = [False] * (self.sink+1)
        queue = []
        queue.append(s)
        visited[s] = True

        while queue:
            u = queue.pop(0)

            for ind, val in enumerate(rGraph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    path[ind] = u

        return True if visited[t] else False

    #Ford-Fulkerson algorithm
    def fordFulkerson(self,graph,s,t):
        u = 0
        v = 0

        rGraph = [[0 for column in range(self.sink+1)] for row in range(self.sink+1)]

        for u in range(self.sink+1):
            for v in range(self.sink+1):
                rGraph[u][v] = graph[u][v]

        path = [-1] * (self.sink+1)
        maxFlow = 0

        while self.breadthFirstSearch(rGraph,s,t,path):
            pathFlow = float("Inf")
            s = self.sink

            while(s != self.source):
                pathFlow = min(pathFlow,rGraph[path[s]][s])
                s = path[s]

            maxFlow += pathFlow
            v = self.sink

            while(v != self.source):
                u = path[v]
                rGraph[u][v] -= pathFlow
                rGraph[v][u] += pathFlow
                v = path[v]

        return maxFlow

#end of segmentationClass class

