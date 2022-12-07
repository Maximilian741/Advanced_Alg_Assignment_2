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
    #We are working with adjacency matrixes in this implementation, so we will use a 2D array to represent the graph.
    def segmentImage(self,I):
        #get the size of the image
        self.size = I.shape[0]
        #set the source node
        self.source = self.size**2
        #set the sink node
        self.sink = self.size**2+1
        #set the number of nodes
        self.numNodes = self.size**2+2
        #set the number of edges
        self.numEdges = 2*(self.size**2) + 2*self.size
        #set the p0 value
        self.p0 = 2
        #set the x_a value
        self.x_a = np.array([0,0])
        #set the x_b value
        self.x_b = np.array([self.size-1,self.size-1])
        #create the graph
        self.graph = np.zeros((self.numNodes,self.numNodes))
        #set the source edges
        for i in range(self.size):
            for j in range(self.size):
                self.graph[self.source][i*self.size+j] = 442 - self.distance(np.array([i,j]),self.x_a)
        #set the sink edges
        for i in range(self.size):
            for j in range(self.size):
                self.graph[i*self.size+j][self.sink] = 442 - self.distance(np.array([i,j]),self.x_b)
        #set the edges between nodes
        for i in range(self.size):
            for j in range(self.size):
                #set the edges to the right
                if j < self.size-1:
                    self.graph[i*self.size+j][i*self.size+j+1] = self.p0
                #set the edges to the left
                if j > 0:
                    self.graph[i*self.size+j][i*self.size+j-1] = self.p0
                #set the edges to the top
                if i > 0:
                    self.graph[i*self.size+j][(i-1)*self.size+j] = self.p0
                #set the edges to the bottom
                if i < self.size-1:
                    self.graph[i*self.size+j][(i+1)*self.size+j] = self.p0
        #run the Ford-Fulkerson algorithm
        maxFlow = self.fordFulkerson(self.graph,self.source,self.sink)
        #create the output array
        L = np.zeros((self.size,self.size))
        #set the output array
        for i in range(self.size):
            for j in range(self.size):
                if self.graph[i*self.size+j][self.sink] == 0:
                    L[i][j] = 1
        return L
        

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
            for ind,val in enumerate(rGraph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    path[ind] = u
                    visited[ind] = True

        return True if visited[t] else False

    #Ford-Fulkerson algorithm
    def fordFulkerson(self,graph,s,t):
        rGraph = graph
        path = [-1] * (self.sink+1)
        maxFlow = 0

        while self.breadthFirstSearch(rGraph,s,t,path):
            pathFlow = float("Inf")
            s = t
            while(s != self.source):
                pathFlow = min(pathFlow,rGraph[path[s]][s])
                s = path[s]

            maxFlow += pathFlow
            v = t
            while(v != self.source):
                u = path[v]
                rGraph[u][v] -= pathFlow
                rGraph[v][u] += pathFlow
                v = path[v]

        return maxFlow

    #Create an adjacency list from image
    #This method takes an mage as input and returns an adjacency list( python dictionary).  This method is used inside 
    #the segmentImage method.  This adjacency list will be turned into a matrix in the adjacencyListToMatrix method.
    #in my implementaion of the segmentation algorithm, I use an adjacency list to represent the graph.
    def createAdjacencyListFromImage(self,I):
        #create the adjacency list
        adjacencyList = {}
        #set the source node
        adjacencyList[self.source] = []
        #set the sink node
        adjacencyList[self.sink] = []
        #set the nodes
        for i in range(self.size):
            for j in range(self.size):
                adjacencyList[i*self.size+j] = []
        #set the edges
        for i in range(self.size):
            for j in range(self.size):
                #set the edges to the right
                if j < self.size-1:
                    adjacencyList[i*self.size+j].append(i*self.size+j+1)
                #set the edges to the left
                if j > 0:
                    adjacencyList[i*self.size+j].append(i*self.size+j-1)
                #set the edges to the top
                if i > 0:
                    adjacencyList[i*self.size+j].append((i-1)*self.size+j)
                #set the edges to the bottom
                if i < self.size-1:
                    adjacencyList[i*self.size+j].append((i+1)*self.size+j)
        #set the source edges
        for i in range(self.size):
            for j in range(self.size):
                adjacencyList[self.source].append(i*self.size+j)
        #set the sink edges
        for i in range(self.size):
            for j in range(self.size):
                adjacencyList[i*self.size+j].append(self.sink)
        return adjacencyList
       

    #Convert adjacency list to matrix.
    #In my segmentation I work with adjacecny lists, however for the testing script we need to convert the adjacency list
    #to a matrix.  This method takes an adjacency list as input and returns a matrix.  It is called in Jordans testing script.
    def adjacencyListToMatrix(self,adjacencyList):
        #create the matrix
        matrix = np.zeros((self.numNodes,self.numNodes))
        #set the matrix values to 0
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                matrix[i][j] = 0
        #If there is an edge between two nodes, set the matrix value to 1
        for i in range(self.numNodes):
            for j in range(len(adjacencyList[i])):
                if adjacencyList[i][j] != self.source and adjacencyList[i][j] != self.sink:
                    matrix[i][adjacencyList[i][j]] = 1
        return matrix



#end of segmentationClass class

