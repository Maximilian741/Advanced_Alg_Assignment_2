#Max Casteel
#11/25/2022
#Software Assignment 2
#Advanced Algorithms - Dr. Jordan Malof - Fall 2022
#implements Ford-Fulkerson algorithm to find max flow in a graph
#*************************************************************************************************************************
#*************************************************************************************************************************
#*************************************************************************************************************************
#NOTE: I have not been able to get this to work correctly yet.  I am still working on it.  I am submitting this as is, but I will continue to work on it.
#      This is for the purpose to actually understand this algorihtm and not just get a grade.
#      I will continue to work on this until the night of 12/13/2022, and then I will submit what I have.
#      I am trying to do this before 11:00pm so to only lose 12% of my grade.
#*************************************************************************************************************************
#*************************************************************************************************************************
#*************************************************************************************************************************
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
#import breadthFirstSearch as bfs

#Prompt - Implement (i) Ford-Fulkerson algorithm, and then use it in (ii) an implementation
#of the segmentation algorith described in section 7.10 of the kleinberg and Tardos book.
#For full credit, you must implement the algorithm as described in Section 7.10 of the Kleinberg and Tardos book.

#Let 'x' \in\mathbb{N^2} be row/column coordinates of a pixel, and I(x) \in\mathbb{N^3} be RGB values of a pixel at x.
#Per the implementation in section 7.10, each pixel is treated as a graph and must have weights for (i), the likelihood of that pixel being in the 
#background, denoted by b(x).  (ii), the likelihood of that pixel being in the forground
#denoted here by a(x).  (iii), and the weights being neighboring pixels, denoted p(x_i,x_j), where x_i and x_j are the coordinates
#of two pixels, i and j.  You must set these values in the following way:


#a(x) = 442 - round(distance(x,x_a)))
#b(x) = 442 - round(distance(x,x_b)))

#p(x_i,x_j) = p_0, distance(x_i,x_j) < 2 
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
    #This can be handled by the Ford-Fulkerson algorithm defined below.
    #The graph will be a 2D array of integers.  The first dimension will be the number of nodes in the graph.
    def segmentImage(self,I):
        #set all the values specified in the api above.
        #p0 is the weight of the edges between neighboring pixels. It is set to 2 bec
        self.p0 = 2
        self.x_a = np.array([I.shape[0]-1,I.shape[1] - 1])
        #self.x_b = np.array([I.shape[0]-1,I.shape[1] - 1])
        self.x_b = np.array([I.shape[0],0])
        self.source = 0
        self.size = I.shape[0]*I.shape[1]
        self.numNodes = I.shape[0]*I.shape[1]+2
        self.sink = I.shape[0]*I.shape[1]+1
        #The graph will be a 2D array of integers.  The first dimension will be the number of nodes in the graph.
        self.graph = np.zeros((I.shape[0]*I.shape[1]+2,I.shape[0]*I.shape[1]+2))
        
        self.L = np.zeros((I.shape[0],I.shape[1]))
        self.I = np.zeros((I.shape[0],I.shape[1]))
        self.bgraph = self.buildGraph(I)
        self.maxFlow = self.fordFulkerson(self.bgraph,self.source,self.sink)
        self.buildL(self.maxFlow)
        return self.buildL(self.maxFlow)



    #distance function
    def distance(self,x,y):
        return np.sqrt(np.sum((x-y)**2))


    #buildGraph
    #this method will build the graph for the Ford-Fulkerson algorithm
    #the graph will be a 2D array of integers.  The first dimension will be the number of nodes in the graph.
    #The input to this method will be the image I.
    #The output will be the graph.
    def buildGraph(self, I):
        #for each pixel in the image
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                #set the weights for the source and sink
                #the source will be connected to all the nodes in the foreground
                #the sink will be connected to all the nodes in the background
                self.graph[self.source][i*I.shape[1]+j+1] = 442 - round(self.distance(self.x_a,np.array([i,j])))
                self.graph[i*I.shape[1]+j+1][self.sink] = 442 - round(self.distance(self.x_b,np.array([i,j])))
                #for each pixel in the image
                for k in range(I.shape[0]):
                    for l in range(I.shape[1]):
                        #if the distance between the two pixels is less than 2
                        if self.distance(np.array([i,j]),np.array([k,l])) < 2:
                            #set the weight to p0
                            self.graph[i*I.shape[1]+j+1][k*I.shape[1]+l+1] = self.p0
        return self.graph

    #buildL
    #this method will build the output image for the Ford-Fulkerson algorithm
    #the input to this method will be the maximum flow of the graph gererated by the ford fulkerson algorithm
    #
    #the output image will be the same size as the input image, but the values will be 0 or 1.
    #in this method, we will set the value to 1 if it is a maximum flow, and 0 otherwise.
    #the output will be the image L.
    #we will need to check all around the pixel to see if it is a maximum flow.
    def buildL(self,maxFlow):
        #for each pixel in the image
        for i in range(self.I.shape[0]):
            for j in range(self.I.shape[1]):
                ##check all around the pixel to see if it is a maximum flow.
                #if the pixel is a maximum flow, set the value to 1
                #the below line has an incorrect index to scalar comparison.
                if self.bgraph[self.source][i*self.I.shape[1]+j+1] == self.p0:
                    self.L[i,j] = 1
                #if the pixel is not a maximum flow, set the value to 0
                else:
                    self.L[i,j] = 0
        return self.L
       

    #breadth-first search
    #this method will be used to find the path from the source to the sink
    #within the ford fulkerson algorithm.
    #this method will return true if there is a path from the source to the sink.
    def breadthFirstSearch(self,bgraph,s,t,path):
        visited = [False] * (self.sink+1)
        queue = []
        queue.append(s)
        visited[s] = True
        #while the queue is not empty
        while queue:
            u = queue.pop(0)
            #for each node in the graph
            for ind,val in enumerate(bgraph[u]):
                #if the node has not been visited and the value is greater than 0
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    path[ind] = u
                    #if the sink has been reached, return true
        return True if visited[t] else False

    #Ford-Fulkerson algorithm
    #this is the implementation of the Ford-Fulkerson algorithm
    #this method will return the maximum flow of the graph, furhtermore it will be used to create the output image.
    #the output image will be created by looking at the edges between the nodes and the sink.  If the edge is saturated, then the pixel is in the foreground.
    #if the edge is not saturated, then the pixel is in the background.
    #the output image will be a 2D array of binary values.
    #the output image will be the same size as the input image.
    def fordFulkerson(self,bgraph,s,t):
        #initialize the residual graph
        rGraph = bgraph
        #initialize the path
        path = [-1] * (self.sink+1)
        maxFlow = 0
        #while there is a path from the source to the sink
        while self.breadthFirstSearch(rGraph,s,t,path):
            #initialize the path flow
            pathFlow = float("Inf")
            #for each node in the path
            v = t
            while v != s:
                u = path[v]
                #update the path flow
                pathFlow = min(pathFlow,rGraph[u][v])
                v = path[v]
            #for each node in the path
            v = t
            while v != s:
                u = path[v]
                #update the residual graph
                rGraph[u][v] -= pathFlow
                rGraph[v][u] += pathFlow
                v = path[v]
            #update the max flow
            maxFlow += pathFlow
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
        matrix = np.zeros((self.size*self.size+2,self.size*self.size+2))
        #set the source edges
        for i in range(self.size):
            for j in range(self.size):
                matrix[self.source][i*self.size+j] = 1
        #set the sink edges
        for i in range(self.size):
            for j in range(self.size):
                matrix[i*self.size+j][self.sink] = 1
        #set the edges
        for i in range(self.size):
            for j in range(self.size):
                #set the edges to the right
                if j < self.size-1:
                    matrix[i*self.size+j][i*self.size+j+1] = 1
                #set the edges to the left
                if j > 0:
                    matrix[i*self.size+j][i*self.size+j-1] = 1
                #set the edges to the top
                if i > 0:
                    matrix[i*self.size+j][(i-1)*self.size+j] = 1
                #set the edges to the bottom
                if i < self.size-1:
                    matrix[i*self.size+j][(i+1)*self.size+j] = 1
        return matrix



#end of segmentationClass class

