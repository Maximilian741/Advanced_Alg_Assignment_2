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
        #initialize L
        L = np.zeros((I.shape[0],I.shape[1],1))
        #initialize a(x)
        a = np.zeros((I.shape[0],I.shape[1],1))
        #initialize b(x)
        b = np.zeros((I.shape[0],I.shape[1],1))
        #initialize p(x_i,x_j)
        p = np.zeros((I.shape[0],I.shape[1],1))
        #initialize x_a
        x_a = np.zeros((1,2))
        #initialize x_b
        x_b = np.zeros((1,2))
      
        #set x_a
        x_a[0,0] = 0
        x_a[0,1] = 0
        #set x_b
        x_b[0,0] = I.shape[0]-1
        x_b[0,1] = I.shape[1]-1
        #set p_0
        p_0 = 2
        #set a(x)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                a[i,j] = 442 - np.linalg.norm(I[i,j] - I[x_a[0,0],x_a[0,1]])
        #set b(x)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                b[i,j] = 442 - np.linalg.norm(I[i,j] - I[x_b[0,0],x_b[0,1]])
        #set p(x_i,x_j)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                if np.linalg.norm([i,j] - [x_a[0,0],x_a[0,1]]) < 2 or np.linalg.norm([i,j] - [x_b[0,0],x_b[0,1]]) < 2:
                    p[i,j] = p_0
                else:
                    p[i,j] = 0

        #set up graph
        G = nx.DiGraph()
        #add nodes
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                G.add_node((i,j))
        #add edges
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                if i+1 < I.shape[0]:
                    G.add_edge((i,j),(i+1,j),capacity = p[i+1,j])
                if i-1 >= 0:
                    G.add_edge((i,j),(i-1,j),capacity = p[i-1,j])
                if j+1 < I.shape[1]:
                    G.add_edge((i,j),(i,j+1),capacity = p[i,j+1])
                if j-1 >= 0:
                    G.add_edge((i,j),(i,j-1),capacity = p[i,j-1])
                G.add_edge('s',(i,j),capacity = a[i,j])
                G.add_edge((i,j),'t',capacity = b[i,j])
        #find max flow
        max_flow = nx.maximum_flow(G,'s','t')
        #find min cut
        min_cut = nx.minimum_cut(G,'s','t')
        #set L
        for i in range(I.shape[0]):

            for j in range(I.shape[1]):
                if (i,j) in min_cut[1]:
                    L[i,j] = 1
                else:
                    L[i,j] = 0
        return L

        