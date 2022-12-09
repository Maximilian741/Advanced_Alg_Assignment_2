#Max Casteel
#11/27/2022
#Software Assignment 2
#Advanced Algorithms - Dr. Jordan Malof - Fall 2022
#Testing script for segmentationClass.py


#You must provide a valide .py testing script with your class that performs the following steps:
#Each of these steps must come with at least one comment that indicates what you are doing in that step.
#1.Import yur python class, assuming the class definition .py file is in the same file directory as the test script.
#2. Load the test image, assuming the test image is in the same file directory as the test script.
#3. Display the image using matplotlib package.
#4. Instantiate your class, and set the hyperparameters.
#5. Input the image to the segmentImage() function using the API specified above.
#6. Display the adjacency matrix for the graph nodes corresponding to the pixels at location (0,0) and (1,0) in the image.
#7. Display the segmented image in black and white color using matplotlib, where segmentation values of 1 correspond to foreground (color of white),
#and values of 0 corrrespond to the background (color of black).

#I will input one or two 25x25 pixel images of my choosing into your code to verify that it produces the correct output for the top-levl segmentation, as well
#as some intermediate steps.  I will also vary the user-chosen hyperparmeters
#such as x_a, x_b, and p_0 and verify that the output changes properly.



#1. Import your python class, assuming the class definition .py file is in the same file directory as the test script.
import segmentationClass as sc
#import matplotlib
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib as mpl

#2. Load the test image, assuming the test image is in the same file directory as the test script.
img = mpimg.imread('test_image.png')

#3. Display the image using matplotlib package.
plt.imshow(img)


#4. Instantiate your class, and set the hyperparameters.

obj = sc.segmentationClass()
obj.p0 = 2
obj.x_a = np.array([0,0])
obj.x_b = np.array([1,0])


#5. Input the image to the segmentImage() function using the API specified above.
#t = obj.segmentImage(img)
obj.segmentImage(img)



#6. Display the adjacency matrix for the graph nodes corresponding to the pixels at location (0,0) and (1,0) in the image.
print(obj.adjacencyListToMatrix(obj.createAdjacencyListFromImage(img))[[0,3],:])


#7. Display the segmented image in black and white color using matplotlib, where segmentation values of 1 correspond to foreground (color of white),
#and values of 0 corrrespond to the background (color of black).
plt.imshow(t, cmap = mpl.cm.gray)
plt.show()



