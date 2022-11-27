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


#Import your python class, assuming the class definition .py file is in the same file directory as the test script.
import segmentationClass as sc

#Load the test image, assuming the test image is in the same file directory as the test script.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib as mpl


#Display the image using matplotlib package.
img = mpimg.imread('testImage.png')
imgplot = plt.imshow(img)
plt.show()

#Instantiate your class, and set the hyperparameters.
obj = sc.segmentationClass()

#Input the image to the segmentImage() function using the API specified above.
L = obj.segmentImage(img)

#Display the adjacency matrix for the graph nodes corresponding to the pixels at location (0,0) and (1,0) in the image.
print(L[0,0])
print(L[1,0])

#Display the segmented image in black and white color using matplotlib, where segmentation values of 1 correspond to foreground (color of white),
#and values of 0 corrrespond to the background (color of black).
plt.imshow(L, cmap='gray')
plt.show()

