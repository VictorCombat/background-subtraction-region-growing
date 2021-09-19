import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#############################
# Read the image            #
#############################

img = cv.imread("data/roi.jpg")
if len(sys.argv) == 2:
    img = cv.imread(sys.argv[1])
# cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()


#############################
# Create color image        #
#############################
width = 512
height = 512
img2 = np.zeros((height, width, 3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
img2[:,0:width//2] = (255,0,0)      # (B, G, R)
img2[:,width//2:width] = (0,123,56)
# cv.imshow("Image", img2)

cv.waitKey(0)
cv.destroyAllWindows()

#############################
# noise removal / filtering #
#############################
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
#convertit l'image en espace colorimétrique CIELAB, puis 
#en désactivant séparément les composants L et AB avec des paramètres h différents.
# plt.subplot(121),plt.imshow(img)
# plt.subplot(122),plt.imshow(dst)
# plt.show()

# cv.imshow("Image",dst)
cv.waitKey(0)
cv.destroyAllWindows()


#############################
# image averaging /smoothing#
# low pass filtering        #
#############################

####################################
#2D Convolution ( Image Filtering )#
####################################

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

#cv.imshow("Image",dst)
# cv.waitKey(0)
# cv.destroyAllWindows()


####################################
#			Average                #
####################################
blur = cv.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


####################################
#	 Filtrage gaussien             #
####################################

blur2 = cv.GaussianBlur(img,(5,5),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur2),plt.title('GaussBlurred')
plt.xticks([]), plt.yticks([])
plt.show()


####################################
#	Filtrage médian                #
####################################

median = cv.medianBlur(img,5)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('MedianBlur')
plt.xticks([]), plt.yticks([])
plt.show()


####################################
#	Filtrage bilateral             #
####################################
bilateral = cv.bilateralFilter(img,9,75,75)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bilateral),plt.title('BilateralBlur')
plt.xticks([]), plt.yticks([])
plt.show()








