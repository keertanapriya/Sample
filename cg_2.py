#1)
# SAMPLING AND QUANTIZATION

import cv2 as cv
import numpy as np

inputImg = cv.imread("scenery.jpg")

'''
# display the image as such
cv.imshow('Input Image', inputImg)
cv.waitKey(0)
'''

'''
# display the image after converting to gray
grayImg = cv.cvtColor(inputImg, cv.COLOR_RGB2GRAY)
cv.imshow("Gray Image", grayImg)
cv.waitKey(0)
'''

'''
# resize an image
resizeImg = cv.resize(inputImg, (200, 200))
cv.imshow("Resized Image", resizeImg)
cv.waitKey(0)
cv.imwrite("Resized_Image.jpg", resizeImg)
'''

# convert to HSV and extract specific color images
hsv = cv.cvtColor(inputImg, cv.COLOR_BGR2HSV)

# to extract green
green = np.uint8([[[0, 255, 0]]])
hsvGreen = cv.cvtColor(green, cv.COLOR_BGR2HSV)

hueGreen = hsvGreen[0][0][0]
lowerGreen = np.array([hueGreen - 10, 100, 100])
upperGreen = np.array([hueGreen + 10, 255, 255]) 

maskGreen = cv.inRange(hsv, lowerGreen, upperGreen)

# to extract blue
blue = np.uint8([[[255, 0, 0]]])
hsvBlue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)

hueBlue = hsvBlue[0][0][0]
lowerBlue = np.array([hueBlue - 10, 100, 100])
upperBlue = np.array([hueBlue + 10, 255, 255])

maskBlue = cv.inRange(hsv, lowerBlue, upperBlue)

mask = maskGreen + maskBlue

res = cv.bitwise_and(inputImg, inputImg, mask=mask)

cv.imshow("Masked", mask)
cv.waitKey(0)
cv.imshow("Green", res)
cv.waitKey(0)

#2)
# EDGE DETECTION

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('sudoku.jpg', cv.IMREAD_GRAYSCALE)
 
laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize = 5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize = 5)
 
plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(sobely, cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


#3)
# HISTOGRAM EQUALIZATION

import cv2
import numpy as np
import random   
import matplotlib.pyplot as plt

image = cv2.imread('b_w.jpg', cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist(image, [0], None, [256], [0, 256])

plt.xlabel("pixel value")
plt.ylabel("count")
plt.plot(histogram)
plt.legend()
plt.show()

eq_histogram_image = cv2.equalizeHist(image)
eq_histogram = cv2.calcHist(eq_histogram_image, [0], None, [256], [0, 256])


plt.xlabel("pixel value")
plt.ylabel("count")
plt.plot(eq_histogram)
plt.legend()
plt.show()

cv2.imshow("original", image)
cv2.waitKey(0)

cv2.imshow("Equalized", eq_histogram_image)
cv2.waitKey(0)

cv2.destroyAllWindows()

# HISTOGRAM MATCHING

import cv2
import numpy as np

def histogram_matching(source, reference):
    # Calculate the histograms.
    src_hist, bins = np.histogram(source.flatten(), 256, [0,256])
    ref_hist, bins = np.histogram(reference.flatten(), 256, [0,256])

    # Calculate the cumulative distribution function for both histograms.
    src_cdf = np.cumsum(src_hist)
    src_cdf_normalized = src_cdf / float(src_cdf.max())
    ref_cdf = np.cumsum(ref_hist)
    ref_cdf_normalized = ref_cdf / float(ref_cdf.max())

    # Create a lookup table to map pixel values from the source to the reference.
    lookup_table = np.zeros(256)
    g_j = 0
    for g_i in range(256):
        while ref_cdf_normalized[g_j] < src_cdf_normalized[g_i] and g_j < 255:
            g_j += 1
        lookup_table[g_i] = g_j

    # Map the pixel values of the source image through the lookup table.
    matched_image = lookup_table[source.flatten()].reshape(source.shape).astype(np.uint8)

    return matched_image

# Load source and reference images
source_img = cv2.imread('scenery.jpg', cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread('b_w.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram matching
matched_img = histogram_matching(source_img, reference_img)

# Display the result
cv2.imshow('Source Image', source_img)
cv2.imshow('Reference Image', reference_img)
cv2.imshow('Matched Image', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#4)
# MAX FILTER

import cv2
import numpy as np

def max_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel)

# Example usage
image = cv2.imread("img.png")
kernel_size = 5  # Adjust kernel size as needed
max_filtered_image = max_filter(image, kernel_size)
cv2.imshow("Max Filtered Image", max_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# MEAN FILTER

import cv2

def mean_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

# Example usage
image = cv2.imread("img.png")
kernel_size = 5  # Adjust kernel size as needed
mean_filtered_image = mean_filter(image, kernel_size)
cv2.imshow("Mean Filtered Image", mean_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# MEDIAN FILTER

import cv2

def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# Example usage
image = cv2.imread("img.png")
kernel_size = 5  # Adjust kernel size as needed
median_filtered_image = median_filter(image, kernel_size)
cv2.imshow("Median Filtered Image", median_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# MIN FILTER

import cv2
import numpy as np

def min_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel)

# Example usage
image = cv2.imread("img.png")
kernel_size = 5  # Adjust kernel size as needed
min_filtered_image = min_filter(image, kernel_size)
cv2.imshow("Min Filtered Image", min_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# LOG TRANSFORMATION

import cv2
import numpy as np

def log_transformation(image):
    # Normalizing the pixel values
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Applying log transformation
    transformed_image = np.log1p(normalized_image)
    # Denormalizing the pixel values
    transformed_image = cv2.normalize(transformed_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return transformed_image

# Example usage
image = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)
log_transformed_image = log_transformation(image)
cv2.imshow("Log Transformed Image", log_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




#5)
# Python program to draw smily face
import turtle

# turtle object
pen = turtle.Turtle()

# function for creation of eye
def eye(col, rad):
	pen.down()
	pen.fillcolor(col)
	pen.begin_fill()
	pen.circle(rad)
	pen.end_fill()
	pen.up()


# draw face
pen.fillcolor('yellow')
pen.begin_fill()
pen.circle(100)
pen.end_fill()
pen.up()

# draw eyes
pen.goto(-40, 120)
eye('white', 15)
pen.goto(-37, 125)
eye('black', 5)
pen.goto(40, 120)
eye('white', 15)
pen.goto(40, 125)
eye('black', 5)


# draw mouth
pen.goto(-40, 85)
pen.down()
pen.right(90)
pen.circle(40, 180)
pen.up()

turtle.mainloop()