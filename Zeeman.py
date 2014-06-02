
__author__ = 'Lothilius'

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import pylab as P

from skimage import data, filter, exposure
from skimage.color import rgb2gray
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter


#Pull data from CSV file
def arrayFromFile(filename):
    """Given an external file containing numbers,
            create an array from those numbers."""
    dataArray = []
    with open(filename, 'r+') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            dataArray.append(row)
    return dataArray[1565]

#Reduce data to a few peak points
def peaks(data):
    opStack = MyQue.Stack()
    indx = MyQue.Stack()
    i = 0
    last = 0
    lasti = 0
    vlast = 0

    for pixel in data[1565]:
        if len(pixel) == 2:
            pixel = pixel.replace("0.","0")
        if opStack.isEmpty():
            opStack.push(pixel)
            indx.push(i)
            i = i + 1
        elif last > pixel:
            if vlast <= last and last != opStack.peek() and i - indx.peek() <= 3:
                indx.push(lasti)
                opStack.push(last)
                i = i + 1
            else:
                i = i + 1
        elif last < pixel: #and i - indx.peek() >= 10:
            if i - indx.peek() <= 10:
                indx.pop()
                opStack.pop()
                indx.push(i)
                opStack.push(pixel)
                i = i + 1
            else:
                indx.push(i)
                opStack.push(pixel)
                i = i + 1
        else:
            i = i + 1
        last = pixel
        lasti = i
        vlast = last

    thePeaks = [0]*4752
    for i, d in zip(indx, opStack):
        thePeaks[i] = d

    return thePeaks

# Return Radius given two peaks on the same ring
def radius(peakL, peakR):
   rad = (peakR - peakL)
   return rad

#write an array to a file.
def arrayTofile(dataArray, fileNum):
    fileName = "/Users/admin/Dropbox/School/Spring-2014/PHY-474/Labs/Zeeman/play_data/dataManipulation/ImageData"+fileNum+"_a.csv"
    print(fileName)
    with open(fileName, 'w+', newline='') as csvfile:
        linewriter = csv.writer(csvfile, delimiter= ",")
        for each in dataArray:
            linewriter.writerow([each])
    print("done")

#Create file from array with finaldata.csv as the name and append.
def dataTofile(dataArray):
    fileName = "Finaldata2.csv"
    with open(fileName, 'a', newline='') as csvfile:
        linewriter = csv.writer(csvfile, delimiter= ",")
        linewriter.writerow(dataArray)

#Create array of data for creating the Delta enregy function.
def dataArray(peakL1, peakR1, peakL2, peakI, peakO):
    rad1 = radius(peakL1,peakR1) / 2
    halfPoint = peakL1 + rad1
    rad2 = radius(peakL2, halfPoint)
    radI = radius(peakI, halfPoint)
    radO = radius(peakO, halfPoint)
    radArray = [rad1, rad2, radI, radO]
    return radArray

#Ternary Search
def ternarySearch(f, left, right, absolutePrecision):
    #left and right are the current bounds; the maximum is between them
    if (right - left) < absolutePrecision:
        return (left + right)/2

    leftThird = (2*left + right)/3
    rightThird = (left + 2*right)/3

    if f[leftThird] < f[rightThird]:
        return ternarySearch(f, leftThird, right, absolutePrecision)
    else:
        return ternarySearch(f, left, rightThird, absolutePrecision)


#Processes jpg file
def jpg_to_array(file_path, file_name):
    full_path = file_path + file_name
    image = data.load(full_path)
    #plt.imshow(image)
    #plt.show()
    return image

#Remove all but one color
def strip_color(image_rgb1, color_of_interest):
    image_rgb = ski.img_as_ubyte(image_rgb1)

    if color_of_interest == 0:
        colorow = np.array([[1.0, 0, 0]] * len(image_rgb[0]))
    elif color_of_interest == 1:
        colorow = np.array([[0, 1.0, 0]] * len(image_rgb[0]))
    elif color_of_interest == 2:
        colorow = np.array([[0, 0, 1.0]] * len(image_rgb[0]))

    image_proc = np.array([colorow] * len(image_rgb))

    image_proc = image_proc * image_rgb
    image_proc = rgb2gray(image_proc)

    rings = ski.filter.sobel(image_proc)

    img = get_radius(image_proc, image_rgb1)
    plt.imshow(img)

    plt.show()

    return rings

#Detect radius
def get_radius(edges, image_rgb1):
    image = ski.img_as_ubyte(image_rgb1[1127:2127, 1900:2900])
    plt.imshow(edges)
    plt.show()

    hough_radii = np.arange(560, 570, 2)
    hough_res = hough_circle(edges, hough_radii)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, num_peaks=2)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius, radius])

    # Draw the most prominent 5 circles
    image = ski.color.gray2rgb(image)
    for idx in np.argsort(accums)[::-1][:1]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_y, center_x, radius)
        image[cy, cx] = (220, 0, 0)

    return image

#plot data
def plotevents(datalist):
    x = range(0, len(datalist))
    x = np.array(x)
    y = datalist

    plt.plot(y)
    #fig, ax = plt.subplots()
    #plt.scatter(x,y,s=20, marker='.', c='blue')

    plt.show()
    return "done"


def main():
    for file in os.listdir("/Users/"):
        if file == 'martin':
            print("Welcome Martin")
            inputFileDer = "/Users/martin/Dropbox/School/Spring-2014/PHY-474/Labs/Zeeman/play_data/as_images/"
            break
        elif file == 'admin':
            print("Welcome Admin")
            inputFileDer = "/Users/admin/Dropbox/School/Spring-2014/PHY-474/Labs/Zeeman/play_data/as_images/"
            break

    file_name = raw_input("Type in file name: ")
    #data = arrayFromFile(inputFileDer+fileNum)
    image = jpg_to_array(inputFileDer, file_name)

    color_of_interest = raw_input("What color is of interest? ")

    if ('Red' in color_of_interest) or ('red' in color_of_interest):
        color_of_interest = 0
    elif ('Green' in color_of_interest) or ('green' in color_of_interest):
        color_of_interest = 1
    elif ('Blue' in color_of_interest) or ('blue' in color_of_interest):
        color_of_interest = 2

    image = strip_color(image, color_of_interest)
    plotevents(image[1622])
    print('The red and blue have been stripped from image.')


    peakL1 = int(raw_input("Type in potential n peak pixel bin: "))
    peakPrec = int(raw_input("Type in precision: "))
    peakL1 = ternarySearch(image[1622], peakL1-15, peakL1+15, peakPrec)
    print(peakL1)


main()