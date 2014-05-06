__author__ = 'Lothilius'

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

from skimage import data, filter, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

inputFileDer = "/Users/martin/Dropbox/School/Spring-2014/PHY-474/Labs/Zeeman/play_data/dataManipulation/canon1_image_data/ImageData"
fileNum = ""

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



def main():
    fileNum = raw_input("Type in file number: ")
    data = arrayFromFile(inputFileDer+fileNum+'_a.csv')
    #rgbMap = peaks(data)
    #arrayTofile(rgbMap, fileNum)
    peakL1 = int(input("Type in potential n peak pixel bin: "))
    peakPrec = int(input("Type in precision: "))
    peakL1 = ternarySearch(data, peakL1-15, peakL1+15, peakPrec)
    print(peakL1)

    # peakR1 = int(input("Type in potential opposite peak pixel bin: "))
    # peakL2 = int(input("Type in n + 1 peak: "))
    # peakI = int(input("Type in potential inner peak: "))
    # peakO = int(input("Type in potential outer peak: "))
    #
    # finalData = dataArray(peakL1, peakR1, peakL2, peakI, peakO)
    # finalData.insert(0, fileNum)
    # dataTofile(finalData)

main()