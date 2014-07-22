__author__ = 'Lothilius'

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import pylab as P
import time

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
            pixel = pixel.replace("0.", "0")
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
        elif last < pixel:  #and i - indx.peek() >= 10:
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

    thePeaks = [0] * 4752
    for i, d in zip(indx, opStack):
        thePeaks[i] = d

    return thePeaks


# Return Radius given two peaks on the same ring
def radius(peakL, peakR):
    rad = (peakR - peakL)
    return rad


#write an array to a file.
def arrayTofile(dataArray, fileNum):
    fileName = "/Users/admin/Dropbox/School/Spring-2014/PHY-474/Labs/Zeeman/play_data/dataManipulation/ImageData" + fileNum + "_a.csv"
    print(fileName)
    with open(fileName, 'w+', newline='') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=",")
        for each in dataArray:
            linewriter.writerow([each])
    print("done")


#Create file from array with finaldata.csv as the name and append.
def dataTofile(dataArray):
    fileName = "Finaldata.csv"
    with open(fileName, 'w') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=",")
        linewriter.writerows(dataArray)
    csvfile.close()


#Create array of data for creating the Delta enregy function.
def dataArray(peakL1, peakR1, peakL2, peakI, peakO):
    rad1 = radius(peakL1, peakR1) / 2
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
        return (left + right) / 2

    leftThird = (2 * left + right) / 3
    rightThird = (left + 2 * right) / 3

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
def strip_color(image_rgb1, color_of_interest, sig):
    image_rgb = ski.img_as_ubyte(image_rgb1)

    if color_of_interest == 0:
        colorow = np.array([[1.0, 0, 0]] * len(image_rgb[0]))
    elif color_of_interest == 1:
        colorow = np.array([[0, 1.0, 0]] * len(image_rgb[0]))
    elif color_of_interest == 2:
        colorow = np.array([[0, 0, 1.0]] * len(image_rgb[0]))

    image_stripped = np.array([colorow] * len(image_rgb))

    image_stripped = image_stripped * image_rgb

    image_stripped = rgb2gray(image_stripped)

    image_proc = ski.filter.canny(image_stripped, sigma=sig, low_threshold=0, high_threshold=7)

    #Uses up the majority of time.
    img, radii = get_center(image_proc, image_rgb)

    print('The red and blue have been stripped from image.')

    return image_stripped, image_proc, radii


#Use edges to get the center.
def get_center(edges, image_rgb1):
    #image = ski.img_as_ubyte(edges[977:2277, 1650:3150])
    image = edges

    hough_radii = np.arange(400, 565, 5)
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

    # Draw the most prominent 15 circles
    center_array = []
    image = ski.color.gray2rgb(image)
    for idx in np.argsort(accums)[::-1][:15]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        center_array.append([center_x, center_y])
        cx, cy = circle_perimeter(center_y, center_x, radius)
        image[cy, cx] = (220, 0, 0)

    return image, center_array


#plot data
def plotevents(datalist):
    x = range(0, len(datalist))
    x = np.array(x)
    y = datalist

    plt.plot(y)
    #fig, ax = plt.subplots()
    #plt.scatter(x, y, s=20, marker='.', c='blue')

    #plt.show()
    return "done"


#Create histogram plot for center of circles.
def histo_plot(image, center):
    h = np.array([0] * len(image[0]))
    hist = np.array([h] * len(image))

    for each in center:
        hist[each[0]][each[1]] += 1
    H, xedges, yedges = np.histogram2d(center[:, 0], center[:, 1], bins=25)
    plt.pcolor(xedges, yedges, H)
    plt.show()

    return hist


#Get edges from processed image
def getedge(center, slice):
    r1 = []
    for i, item in enumerate(slice[center:]):
        if item != False:
            r1.append(i)
    return r1 + center


#Calculate the spacial frequency
def space_freq(rn, rm, rj, etelon=.01):
    sf = np.divide(1, 2 * etelon) * np.divide(np.square(rn) - np.square(rj), (np.square(rn) - np.square(rm)))
    return sf


#Compare Radius from calibration image and sub radius image
def cal_center(calcenter, center):
    dif_x = 0
    dif_y = 0
    if calcenter == center:
        return dif_x, dif_y
    elif calcenter[0] > center[0]:
        dif_x = calcenter[0] - center[0]
        if calcenter[1] > center[1]:
            dif_y = calcenter[1] - center[1]
        elif calcenter[1] > center[1]:
            dif_y = center[1] - calcenter[1]
        else:
            pass
        return dif_x, dif_y
    else:
        dif_x = center[0] - calcenter[0]
        if calcenter[1] > center[1]:
            dif_y = calcenter[1] - center[1]
        elif calcenter[1] > center[1]:
            dif_y = center[1] - calcenter[1]
        else:
            pass
        return dif_x, dif_y


#Get magnetic field function
def mag_field():
    raw_mag = [[3, .465], [3.4, .530], [3.8, .590], [4.2, .651], [4.6, .712], [5, .774],
               [5.4, .832]]  #raw_input("Type in Amps and magnetic field tuples: ")
    b_field = []

    raw_mag = np.array(raw_mag)

    x = raw_mag[:, 0]
    y = raw_mag[:, 1]

    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(x)
    print coefficients
    print polynomial

    plt.subplot(5, 4, 9)
    #plt.plot(x, ys, xerr=0, yerr=(y * 0.05))
    plt.errorbar(x, ys, xerr=0, yerr=(y * 0.05))
    plt.ylabel('Measured Magnetic Field')
    plt.xlabel('Amps Applied')
    plt.xlim(2, 6)
    plt.ylim(0, 1)

    return polynomial


def main():
    #dtype={'names': ['amps', 'rowT', 'colR', 'rowB', 'colL'], 'formats': ['f2', 'i1', 'i1', 'i1', 'i1']})
    for file in os.listdir("/Users/"):
        if file == 'martin':
            print("Welcome Martin")
            inputFileDer = "/Volumes/Ket/Dropbox/School/Summer-2014/SSC-479R/comp-cert/as_images/"
            break
        elif file == 'admin':
            print("Welcome Admin")
            inputFileDer = "/Users/admin/Dropbox/School/Summer-2014/SSC-479R/comp-cert/as_images/"
            break
    run = 1
    calibration = []
    sfreq = []

    #create list of image files in inputFileDer directory
    image_list = os.listdir(inputFileDer)

    for item in image_list:
        if run == 1:
            color_of_interest = raw_input("What color is of interest? ")
            if ('Red' in color_of_interest) or ('red' in color_of_interest):
                color_of_interest = 0
            elif ('Green' in color_of_interest) or ('green' in color_of_interest):
                color_of_interest = 1
            elif ('Blue' in color_of_interest) or ('blue' in color_of_interest):
                color_of_interest = 2
        else:
            file_name = item

            #Strip file name in to the amps for the file.
            amps = item.replace('.JPG', '').replace('_', '.')
            amps = float(amps)
        #data = arrayFromFile(inputFileDer+fileNum)
        org_image = jpg_to_array(inputFileDer, file_name)

        if run == 1:
            image_stripped, image_proc, center = strip_color(org_image, color_of_interest, sig=7)
        else:
            image_stripped, image_proc2, center = strip_color(org_image, color_of_interest, sig=4)

        #Show detected rings using canny algorithm
        plt.subplot(311)
        plt.suptitle(file_name)
        plt.imshow(image_proc, origin='lower')
        plt.gray()

        #Show the stripped down image
        plt.imshow(image_stripped, origin='lower', alpha=.5)
        plt.gray()


        #Print events in the average of the radius in the y access.
        avrg_y = np.round(np.mean(center, axis=0)[0], decimals=0)
        print('Average y value for center: ' + str(avrg_y))
        uncertanty_y = np.round(np.std(center, axis=0)[0], decimals=1)
        print('Uncertainty in y: ' + str(uncertanty_y))

        avrg_x = np.round(np.mean(center, axis=0)[1], decimals=0)
        print('Average x value for center: ' + str(avrg_x))
        uncertanty_x = np.round(np.std(center, axis=0)[1], decimals=1)
        print('Uncertainty in x: ' + str(uncertanty_x))



        #y value will give horizontal slice.
        base_line = avrg_y - org_image[avrg_y][avrg_x][1]
        plotevents(org_image[avrg_y][:, 1] + base_line)
        l = plt.axhline(y=avrg_y, color='r')
        plotevents(image_stripped[avrg_y] + base_line)
        l = plt.axvline(x=avrg_x, color='r')
        plt.margins(0)
        os.system("afplay woohoo.wav")
        #plt.show()


     #Get first right peak
        plt.subplot(312)
        if run == 1:
            edges_array = getedge(avrg_x, image_proc[avrg_y])
            peakPrec = uncertanty_x
            main_peak_list = []
            for i, item in enumerate(edges_array):
                if i + 1 < len(edges_array):
                    peakL1 = ternarySearch(image_stripped[avrg_y], item, edges_array[i + 1], peakPrec)
                    main_peak_list.append([peakL1, item, edges_array[i + 1]])

            main_peak_list = np.array(main_peak_list)[::2]
            for each in main_peak_list:
                l = plt.axvline(x=each[0], color='r')

            calibration = sorted(main_peak_list[:, 1])
            calibration = np.append(calibration, main_peak_list[:, 2])
        else:
            #Find and Graph lines of main peaks
            edges_array = sorted(calibration)
            peakPrec = uncertanty_x
            main_peak_list = []
            for i, item in enumerate(edges_array):
                if i + 1 < len(edges_array):
                    peakL1 = ternarySearch(image_stripped[avrg_y], item, edges_array[i + 1], peakPrec)
                    main_peak_list.append([peakL1, item, edges_array[i + 1]])

            main_peak_list = np.round(np.array(main_peak_list)[::2])

            for each in main_peak_list:
                l = plt.axvline(x=each[0], color='r')

            #Find and Graph lines for 1st secondary peaks
            jminus_peak_list = []
            jplus_peak_list = []
            for i, edge in enumerate(main_peak_list):
                if i + 1 < len(main_peak_list):
                    j = i + 1
                    limit = (main_peak_list[:, 0][i] - main_peak_list[:, 0][j])/2
                    peakL1 = ternarySearch(image_stripped[avrg_y], edge[1] + limit, edge[1], peakPrec)
                    jminus_peak_list.append([peakL1, edge[1] + limit, edge[1]])

            jminus_peak_list = np.round(np.array(jminus_peak_list))

            for each in jminus_peak_list:
                l = plt.axvline(x=each[0], color='g')

            #Find and Graph lines for 2nd secondary peaks
            jplus_peak_list = []
            for i, edge in enumerate(main_peak_list):
                if i + 1 < len(main_peak_list):
                    j = i + 1
                    limit = (main_peak_list[:, 0][i] - main_peak_list[:, 0][j])/2
                    peakL1 = ternarySearch(image_stripped[avrg_y], edge[2], edge[2] - limit, peakPrec)
                    jplus_peak_list.append([peakL1, edge[2], edge[2] - limit])

            jplus_peak_list = np.round(np.array(jplus_peak_list))

            for each in jplus_peak_list:
                l = plt.axvline(x=each[0], color='g')

            sfreq = np.round(space_freq(main_peak_list[0][0], main_peak_list[1][0], jminus_peak_list[0][0]), 2)

        #Build and graph main image
        field_image = np.array([image_stripped[avrg_y]] * 300)
        field_image_proc = np.array([image_proc[avrg_y]] * 300)
        field_image_proc2 = np.array([image_proc[avrg_y]] * 300)

        plt.imshow(field_image_proc, origin='lower')
        plt.imshow(field_image_proc2, origin='lower')
        plt.imshow(field_image, origin='lower', alpha=.5)
        plotevents(org_image[avrg_y][:, 1])
        plotevents(image_stripped[avrg_y])
        l = plt.axvline(x=avrg_x, color='r')
        plt.margins(0)
        os.system("afplay woohoo.wav")
        plt.subplot_tool()
        plt.show()

        run = 2


        center = np.array(center)

        #H = histo_plot(image_proc, center)

        #N , Bins, Patches = plt.hist(center[:, 0], 15)
        #n , bins, patches = plt.hist(center[:, 1], 15)

        #plt.show()




main()