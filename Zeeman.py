__author__ = 'Lothilius'

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import math
import getpass
import time

from skimage import data, filter, exposure
from skimage.color import rgb2gray
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

measured_data = []
start = time.time()

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
        return round((left + right) / 2), f[(left + right) / 2]

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


#Create histogram for array of values.
def histo_plot(data, num_bins=10):
    plt.figure('Data Histogram')
    n, bins, patches = plt.hist(data, bins=num_bins)

#Get edges from processed image given a slice and the determined center
def getedge(center, slice):
    r1 = []
    l1 =[]
    #Build right of center array
    for i, item in enumerate(slice[center:]):
        if item != False:
            r1.append(i)

     #Build left of center array
    for i, item in enumerate(slice[:center]):
        if item != False:
            l1.append(i)

    return l1, r1 + center


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

    #plt.subplot(5, 4, 9)
    plt.figure(0)
    plt.errorbar(x, y, color='k', ecolor='red', xerr=0, yerr=(y * 0.008), marker='o', markersize=4.0, linestyle='')
    plt.plot(x, ys)

    plt.ylabel('Measured Magnetic Field (T)')
    plt.xlabel('Applied Current (amps)')
    plt.xlim(x[0] * .95, x[-1] * 1.02)
    plt.show()

    return polynomial

#Find and plot best fit line
def best_fit(data_x, data_y):
    x = data_x
    y = data_y

    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(x)
    # print coefficients
    print polynomial

    #plt.subplot(5, 4, 10)
    plt.plot(x, ys)

def get_calibration(inputFileDer, file_name):
        color_of_interest = raw_input("What color is of interest? ")
        if ('Red' in color_of_interest) or ('red' in color_of_interest):
            color_of_interest = 0
        elif ('Green' in color_of_interest) or ('green' in color_of_interest):
            color_of_interest = 1
        elif ('Blue' in color_of_interest) or ('blue' in color_of_interest):
            color_of_interest = 2

        #data = arrayFromFile(inputFileDer+fileNum)
        org_image = jpg_to_array(inputFileDer, file_name)

        image_stripped, image_proc, center = strip_color(org_image, color_of_interest, sig=7)

        #Show detected rings using the canny algorithm
        #plt.subplot(5, 4, 1)
        plt.figure(1)
        # plt.subplot(2, 1, 1)
        amps = 0.0
        amps = str(amps) + ' amps'
        plt.title(amps)
        plt.ylabel('Pixel Bin')
        plt.xlabel('Pixel Bin')
        plt.imshow(image_proc, origin='lower')
        plt.gray()

        #Show the stripped down image
        plt.imshow(image_stripped, origin='lower', alpha=.5)
        plt.gray()


        #Print average of the center in the y axis.
        avrg_y = np.round(np.mean(center, axis=0)[0], decimals=0)
        print('Average y value for center: ' + str(avrg_y))
        uncertanty_y = np.round(np.std(center, axis=0)[0], decimals=1)
        print('Uncertainty in y: ' + str(uncertanty_y))

        #Print average of the center in the x axis.
        avrg_x = np.round(np.mean(center, axis=0)[1], decimals=0)
        print('Average x value for center: ' + str(avrg_x))
        uncertanty_x = np.round(np.std(center, axis=0)[1], decimals=1)
        print('Uncertainty in x: ' + str(uncertanty_x))



        #y value will give horizontal slice.
        # base_line = avrg_y - org_image[avrg_y][avrg_x][1]
        # plotevents(org_image[avrg_y][:, 1] + base_line)
        l = plt.axhline(y=avrg_y, color='r')
        # plotevents(image_stripped[avrg_y] + base_line)
        l = plt.axvline(x=avrg_x, color='r')
        # plt.margins(0)
        #plt.show()

        #Use edge array to get peak list
        edges_left, edges_right= getedge(avrg_x, image_proc[avrg_y])
        #edges_array = image_proc[avrg_y]
        peakPrec = uncertanty_x
        main_peak_list = []

        for i, item in enumerate(edges_right):
            if i + 1 < len(edges_right):
                peakL1, value = ternarySearch(image_stripped[avrg_y], item, edges_right[i + 1], peakPrec)
                main_peak_list.append([peakL1, item, edges_right[i + 1], round(value)])

        #Remove any peak with less than half the maximum peak intensity
        new = []
        for item in main_peak_list:
            new.append(item[3])

        half_largest_peak = max(new) / 2

        main_peaks = []
        for i, item in enumerate(main_peak_list):
            if item[3] > half_largest_peak:
                main_peaks.append(item)

        main_peak_list = main_peaks

        #Array of edges used to define boundaries for the primary peaks
        calibration = np.array(main_peak_list)

        #Build and graph main image
        field_image = np.array([image_stripped[avrg_y]] * 300)
        field_image_proc = np.array([image_proc[avrg_y]] * 300)
        #field_image_proc2 = np.array([image_proc[avrg_y]] * 300)

        #Plot Intensity field with plot, peaks, and edges
        #plt.subplot(5, 4, 1)
        plt.figure(2, figsize=(5, 1.5))
        plt.imshow(field_image_proc, origin='lower')
        #plt.imshow(field_image_proc2, origin='lower')
        plt.imshow(field_image, origin='lower', alpha=.5)
        plt.xlim(avrg_x * .98, main_peak_list[-1][0] * 1.02)
        l = plt.axvline(x=avrg_x, color='r')
        plt.margins(0)


        #Plot intensities
        plt.figure(3, figsize=(5, 1.5))
        for each in main_peak_list:
             l = plt.axvline(x=each[0], color='r')
        plotevents(image_stripped[avrg_y])
        l = plt.axvline(x=avrg_x, color='r')
        plt.margins(0)
        plt.ylabel('Intensity ')
        plt.xlabel('Pixel Bin')
        plt.xlim(avrg_x * .98, main_peak_list[-1][0] * 1.02)

        os.system("afplay woohoo.wav")

        plt.show()

        return image_stripped, image_proc, org_image, calibration, color_of_interest, avrg_y, avrg_x

def get_sf(calibration, inputFileDer, file_name, color_of_interest, run):
        #Strip file name in to the amps for the file.
        amps = file_name.replace('.JPG', '').replace('_', '.')
        amps = float(amps)


        #Strip color and get center
        org_image = jpg_to_array(inputFileDer, file_name)

        image_stripped, image_proc, center = strip_color(org_image, color_of_interest, sig=4)

        #Show detected rings using the canny algorithm
        placement = run + 8

        #plt.subplot(5, 4, placement)
        #plt.subplot(2, 1, 1)
        plt.title(str(amps) + 'amps')
        plt.imshow(image_proc, origin='lower')
        plt.gray()

        #Show the stripped down image
        plt.imshow(image_stripped, origin='lower', alpha=.5)
        plt.gray()


        #Print average of the center in the y axis.
        avrg_y = np.round(np.mean(center, axis=0)[0], decimals=0)
        print('Average y value for center: ' + str(avrg_y))
        uncertanty_y = np.round(np.std(center, axis=0)[0], decimals=1)
        print('Uncertainty in y: ' + str(uncertanty_y))

        #Print average of the center in the x axis.
        avrg_x = np.round(np.mean(center, axis=0)[1], decimals=0)
        print('Average x value for center: ' + str(avrg_x))
        uncertanty_x = np.round(np.std(center, axis=0)[1], decimals=1)
        print('Uncertainty in x: ' + str(uncertanty_x))



        #y value will give horizontal slice.
        # base_line = avrg_y - org_image[avrg_y][avrg_x][1]
        # plotevents(org_image[avrg_y][:, 1] + base_line)
        l = plt.axhline(y=avrg_y, color='r')
        # plotevents(image_stripped[avrg_y] + base_line)
        l = plt.axvline(x=avrg_x, color='r')
        plt.margins(0)
        plt.ylabel('Pixel Bin')
        plt.xlabel('Pixel Bin')
        os.system("afplay woohoo.wav")
        #plt.show()


        #Get first right peak
        placement = run + 12

        #plt.subplot(5, 4, placement)
        #plt.subplot(2, 1, 2)
        plt.title(amps)

        #Find and Graph lines of main peaks for non calibration files.
        edges_array = calibration
        peakPrec = uncertanty_x
        main_peak_list = []
        for i, item in enumerate(edges_array):
            if i + 1 < len(edges_array):
                peakL1, value = ternarySearch(image_stripped[avrg_y], item[1], item[2], peakPrec)
                main_peak_list.append([peakL1, item[1], item[2], value])

        main_peak_list = np.array(main_peak_list)

        #Find and Graph lines for 1st secondary peaks
        jminus_peak_list = []
        for i, edge in enumerate(main_peak_list):
            if i + 1 < len(main_peak_list):
                j = i + 1
                limit = (main_peak_list[:, 0][i] - main_peak_list[:, 0][j]) / 2
                peakL1, value = ternarySearch(image_stripped[avrg_y], edge[1] + limit, edge[1], peakPrec)
                jminus_peak_list.append([peakL1, i, edge[1] + limit, edge[1], value])

        jminus_peak_list = np.array(jminus_peak_list)

        #Find and Graph lines for 2nd secondary peaks
        jplus_peak_list = []
        for i, edge in enumerate(main_peak_list):
            if i + 1 < len(main_peak_list):
                j = i + 1
                limit = (main_peak_list[:, 0][i] - main_peak_list[:, 0][j]) / 2
                peakL1, value = ternarySearch(image_stripped[avrg_y], edge[2], edge[2] - limit, peakPrec)
                jplus_peak_list.append([peakL1, i, edge[2], edge[2] - limit, value])

        jplus_peak_list = np.array(jplus_peak_list)

        sfreq_minus = []
        sfreq_plus = []
        global measured_data
        for i, each in enumerate(main_peak_list):
            if i < len(main_peak_list) - 1:
                measured_data.append([amps, main_peak_list[i][0], main_peak_list[i + 1][0], jminus_peak_list[i][0], jplus_peak_list[i][0]])

                sfreq_minus.append(np.round(space_freq(main_peak_list[i][0], main_peak_list[i + 1][0], jminus_peak_list[i][0]), 2))
                sfreq_plus.append(np.round(space_freq(main_peak_list[i][0], main_peak_list[i + 1][0], jplus_peak_list[i][0]), 2))

        #build spacial frequency array of values using all peaks
        sfm_mean = np.round(np.mean(sfreq_minus), decimals=2)
        sfp_mean = np.round(np.mean(sfreq_plus), decimals=2)

        #Create uncertainty based on standard deviation
        un_sfm = np.round((np.std(sfreq_minus) / math.sqrt(len(sfreq_minus)) * 2), decimals=2)
        un_sfp = np.round((np.std(sfreq_plus) / math.sqrt(len(sfreq_minus)) * 2), decimals=2)

        print(np.round(sfreq_minus, 2))
        print(np.round(sfreq_plus, 2))

        #Build and graph field image
        field_image = np.array([image_stripped[avrg_y]] * 300)
        field_image_proc = np.array([image_proc[avrg_y]] * 300)
        #field_image_proc2 = np.array([image_proc[avrg_y]] * 300)

        plt.imshow(field_image_proc, origin='lower')
        #plt.imshow(field_image_proc2, origin='lower')
        plt.imshow(field_image, origin='lower', alpha=.5)
        l = plt.axvline(x=avrg_x, color='r')

        plt.ylabel('Intensity')
        plt.xlabel('Pixel Bin')
        plt.xlim(avrg_x * .95, main_peak_list[-1][0] * 1.10)

        plt.margins(0)


        l = plt.axvline(x=avrg_x, color='r')

        for each in main_peak_list:
            l = plt.axvline(x=each[0], color='r')
        for each in jminus_peak_list:
            l = plt.axvline(x=each[0], color='b')
        for each in jplus_peak_list:
            l = plt.axvline(x=each[0], color='g')

        #plotevents(org_image[avrg_y][:, 1])
        plotevents(image_stripped[avrg_y])
        l = plt.axvline(x=avrg_x, color='r')
        plt.ylabel('Intensity')
        plt.xlabel('Pixel Bin')
        plt.xlim(avrg_x * .95, main_peak_list[-1][0] * 1.10)

        plt.margins(0)

        #plt.show()

        return sfm_mean, un_sfm, sfp_mean, un_sfp, amps


def main():
    #dtype={'names': ['amps', 'rowT', 'colR', 'rowB', 'colL'], 'formats': ['f2', 'i1', 'i1', 'i1', 'i1']})
    user_name = getpass.getuser()

    if user_name == 'martin':
        print("Welcome Martin")
        inputFileDer = "/Volumes/Ket/Users/Martin/Dropbox/School/Summer-2014/SSC-479R/comp-cert/as_images/"
    elif user_name == 'admin':
        print("Welcome Admin")
        inputFileDer = "/Users/admin/Dropbox/School/Summer-2014/SSC-479R/comp-cert/as_images/"
    else:
        print("Welcome ", str(user_name))
        inputFileDer = raw_input("Please enter directory of image files ending with a slash: ")

    #Initialize Some variables
    run = 1
    final_data = np.array([[0, 0, 0, 0]])
    uncertainty_b = .05
    global measured_data
    b_field = mag_field()

    file_name = raw_input("Type in B=0 file name: ")

    #create list of image files in inputFileDer directory
    image_list = os.listdir(inputFileDer)
    if image_list[0][-3:] == 'ore':
        stupid_hidden_file = 2
    else:
        stupid_hidden_file = 1

    image_stripped, image_proc, org_image, calibration, color_of_interest, avrg_y, avrg_x = get_calibration(inputFileDer, file_name)

    for item in image_list[stupid_hidden_file:]:
        sfm_mean, un_sfm, sfp_mean, un_sfp, amps = get_sf(calibration, inputFileDer, item, color_of_interest, run)

        #Place data in to final j array
        if run == 1:
            final_minus = np.array([[sfm_mean, un_sfm, b_field(amps), uncertainty_b]])
            final_plus = np.array([[sfp_mean, un_sfp, b_field(amps), uncertainty_b]])
        else:
            final_minus = np.append(final_minus, [[sfm_mean, un_sfm, b_field(amps), uncertainty_b]], axis=0)
            final_plus = np.append(final_plus, [[sfp_mean, un_sfp, b_field(amps), uncertainty_b]], axis=0)

        final_data = np.append(final_data, [[sfm_mean, un_sfm, b_field(amps), uncertainty_b]], axis=0)
        final_data = np.append(final_data, [[sfp_mean, un_sfp, b_field(amps), uncertainty_b]], axis=0)
        print(np.round(final_data, 2))

        os.system("afplay woohoo.wav")


        run += 1

    dataTofile(measured_data)
    #print(measured_data)

    #Transfer final data to x and y with errors for error plot
    y = final_data[:, 0]
    un_y = final_data[:, 1]
    x = final_data[:, 2]
    un_x = final_data[:, 3]

    os.system("afplay about_time.m4a")
    best_fit(final_minus[:, 2], final_minus[:, 0])
    best_fit(final_plus[:, 2], final_plus[:, 0])

    plt.errorbar(x, y, xerr=un_x, yerr=un_y, fmt='+')
    plt.ylabel('Spatial Frequency')
    plt.xlabel('Magnetic Field')
    #plt.xlim(2, 6)
    #plt.ylim(0, 1)
    plt.subplot_tool()
    duration = time.time() - start

    print "Program duration: ", duration/60

    plt.show()

main()