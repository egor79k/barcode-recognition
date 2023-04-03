# import the necessary packages
import numpy as np
import argparse
import cv2
import math
from matplotlib import pyplot as plt

def CountAngle(l1, l2):
    y1 = l1[0]
    y2 = l2[0]
    c = abs(y1-y2)
    a = (y1**2 + 1)**0.5
    b = (y2**2+1)**0.5
    cos = (a**2+b**2-c**2)/(2*a*b)
    return math.acos(cos)*180/math.pi

def CountIntersection(line1, line2):
    cross_x = round((line2[1] - line1[1])/(line1[0] - line2[0]))
    cross_y = round(line1[0]*cross_x + line1[1])
    return [cross_x, cross_y]


def Normalize(image_name, output_name):
    Image_Max = 500

    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_y, image_x = gray.shape
    # if image_y > Image_Max:
    #     gray = cv2.resize(gray, (image_x, Image_Max))
    #     image_y = Image_Max
    # if image_x > Image_Max:
    #     gray = cv2.resize(gray, (Image_Max, image_y))
    #     image_x = Image_Max
    # img = cv2.resize(img, (image_x, image_y))
    blur = cv2.medianBlur(gray, 5)
    # adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # thresh_type = cv2.THRESH_BINARY_INV
    # bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)

    # black_copy = np.zeros_like(gray)
    # bigger_one = np.concatenate((gray, black_copy), axis=0)
    # bigger_one = np.concatenate((black_copy, bigger_one), axis=0)
    # another_copy = np.zeros_like(bigger_one)
    # bigger_one = np.concatenate((another_copy, bigger_one), axis=1)
    # bigger_one = np.concatenate((bigger_one, another_copy), axis=1)
    #plt.imshow(bigger_one), plt.show()
    
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(blur)  # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)


    left_line = []
    right_line = []
    top_line = []
    bottom_line = []
    lll = [left_line]*4
    left_x = image_x
    right_x = 0
    top_y = image_y
    bottom_y = 0
    if lines is None:
        return 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            line_a = (y1-y2)/(x1-x2)
            line_b = y1-line_a*x1
            aver_x = (x1 + x2)/2
            aver_y = (y1 + y2)/2
            if abs(line_a) == math.inf:
                continue
            if line_a>1 or line_a < -1:
                if aver_x < left_x:
                    left_x = aver_x
                    left_line = [line_a, line_b]
                    lll[0] = line
                if aver_x > right_x:
                    right_x = aver_x
                    right_line = [line_a, line_b]
                    lll[2] = line
            else:
                if aver_y < top_y:
                    top_y = aver_y
                    top_line = [line_a, line_b]
                    lll[3] = line
                if aver_y > bottom_y:
                    bottom_y = aver_y
                    bottom_line = [line_a, line_b]
                    lll[1] = line
    # cv2.line(line_image,(lll[0][0][0], lll[0][0][1]), (lll[0][0][2], lll[0][0][3]),(255,0,0),5)
    # cv2.line(line_image,(lll[1][0][0], lll[1][0][1]), (lll[1][0][2], lll[1][0][3]),(255,0,0),5)
    # cv2.line(line_image,(lll[2][0][0], lll[2][0][1]), (lll[2][0][2], lll[2][0][3]),(255,0,0),5)
    # cv2.line(line_image,(lll[3][0][0], lll[3][0][1]), (lll[3][0][2], lll[3][0][3]),(255,0,0),5)

    intersections = [CountIntersection(left_line, top_line), CountIntersection(left_line, bottom_line), CountIntersection(bottom_line, right_line), CountIntersection(right_line, top_line)]




    # lines_edges = cv2.addWeighted(gray, 0.8, line_image, 1, 0)
    # cv2.line(line_image,(intersections[0][0], intersections[0][1]),(intersections[0][0], intersections[0][1]),(255,0,0),10)
    # cv2.line(line_image, (intersections[1][0], intersections[1][1]), (intersections[1][0], intersections[1][1]),(255,0,0),10)
    # cv2.line(line_image, (intersections[2][0], intersections[2][1]), (intersections[2][0], intersections[2][1]),(255,0,0),10)
    # plt.imshow(line_image),plt.show()

    new_coord = max(image_x, image_y)
    pts1 = np.float32(intersections)
    pts2 = np.float32([[0, 0], [0, new_coord], [new_coord, new_coord], [new_coord, 0]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(new_coord,new_coord))
    cv2.imwrite(output_name, dst)
    #plt.imshow(dst),plt.show()

#Normalize('p1.jpg', 'test.jpg')