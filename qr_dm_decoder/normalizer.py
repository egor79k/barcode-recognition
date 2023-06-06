# import the necessary packages
import numpy as np
# from pylsd import lsd
import argparse
import cv2
import math
from matplotlib import pyplot as plt
import os

def CountSide(line1, line2):
    k = (line1[1][1] - line1[0][1])/(line1[1][1] - line1[0][1])
    k = -1/k
    b = line1[0][1] - line1[0][0]*k
    k2 = (line2[1][1] - line2[0][1])/(line2[1][1] - line2[0][1])
    b2 = line2[0][1] - line2[0][0]*k2
    cross_x = (b2-b)/(k - k2)
    cross_y = cross_x*k2+b2
    return ((cross_x - line1[0][0])**2 +(cross_y - line1[0][1]**2))**0.5

def IsInside(line, x):
    return x>line[0][0] and x<line[0][2]

def CountDisp(img, x1, y1, x2, y2):
    cude_side = 10
    stay = 0
    aver_x = (x1+x2)//2
    aver_y = (y1+y2)//2
    if x2 == x1:
        k = 10
    else:
        k = (y1 - y2)/(x2 - x1)

    if k > 0:
        mtx = np.array(img[aver_x + stay: aver_x + stay+ cude_side, aver_y + stay : aver_y + stay + cude_side])
        mtx2 = np.array(img[aver_x - stay - cude_side : aver_x - stay, aver_y - stay - cude_side : aver_y - stay])
        #cv2.line(img, (aver_x + stay, aver_y + stay), (aver_x + stay + cude_side, aver_y + stay + cude_side), (255, 0, 0), 1)
        #cv2.line(img, (aver_x - stay - cude_side, aver_y - cude_side), (aver_x - stay, aver_y), (255, 0, 0), 1)
    else:
        mtx = np.array(img[aver_x + stay: aver_x + stay + cude_side, aver_y - stay - cude_side : aver_y - stay])
        mtx2 = np.array(img[aver_x - stay - cude_side : aver_x - stay, aver_y  + stay: aver_y + cude_side + stay])
        #cv2.line(img, (aver_x + stay, aver_y - stay - cude_side), (aver_x + stay + cude_side, aver_y - stay), (255, 0, 0), 1)
        #cv2.line(img, (aver_x - stay - cude_side, aver_y + stay), (aver_x - stay, aver_y + cude_side + stay), (255, 0, 0), 1)
    #print(k)
    #cv2.imshow('sdf', img), cv2.waitKey(0)
    return np.var(mtx), np.var(mtx2)

def CountAngle(l1, l2):
    y1 = l1[0]
    y2 = l2[0]
    c = abs(y1-y2)
    a = (y1**2 + 1)**0.5
    b = (y2**2+1)**0.5
    cos = (a**2+b**2-c**2)/(2*a*b)
    return math.acos(cos)*180/math.pi

def CountIntersection(line1, line2):
    if line1[0][0] == line1[0][2]:
        if line2[0][0] == line2[0][2]:
            return math.inf, math.inf
        else:
            k22 = (line2[0][1] - line2[0][3])/(line2[0][0] - line2[0][2])
            b22 = line2[0][1] - line2[0][0]*k22
            x_cross = line1[0][0]
            y_cross = k22*x_cross + b22
    elif line2[0][0] == line2[0][2]:
        k11 = (line1[0][1] - line1[0][3])/(line1[0][0] - line1[0][2])
        b11 = line1[0][1] - line1[0][0]*k11
        x_cross = line2[0][0]
        y_cross = k11*x_cross + b11
    else:
        k11 = (line1[0][1] - line1[0][3])/(line1[0][0] - line1[0][2])
        b11 = line1[0][1] - line1[0][0]*k11

        k22 = (line2[0][1] - line2[0][3])/(line2[0][0] - line2[0][2])
        b22 = line2[0][1] - line2[0][0]*k22

        x_cross = (b22 - b11)/(k11 - k22)
        y_cross = k11*x_cross + b11
        if k11 == k22:
            return math.inf, math.inf
    return [round(x_cross), round(y_cross)]
    # if line1[0][1] == -math.inf:
    #     cross_x = line1[0][0]
    #     cross_y = round(line2[0][0]*cross_x + line2[0][1])
    # elif line2[0][1] == -math.inf:
    #     cross_x = line2[0][0]
    #     cross_y = round(line1[0][0]*cross_x + line1[0][1])
    # else:
    #     cross_x = (line2[0][1] - line1[0][1])//(line1[0][0] - line2[0][0])
    #     cross_y = round(line1[0][0]*cross_x + line1[0][1])
    # return cross_x, cross_y

def CountDistance(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def ThreeLines(lines):
    answer = [CountIntersection(lines[0], lines[1]),
              CountIntersection(lines[1], lines[2])]
    left_right = []
    top_bottom = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            k = abs((y1-y2)/(x1-x2))
            if k>1:
                left_right.append(line)
            else:
                top_bottom.append(line)
    if len(left_right) == 0 or len(top_bottom) == 0:
        return 0
    if len(left_right) == 1:
        x1, y1 = top_bottom[0][0][2:4]
        x2, y2 = top_bottom[1][0][2:4]
        k = (left_right[0][0][1] - left_right[0][0][3])/(left_right[0][0][0] - left_right[0][0][2])

        b = y1 - x1*k
        test_y = k*x2 + b
        if abs(k) == math.inf or abs(k) == 0:
            answer.append([x1, max(y1, y2)])
            answer.append([x2, max(y1, y2)])
        elif test_y < y2:
            answer.append([ x1, y1])
            answer.append([(test_y - b)//k, test_y])
        else:
            b = y2 - x2*k
            test_y = k*x1 + b
            test_x = round((test_y - b)/k)
            answer.append([ test_x, round(test_y)])
            answer.append([x2, y2])
    else:

        x1, y1 = left_right[0][0][2:4]
        x2, y2 = left_right[1][0][2:4]
        k = (top_bottom[0][0][1] - top_bottom[0][0][3])/(top_bottom[0][0][0] - top_bottom[0][0][2])
        b = y1 - x1*k
        test_y = k*x2 + b
        if abs(k) == 0:
            answer.append([x1, max(y1, y2)])
            answer.append([x2, max(y1, y2)])
        elif test_y < y2:
            answer.append([ x1, y1])
            answer.append([x2, round(test_y)])
        else:
            b = y2 - x2*k
            test_y = k*x1 + b
            test_x = round((test_y - b)/k)
            answer.append([ test_x, round(test_y)])
            answer.append([x2, y2])
    return answer


def DetectQR(img, image_y, image_x):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5),np.uint8)
    gray = cv2.resize(gray, (120, 120))
    blur = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    image_y, image_x = blur.shape
    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    rho = max(image_x, image_y)//100  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = (max(image_x, image_y) // 50 - 2)*50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(blur)  # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # cv2.imshow('dfg', line_image), cv2.waitKey(0)
    if lines is None:
        return 0

    if len(lines) < 3:
         gray = (255 - gray)
         blur = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
         edges = cv2.Canny(blur, low_threshold, high_threshold)
         line_image = np.copy(blur)  # creating a blank to draw lines on
            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
         lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    
    true_lines = [1000, 0, 0, 1000]
    answer = []
    my_lines = lines[:4]
    edges = []
    if len(lines)<=2:
        return 0
    if len(lines) == 4:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        answer = [CountIntersection(lines[0], lines[1]),
                  CountIntersection(lines[1], lines[2]),
                  CountIntersection(lines[2], lines[3]),
                  CountIntersection(lines[3], lines[0])]
    if len(lines) == 3:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        answer = ThreeLines(lines)
    else:
        my_lines =[[image_x, image_y, image_x, 0],
                   [0, 0, image_x, 0],                   
                   [0, 0, 0, image_y],
                   [0, image_y, image_x, image_y]]
        for line in lines:
            for x1, y1, x2, y2 in line:
                aver_x = (x1+x2)//2
                aver_y = (y1+y2)//2
                k = abs((y1-y2)/(x1-x2))
                #print(aver_x, aver_y)
                if aver_x < true_lines[0]:
                    my_lines[0] = line
                    true_lines[0] = aver_x
                if aver_x > true_lines[2]:
                    my_lines[2] = line
                    true_lines[2] = aver_x
                if aver_y < true_lines[3]:
                    true_lines[3] = aver_y
                    my_lines[3] = line
                if aver_y > true_lines[1]:
                    true_lines[1] = aver_y
                    my_lines[1] = line
        # if len(np.unique(my_lines, axis=0))  == 3:
        #     return ThreeLines(my_lines[:3])
        # elif len(np.unique(my_lines, axis=0))  == 2:
        #     return 0
        answer = [CountIntersection(my_lines[0], my_lines[1]),
                  CountIntersection(my_lines[1], my_lines[2]),
                  CountIntersection(my_lines[2], my_lines[3]),
                  CountIntersection(my_lines[3], my_lines[0])]
    if answer == 0:
        return 0
    answer.sort(key = lambda x: x[0])
    if answer[2][1] < answer[3][1]:
        answer[2] = answer[3]
    scr = np.array([[0, 0], [0, image_y], [image_x, image_y]]).astype(np.float32)
    dst = np.array(answer[:3]).astype(np.float32)
    mat = cv2.getAffineTransform(scr, dst)
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))

def DetectDM(img, gray, image_y, image_x):
    low_threshold = 50
    high_threshold = 200
    kernel = np.ones((5, 5),np.uint8)

    #cv2.imshow('asd', blur)
    blur = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('sd', blur)
    for i in range(6):
        blur = cv2.medianBlur(blur, 5)
    #cv2.imshow('sdssdd', blur), cv2.waitKey(0)
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    rho = max(image_x, image_y)//100  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = (max(image_x, image_y) // 50 - 2)*50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray)  # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    try:
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    except:
        print(image_x, image_y)
        return 0


    show_img = np.copy(line_image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(show_img, ( x1, y1), (x2, y2), (255,0,0),3)
    #cv2.imshow('dfg', show_img)
    answer = []
    if len(lines) == 4:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, ( x1, y1), (x2, y2), (255,0,0),3)
        answer = [CountIntersection(lines[0], lines[1]),
                  CountIntersection(lines[1], lines[2]),
                  CountIntersection(lines[2], lines[3]),
                  CountIntersection(lines[3], lines[0])]
    if len(lines) == 3:
        answer = [CountIntersection(lines[0], lines[1]),
                  CountIntersection(lines[1], lines[2])]
        left_right = []
        top_bottom = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                k = abs((y1-y2)/(x1-x2))
                cv2.line(line_image, ( x1, y1), (x2, y2), (255,0,0),3)
                if k>1:
                    left_right.append(line)
                else:
                    top_bottom.append(line)
        if len(left_right) == 1:
            x1, y1 = top_bottom[0][0][2:4]
            x2, y2 = top_bottom[1][0][2:4]
            k = (left_right[0][0][1] - left_right[0][0][3])/(left_right[0][0][0] - left_right[0][0][2])
            b = y1 - x1*k
            test_y = k*x2 + b
            if test_y < y2:
                answer.append([ x1, y1])
                answer.append([(test_y - b)//k, test_y])
                #cv2.line(line_image, ( x1, y1), ((test_y - b)//k, test_y), (255,0,0),3)
            else:
                b = y2 - x2*k
                test_y = k*x1 + b
                test_x = round((test_y - b)/k)
                answer.append([ test_x, round(test_y)])
                answer.append([x2, y2])
                #cv2.line(line_image, (test_x, round(test_y)), ( x2, y2), (255,0,0),3)
        else:

            x1, y1 = left_right[0][0][2:4]
            x2, y2 = left_right[1][0][2:4]
            k = (top_bottom[0][0][1] - top_bottom[0][0][3])/(top_bottom[0][0][0] - top_bottom[0][0][2])
            b = y1 - x1*k
            test_y = k*x2 + b
            if test_y < y2:
                answer.append([ x1, y1])
                answer.append([x2, round(test_y)])
                #cv2.line(line_image, ( x1, y1), (x2, round(test_y)), (255,0,0),3)
            else:
                b = y2 - x2*k
                test_y = k*x1 + b
                test_x = round((test_y - b)/k)
                answer.append([ test_x, round(test_y)])
                answer.append([x2, y2])
                #cv2.line(line_image, (test_x, round(test_y)), ( x2, y2), (255,0,0),3)

    if len(lines) > 4:
        true_lines = []
        left_right = []
        top_bottom = []
        for line in lines:
            for line2 in lines:
                for x1,y1,x2,y2 in line:
                    for x11, y11, x22, y22 in line2:
                        if x1 == x11 and y1 == y11 and x2 == x22 and y2 == y22:
                            continue
                        #cv2.line(line_image, ( x1, y1), (x2, y2), (255,0,0),3)
                        cross_x, cross_y = CountIntersection(line, line2)
                        dist = [min(CountDistance(cross_x, cross_y, x1, y1), CountDistance(cross_x, cross_y, x2, y2)),
                                min(CountDistance(cross_x, cross_y, x11, y11),CountDistance(cross_x, cross_y, x22, y22))]
                        
                        if any(d<10 for d in dist):
                            true_lines.append(line2)
                            #print(line, line2, cross_x, cross_y, dist)

        really_true_lines = []
                            
        answer = really_true_lines

    answer.sort(key = lambda x: x[0])
    answer[2:4].sort(key = lambda x: x[1])
    scr = np.array([[0, 0], [0, image_y], [image_x, image_y]]).astype(np.float32)
    dst = np.array(answer[:3]).astype(np.float32)
    mat = cv2.getAffineTransform(scr, dst)
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
    return answer                                                                                                   

def Normalize(img, is_qr):
    Image_Max = 200
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_y, image_x = gray.shape
    if image_y > Image_Max or image_x > Image_Max:
        blur = cv2.resize(blur, (Image_Max, Image_Max))
        gray = cv2.resize(gray, (Image_Max, Image_Max))
        image_x = Image_Max
        image_y = Image_Max
    if is_qr:
        return  DetectQR(img, image_y, image_x)
    else:
        return DetectDM(img, gray, image_y, image_x)
    

# IdealPositions = {}

# with open('markup.txt') as file:
#     for line in file:
#         data = line.split(' ')
#         IdealPositions[data[0]] = list(map(int, data[1:-1]))
# file.close()

# writefile = open('results.txt', 'a')

# sum_norm_matches = 0
# count = 0
# sum_matches = 0
# comm_count = 0

# # for filename in os.listdir('./QR'):
# #     points = Normalize('./QR/'+filename, True)
# #     cv2.imshow('g ', points), cv2.waitKey(0)

