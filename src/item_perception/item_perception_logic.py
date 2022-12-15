#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import json
import os
import ros_numpy

CURRENT_WD = os.path.dirname(__file__)

COLORS_JSON_PATH = os.path.join(CURRENT_WD,"colors.json")
COLOR_JSON = open(COLORS_JSON_PATH)
COLORS = json.load(COLOR_JSON)

ITEMS_JSON_PATH = os.path.join(CURRENT_WD,"item_info.json")
ITEMS_JSON = open(ITEMS_JSON_PATH)
ITEMS_INFO = json.load(ITEMS_JSON)

IGNORE_X = 500
IGNORE_Y = 2200


def filter_image(image, lower_H, upper_H, lower_S, upper_S, lower_V, upper_V):
    # convert input image to HSV color space with OpenCV
    cv2.imwrite("workstation.png",image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define lower and upper HSV values
    lower = np.array([lower_H,lower_S,lower_V])
    higher = np.array([upper_H,upper_S,upper_V])
    # filter image in HSV color space
    filtered_image = cv2.inRange(hsv_image,lower,higher)
    #imS = cv2.resize(filtered_image, (960, 540))
    #cv2.imshow('contours',imS)
    #cv2.waitKey(0)
    # erode and dilate
    # erode_kernel = np.ones((5,5),np.uint8)
    # dilate_kernel = np.ones((10,10),np.uint8)
    dilate_kernel = np.ones((12,12),np.uint8)
    # filtered_image = cv2.erode(filtered_image,erode_kernel,iterations=1)
    filtered_image = cv2.dilate(filtered_image,dilate_kernel,iterations=1)

    #imS = cv2.resize(filtered_image, (960, 540))
    #cv2.imshow('contours',imS)
    #cv2.waitKey(0)
    return filtered_image

def get_grasp_point(filtered_image,lower_bound,upper_bound):
    contours,_ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grasp_point, box = None, None

    if len(contours) > 0:
        for c in contours:
            contour_area = cv2.contourArea(c)
            print("Estimated contour area: ", contour_area)
            print("Lower bound: {} Upper bound: {}".format(lower_bound, upper_bound))
            if lower_bound < contour_area < upper_bound:
                print(cv2.contourArea(c))
                # granular contours
                #print(cv2.contourArea(c))
                #input()
                #cv2.drawContours(filtered_image, [c], -1, (150, 150, 150), 8)

                rect = cv2.minAreaRect(c)
                (x,y),(w,h),a = rect

                #box = cv2.boxPoints(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box) #turn into ints
                #cv2.drawContours(filtered_image,[box],0,(240,240,240),3)

                # gp_1 = (int((box[0][0]+box[3][0])/2),int((box[0][1]+box[3][1])/2))
                # gp_2 = (((box[0][0]+box[1][0])/2),((box[0][1]+box[1][1])/2))
                # gp_3 = (((box[1][0]+box[2][0])/2),((box[1][1]+box[2][1])/2))
                # gp_4 = (((box[0][0]+box[3][0])/2),((box[0][1]+box[3][1])/2))
                # get the midpoints of each side
                gp_1 = (box[0] + box[3]) / 2
                gp_2 = (box[0] + box[1]) / 2
                gp_3 = (box[1] + box[2]) / 2
                gp_4 = (box[2] + box[3]) / 2
                # Heuristic: Grasp the midpoint that is closest to robot.
                # In our case this should always be the midpoint with the smallest y value.
                grasp_point = min([gp_1, gp_2, gp_3, gp_4], key=lambda gp: gp[1])
                grasp_point = (int(grasp_point[0]), int(grasp_point[1]))

                # get middle of plate
                #middle_point = (int((box[0][0]+box[3][0])/2),int((box[0][1]+box[1][1])/2))

                #for i in range(4):
                #    point = point = (int((box[i][0]+box[(i+1)%4][0])/2),int((box[i][1]+box[(i+1)%4][1])/2))
                #    cv2.circle(filtered_image,point,5,(50,50,200),-1)
                # return None, None
    return grasp_point, box

def get_HSV_bounds(item_request):
    color = ITEMS_INFO[item_request]['color']

    return COLORS[color]['lower_H'],COLORS[color]['upper_H'],COLORS[color] \
        ['lower_S'],COLORS[color]['upper_S'],COLORS[color]['lower_V'],COLORS[color]['upper_V']

def handle_item_request(rgb_image_deque,item_request):
    filtered_image = np.zeros(shape=rgb_image_deque[0].shape[:2]) 
    for rgb_image in rgb_image_deque:
        # image = cv2.imdecode(rgb_image,cv2.IMREAD_COLOR)
        #image_depths = cv2.imdecode(depth_image,cv2.IMREAD_UNCHANGED)
        lower_H, upper_H, lower_S, upper_S, lower_V, upper_V = get_HSV_bounds(item_request)
        lower_bound = ITEMS_INFO[item_request]["lower_bound"]
        upper_bound = ITEMS_INFO[item_request]["upper_bound"]
        # filtered_image = filter_image(rgb_image,lower_H, upper_H, lower_S, upper_S, lower_V, upper_V)
        new_filtered_image = filter_image(rgb_image,lower_H, upper_H, lower_S, upper_S, lower_V, upper_V)
        filtered_image = np.logical_or(new_filtered_image, filtered_image)
        # filtered_image = np.array(filtered_image, dtype="uint8")
        # cv2.imwrite("/home/scazlab/test.png", filtered_image)

    filtered_image = np.array(filtered_image, dtype="uint8") * 255
    # grasp_point, box = get_grasp_point(filtered_image,lower_bound,upper_bound)
    # Remove mask on points outisde of boundary
    filtered_image[:IGNORE_X, :] = 0
    filtered_image[:, IGNORE_Y:] = 0
    grasp_point, box = get_grasp_point(filtered_image,lower_bound,upper_bound)

    print(grasp_point, box, "Grasp point, box")
    #depth_at_point = image_depths[middle_point[1]][middle_point[0]]
    #print(depth_at_point)
    # import pdb; pdb.set_trace()

    return grasp_point, filtered_image
