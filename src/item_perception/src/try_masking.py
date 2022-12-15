#!/usr/bin/env python

import cv2
import numpy as np
import argparse

COLORS = {"dark_pink":{'lower_H': 148,'upper_H':165,
                       'lower_S':100,'upper_S':255,
                       'lower_V':66,'upper_V':255},
          "light_blue":{'lower_H': 95,'upper_H':100,
                       'lower_S':100,'upper_S':255,
                       'lower_V':66,'upper_V':255}}

ITEM_SIZES = {"plate":(6000,12500),
              "bowl":(600,6000)}

def filter_image(image, lower_H, upper_H, lower_S, upper_S, lower_V, upper_V):
    # convert input image to HSV color space with OpenCV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define lower and upper HSV values
    lower = np.array([lower_H,lower_S,lower_V])
    higher = np.array([upper_H,upper_S,upper_V])
    # filter image in HSV color space
    filtered_image = cv2.inRange(hsv_image,lower,higher)
    return filtered_image

def get_bounding_box(filtered_image,image_type):
    contours,_ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if ITEM_SIZES[image_type][0] < cv2.contourArea(c) < ITEM_SIZES[image_type][1]:
            # granular contours
            cv2.drawContours(filtered_image, [c], -1, (150, 150, 150), 8)
            
            rect = cv2.minAreaRect(c)
            (x,y),(w,h),a = rect
            
            #box = cv2.boxPoints(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box) #turn into ints
            cv2.drawContours(filtered_image,[box],0,(240,240,240),3)
 
    cv2.imshow('contours',filtered_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", help="name of item color",type=str)
    parser.add_argument("--type", help="name of item type",type=str)
    parser.add_argument("--image", help="name of item type",type=int)
    args = parser.parse_args()
    
    
    image = cv2.imread(f"images/ws_{args.image}.png")
    cv2.imshow('BGR',image)
    cv2.waitKey(0)
    color = args.color
    image_type = args.type
    filtered_img = filter_image(image,COLORS[color]['lower_H'],COLORS[color]['upper_H'],
                                COLORS[color]['lower_S'],COLORS[color]['upper_S'],
                                COLORS[color]['lower_V'],COLORS[color]['upper_V'])
    #cv2.imshow('mask',filtered_img)
    #cv2.waitKey(0)
    
    contoured_image = get_bounding_box(filtered_img, image_type)
    
