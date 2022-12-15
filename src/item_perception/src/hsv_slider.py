#!/usr/bin/env python

from __future__ import print_function
import cv2 as cv
import argparse

# global
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Image'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")


def  ros_main(rgb_img):

    # cv.namedWindow(window_capture_name)
    # cv.namedWindow(window_detection_name)

    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
    # ret, rgb_image = vid.read()

    # if rgb_image is None:
    #     break

    frame_HSV = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    #frame_threshold = cv.putText(frame_threshold,f"H: {low_H} to {high_H}", (100,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    # cv.imshow(window_capture_name, rgb_image)
    # cv.imshow(window_detection_name, frame_threshold)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    return frame_HSV, frame_threshold
    # After the loop release the cap object

def video_main():
    vid = cv.VideoCapture(0)
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)

    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
    while True:
        ret, rgb_image = vid.read()

        if rgb_image is None:
            break

        frame_HSV = cv.cvtColor(rgb_image, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        #frame_threshold = cv.putText(frame_threshold,f"H: {low_H} to {high_H}", (100,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        cv.imshow(window_capture_name, rgb_image)
        cv.imshow(window_detection_name, frame_threshold)



        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()

def main(args):



    frame = cv.imread(f"images/ws_{args.image}.png")

    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)

    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
    while True:

        if frame is None:
            break
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        #frame_threshold = cv.putText(frame_threshold,f"H: {low_H} to {high_H}", (100,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        cv.imshow(window_capture_name, frame)
        cv.imshow(window_detection_name, frame_threshold)



        key = cv.waitKey(300)
        if key == ord('q') or key == 27:
            break

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    # parser.add_argument('--image', help='Image name', default=0, type=int)
    # args = parser.parse_args()

    # main(args)
    # hsv_sliders_from_image()
    video_main()
