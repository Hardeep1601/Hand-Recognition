# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# global variables
bg = None

# - - - - - Find Average background value - - - - -

def calc_accum_avg(frame, accumulated_weight):
    '''
    Given a frame and a previous accumulated weight, computed the weighted average of the image passed in.
    '''
    # Grab the background
    global bg
    # For first time, create the background from a copy of the frame
    if bg is None:
        bg = frame.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, bg, accumulated_weight)
    
# - - - - - To segment the region of hand - - - - -
    
def segment(frame, threshold=25):
    global bg
    # Calculate the Absolute Differentce between the backgroud and the passed in frame
    diff = cv2.absdiff(bg.astype("uint8"), frame)

    # Apply a threshold to the image so we can grab the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)



#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count1(thresholded, segmented):
    # find the convex hull of the segmented hand region
    # which is the maximum contour with respect to area
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distances = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    max_distance = distances[distances.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * max_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # initialize circular_roi with same shape as thresholded image
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    # draw the circular ROI with radius and center point of convex hull calculated above
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    # approach 1 - eliminating wrist
    #cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
    #print(len(cntsSorted[1:])) # gives the count of fingers

    # approach 2 - eliminating wrist
    # loop through the contours found
    for i, c in enumerate(cnts):

        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


# Function to count fingers using Convexhull

def count(thresholded_img, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)

    # most extreme top,bottom,left and right points
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # finding center point
    cx = (left[0] + right[0]) // 2
    cy = (top[1] + bottom[1]) // 2

    # calculate distance from center to all extreme points
    distance = pairwise.euclidean_distances([(cx, cy)],
                                            Y=[left, right, top, bottom])[0]

    # calculate one of the max distance
    max_distance = distance.max()

    # create circle
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded_img.shape[:2], dtype='uint8')

    cv2.circle(circular_roi, (cx, cy), radius, 255, 10)

    circular_roi = cv2.bitwise_and(thresholded_img, thresholded_img,
                                   mask=circular_roi)

    contours, hierarchy = cv2.findContours(circular_roi.copy(),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_NONE)

    count = 0

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        out_of_wrist = ((cy + (cy * 0.25)) > (y + h))

        limit_points = ((circumference * 0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1

    return count









# - - - - - Main - - - - -
        
if __name__ == "__main__":

    print("Testing code....")
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 20, 300, 300, 600

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            calc_accum_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

                # if yes, unpack the thresholded image and segmented contour
                (thresholded, segmented) = hand

                # count the number of fingers
                fingers = count(thresholded, segmented)

                cv2.putText(clone, "This is " + str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                # print("Test hand code")

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,0,255), 2)

        # increment the number of frames for tracking
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # print("Testing code loop ....")
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# Release the camera and destroy all the windows
camera.release()
cv2.destroyAllWindows()














