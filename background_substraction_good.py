import numpy as np
import cv2
import math
from datetime import datetime
import pyttsx3

cap = cv2.VideoCapture(3)
engine = pyttsx3.init()
engine.setProperty('rate', 400)
bg=None
prev=None
i=0

###################################################################

def recognize_img(roi,BW):
    contours,_ = cv2.findContours(BW.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contour of max area(hand)
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    # Create bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 0)

    # Find convex hull
    hull = cv2.convexHull(contour)

    # Draw contour
    drawing = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)
    # Find convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
    # tips) for all defects
    count_defects = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

        # if angle > 90 draw a circle at the far point
        if angle <= 90:
            count_defects += 1
            cv2.circle(roi, far, 1, [0, 0, 255], -1)

        cv2.line(roi, start, end, [0, 255, 0], 2)

    # Print number of fingers
    print("count_defects=", count_defects)
    # Print number of fingers
    if count_defects == 0:
        cv2.putText(roi, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        engine.say("one")
    elif count_defects == 1:
        cv2.putText(roi, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        engine.say("two")
    elif count_defects == 2:
        cv2.putText(roi, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        engine.say("three")
    elif count_defects == 3:
        cv2.putText(roi, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        engine.say("four")
    elif count_defects == 4:
        cv2.putText(roi, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        engine.say("five")
    else:
        pass
    #engine.runAndWait()
    return drawing,roi


def get_diff(fg,bg):
    diff = cv2.absdiff(bg.astype("uint8"), fg)
    h, w, d = diff.shape
    total = h * w * d
    diff_rate = (diff < 20).sum()
    v=((total-diff_rate)*100)/total
    #print(total,diff_rate,v)
    return diff,v

#######################################################################
while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if prev is None:
        prev=frame
    if i >= 60 and bg is None:
        bg = frame
    x = 10
    y = 60
    w = 300
    h = 250
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 0)
    if get_diff(frame, prev)[1] >0.10 and bg is not None:
        roi_fg = frame[y:y + h, x:x + w]
        roi_bg = bg[y:y + h, x:x + w]
        diff=get_diff(roi_fg,roi_bg)[0]
        gray_diff=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        if (int(datetime.now().strftime('%H')) > 17 or int(datetime.now().strftime('%H')) < 5):  # >6pm
            # on night
            threashold=20
        else:
            # on day
            threashold=10
        mask=cv2.threshold(gray_diff, threashold, 255, cv2.THRESH_BINARY)[1]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.erode(mask, k, iterations=5)
        skin=cv2.bitwise_and(roi_fg,roi_fg,mask=mask)
        cv2.imshow('thresholded2',gray_diff)
        try:
            numpy_horizontal_concat1 = np.concatenate((skin, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), axis=1)
            drawing,roi=recognize_img(skin,mask)
            numpy_horizontal_concat2 = np.concatenate((drawing, roi), axis=1)
            numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)
            cv2.imshow('All Frames', numpy_vertical_concat)
        except  Exception as e:
            print("Recognization failed"+str(e))

    cv2.imshow('Frame', frame)
    prev=frame
    i+=1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()