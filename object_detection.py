import cv2
import numpy as np
import os

# img = cv2.pyrDown(cv2.imread('images/VIN01677.jpg', cv2.IMREAD_UNCHANGED))

def label_all_img_contour(img_dir):
    for img_file in os.listdir(img_dir):
        img = cv2.pyrDown(cv2.imread(os.path.join(img_dir, img_file), cv2.IMREAD_UNCHANGED))
        contour_img = label_single_img_contour(img)
        filename = "contour_" + img_file
        cv2.imwrite(filename, contour_img)

def label_single_img_contour(img):
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    200, 255, cv2.THRESH_BINARY)
    # find contours and get the external one
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))

        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        # and draw the circle in blue
        img = cv2.circle(img, center, radius, (255, 0, 0), 2)

    img = cv2.resize(img, (500,500))
    # cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    #
    # cv2.imshow("contours", img)
    #
    # cv2.resizeWindow("contours", 500, 500)
    #
    # while True:
    #     key = cv2.waitKey(1)
    #     if key == 27: #ESC key to break
    #         break
    #
    # cv2.destroyAllWindows()
    return img

if __name__ == "__main__":
    label_all_img_contour('ItemRecognitionSystem\CORe50_cluttered_detection')
    # img = cv2.pyrDown(cv2.imread('CORe50_cluttered_detection/VIN01698.jpg', cv2.IMREAD_UNCHANGED))
    # img = cv2.pyrDown(cv2.imread('ItemRecognitionSystem\CORe50_cluttered_detection/VIN01682.JPG', cv2.IMREAD_UNCHANGED))
    # label_single_img_contour(img)
