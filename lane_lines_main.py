#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from Image_Proc import Image_Proc

#%matplotlib inline

#Step 1: Test the system on single test images

image_proc = Image_Proc()

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
plt.show()

'''
#cv2.imshow(image)
tmp = image  # for drawing a rectangle
stepSize = 9
(w_width, w_height) = (9, 9)  # window size
for x in range(0, image.shape[1] - w_width, stepSize):
    for y in range(0, image.shape[0] - w_height, stepSize):
        window = image[x:x + w_width, y:y + w_height, :]

        countYellow = 0

        print(x)
        print(y)
        for j in range(0, stepSize):
            for k in range(0, stepSize):
                #ttt = window.item(0,0,0)
                if ( (window.item(k,j,0) > 215) and (window.item(k,j,1) > 215) and (window.item(k,j,2) > 215) ):
                    countYellow = countYellow + 1
                    print(countYellow)

        #plt.imshow(window)
        #plt.show()
        # classify content of the window with your classifier and
        # determine if the window includes an object (cell) or not

        # draw window on image

        if (countYellow > 70):
            cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
            plt.imshow(np.array(tmp).astype('uint8'))
# show all windows
plt.show()
'''


files = os.listdir("test_images/")

rho = 1  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 35  # 35 minimum number of votes (intersections in Hough grid cell)
min_line_length = 5  # 15 minimum number of pixels making up a line
max_line_gap = 2  # 3 maximum gap in pixels between connectable line segments

for file in files:
    if file[0:6] != "output":
        img = mpimg.imread("test_images/" + file)
        plt.imshow(img)
        plt.show()
        #test = image_proc.sliding_window(img, 1, 3)
        #hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #color_filter = image_proc.filter_color(hsv_image, 0)
        #plt.imshow(color_filter)
        #plt.show()

        gray = image_proc.grayscale(img)

        #plt.imshow(gray)
        #plt.show()

        gray = image_proc.gaussian_blur(gray, 3)

        #plt.imshow(gray)
        #plt.show()

        #image_canny = image_proc.canny(gray, 50, 150)
        image_OTSU_canny = image_proc.otsu_canny(gray, 0.2)

        #plt.imshow(image_OTSU_canny)
        #plt.show()

        imshape = img.shape

        print("Printing polynomial values: ")
        print(
            str(.51 * imshape[1]) + ", " +
            str(imshape[0] * .58)
              )

        print(
            str(.49 * imshape[1]) + ", " +
            str(imshape[0] * .58)
              )

        print(
            str(0) + ", " +
            str(imshape[0] * .58)
              )

        print(
            str(imshape[1]) + ", " +
            str(imshape[0])
              )


        vertices = np.array([[(.51 * imshape[1], imshape[0] * .58), (.49 * imshape[1], imshape[0] * 0.58),
                              (0, imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
        image_mask = image_proc.region_of_interest(image_OTSU_canny, vertices);

        #plt.imshow(image_mask)
        #plt.show()

        #rho, theta, threshold, np.array([]), minLineLength=min_line_len,maxLineGap=max_line_gap
        print("Houghlines parameters ")
        print("Rho: " + str(1))
        print("Theta: " + str(np.pi / 180))
        print("Threshold: " + str(threshold))
        print("Min_line_length: " + str(min_line_length))
        print("Max_line_gap: " + str(max_line_gap))

        lines = image_proc.hough_lines(image_mask, 1, np.pi / 180, threshold, min_line_length, max_line_gap)
        # annotated_image = weighted_img(lines, img, α=0.8, β=1.)

        #plt.imshow(lines)
        #plt.show()

        result = image_proc.weighted_img(lines, img, α=0.8, β=1.)

        #plt.imshow(result)
        #plt.show()

        #b, g, r = cv2.split(result)
        #result = cv2.merge((b, g, r))

        cv2.imwrite("test_images/output_" + file, result)
        plt.imshow(result, cmap='gray')
        plt.show()

        #test and write various video sequences

        #Solid Yellow Left Video
        image_proc.writeVideoSequence('test_videos/solidYellowLeft.mp4', 'test_videos_output/solidYellowLeft_Output.mp4')

        #Solid White Right Video
        image_proc.writeVideoSequence('test_videos/solidWhiteRight.mp4', 'test_videos_output/solidWhiteRight_Output.mp4')

        #Challenge video
        image_proc.write_video_sequence('test_videos/challenge.mp4', 'test_videos_output/challengeOutput.mp4')
