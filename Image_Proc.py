import numpy as np
import cv2

class Image_Proc:

    def __init__(self):
        t = 0

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    #Otsu's approached used as explained in https://gist.github.com/endless3cross3/2c3056aebef571c6de1016b2bbf2bdbf
    def otsu_canny(self, image, lowrate=0.1):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding
        ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
        edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

        # return the edged image
        return edged

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img

    # Python 3 has support for cool math symbols.

    def weighted_img(self, img, initial_img, α=0.8, β=1., γ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + γ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, γ)

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        """

        # Get the top-most point to draw the line 
        y_minimum = np.min(y_axis)

        # Draw the top most point via slope/intercept information obtained from linear regression   
        top_point = np.array([(y_minimum - cutoff) / slope, y_minimum], dtype=int)

        # Get the bottom left point
        max_y = np.max(y_axis)

        bottom_point = np.array([(max_y - cutoff) / slope, max_y], dtype=int)

        x1e, y1e = extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1], -1000) # bottom point

        x2e, y2e = extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1],  1000) # top point

        # return the line.

        line = np.array([[x1e,y1e,x2e,y2e]])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        #lines = np.array([line], dtype=np.int32)


        """
        # Get the slope of current line
        left_lines = []  # (slope, intercept)
        left_weights = []  # (length,)
        right_lines = []  # (slope, intercept)
        right_weights = []  # (length,)

        right_lines, left_lines = self.separate_lines(lines)

        if right_lines and left_lines:
            right = self.eliminate_outliers(right_lines, cutoff=(0.45, 0.75))
            x_axis, y_axis, slope, cutoff = self.linear_regr(right)
            print("Processing right lines")
            print("X-axis \t", x_axis, "\t Y-axis", y_axis, "\t Slope", slope, "\t Cuttoff", cutoff, "\n")

            left = self.eliminate_outliers(left_lines, cutoff=(-0.85, -0.6))
            x_axis, y_axis, slope, cutoff = self.linear_regr(left)
            print("Processing left lines")
            print("X-axis \t", x_axis, "\t Y-axis", y_axis, "\t Slope", slope, "\t Cuttoff", cutoff, "\n")

            # print("Line # ", count, "\n", line, "\n", right, "\n", left)
        # count = count + 1;

        # Get best line fit based on linear regression
        # x_axis, y_axis, slope, cutoff = linear_regr(lines)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def slope(self, x1, y1, x2, y2):
        return (y1 - y2) / (x1 - x2)

    """
    Separate the lines based on their slopes. Hence, left lane would have
    a slope less than zero and right lane would have a slope greater than 
    zero
    """

    def separate_lines(self, lines):
        right_lane = []
        left_lane = []

        for x1, y1, x2, y2 in lines[:, 0]:
            current_slope = self.slope(x1, y1, x2, y2)
            if current_slope >= 0:
                right_lane.append([x1, y1, x2, y2, current_slope])
            else:
                left_lane.append([x1, y1, x2, y2, current_slope])
        return right_lane, left_lane

    """
    Eliminate outliers with unusual slope that might dislocate the actual line
    """

    def eliminate_outliers(self, points, cutoff, threshold=0.08):
        points = np.array(points)
        points = points[(points[:, 4] >= cutoff[0]) & (points[:, 4] <= cutoff[1])]
        current_slope = np.mean(points[:, 4], axis=0)
        return points[(points[:, 4] <= current_slope + threshold) & (points[:, 4] >= current_slope - threshold)]

    """ Use linear regression to merge right and left lane sets by finding the 
    most optimal relationship between a group of points. this will give us a line
    that will pass closest to each
    """

    def linear_regr(self, lanes_array):
        x = np.reshape(lanes_array[:, [0, 2]], (1, len(lanes_array) * 2))[0]
        y = np.reshape(lanes_array[:, [1, 3]], (1, len(lanes_array) * 2))[0]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        x = np.array(x)
        y = np.array(x * m + c)
        return x, y, m, c

    def extend_point(self, x1, y1, x2, y2, length):
        line_len = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        x = x2 + (x2 - x1) / line_len * length
        y = y2 + (y2 - y1) / line_len * length
        return x, y

    def process_image(self, image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image where lines are drawn on lanes)

        gray = self.grayscale(image)
        gray = self.gaussian_blur(gray, 3)
        #image_canny = self.canny(gray, 50, 150)
        image_OTSU_canny = self.otsu_canny(gray, 0.2)

        imshape = image.shape
        vertices = np.array([[(.51 * imshape[1], imshape[0] * .58), (.49 * imshape[1], imshape[0] * 0.58), (0, imshape[0]),
                              (imshape[1], imshape[0])]], dtype=np.int32)
        image_mask = self.region_of_interest(image_OTSU_canny, vertices);

        hough_lines1 = self.hough_lines(image_mask, 1, np.pi / 180, 35, 5, 2)
        annotated_image = self.weighted_img(hough_lines1, image, α=0.8, β=1.)

        return annotated_image

    def filter_color(self, image):
        """
        filter the image to mask out darker shades and get lighter pixels only
        """
        # Convert to HSV first
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        filter_light_colors = cv2.inRange(hsv_image, np.array([0, 0, 210]), np.array([255, 255, 255]))
        return cv2.bitwise_and(image, image, mask=filter_light_colors)

    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


