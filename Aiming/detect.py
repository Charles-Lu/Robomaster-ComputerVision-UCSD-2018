import math
import os
import time
import cv2
import numpy as np
import cProfile

from Aiming.classifier import CNN

# =================================== SETTING ===================================

# functional setting
SHOW_ALL = True  # show all debug images
SHOW_RESULT = True or SHOW_ALL  # show only the final result
FRAME_PAUSE = True  # press "enter" to go to next frame
SAVE = False  # save armor samples with prediction
SAVE_PATH = "../Aiming/data"  # the path to save folder; need to change if run from main.py
INPUT_SIZE = (640, 360)  # input resolution; optimized for speed
LEFT_RIGHT_CROP = (INPUT_SIZE[0] - INPUT_SIZE[1] // 3 * 4) // 2  # crop to 4:3
TOP_CROP = INPUT_SIZE[1] // 5  # crop the top 1/5
FINAL_RESOLUTION = (INPUT_SIZE[0] - 2 * LEFT_RIGHT_CROP, INPUT_SIZE[1] - TOP_CROP)  # final resolution after crop

# parameter setting
DEFAULT_RED_RANGE = [[(160, 0, 190), (180, 150, 255)],
                     [(0, 0, 190), (20, 150, 255)],
                     [(0, 0, 230), (180, 255, 255)]]
DEFAULT_SOBEL_THRESH = [240, 255]
DEFAULT_FAR_SOBEL_CLOSURE_SIZE = (3, 3)
DEFAULT_CLOSE_SOBEL_CLOSURE_SIZE = (9, 5)
DEFAULT_CLOSE_CLOSURE_SIZE = (5, 3)
DEFAULT_CLOSE_DILATE_SIZE = (1, 3)
DEFAULT_FAR_CLOSURE_SIZE = (9, 4)

# =================================== END ===================================


def detect_main(mem):
    """
    The main function called for detect thread.
    :param mem: The dictionary shared by all threads in this project.
    :param mode: "cnn"
    """

    # instantiate ArmorLocator
    locator = ArmorLocator("red")

    # wait and read frame from dict
    while "current_frame" not in mem:
        print("wait for camera")
    while True:
        frame_read, frame_t = mem["current_frame"]
        src = cv2.resize(frame_read, INPUT_SIZE)
        src = src[TOP_CROP:, LEFT_RIGHT_CROP:-LEFT_RIGHT_CROP]
        locator.locate(src)


class SampleSavor:
    def __init__(self, path):
        """
        Construct a SampleSavor object to handle predicted sample saving.
        :param path: the path that samples will be saved to
        """
        self.path = path
        self.red_path = os.path.join(self.path, "_red")
        self.blue_path = os.path.join(self.path, "_blue")
        self.negative_path = os.path.join(self.path, "_negative")
        if not os.path.exists(self.red_path):
            os.makedirs(self.red_path)
        if not os.path.exists(self.blue_path):
            os.makedirs(self.blue_path)
        if not os.path.exists(self.negative_path):
            os.makedirs(self.negative_path)

    def save_red(self, sample):
        """
        Save sample with red prediction
        :param sample: the image sample
        """
        cv2.imwrite(os.path.join(self.red_path, "red_" + str(time.time()) + ".jpg"), sample)

    def save_blue(self, sample):
        """
        Save sample with blue prediction
        :param sample: the image sample
        """
        cv2.imwrite(os.path.join(self.blue_path, "blue_" + str(time.time()) + ".jpg"), sample)

    def save_negative(self, sample):
        """
        Save sample with negative prediction
        :param sample: the image sample
        """
        cv2.imwrite(os.path.join(self.negative_path, "negative_" + str(time.time()) + ".jpg"), sample)


class PotentialLight:
    def __init__(self, rect):
        """
        Construct a PotentialLight object to store possible light bar detected
        :param rect: the return value of cv2.minAreaRect() that bounds the light bar
        """
        self.rect = rect
        self.center, self.size, self.angle = rect
        self.points = None  # four vertices of the rectangle bounding the light bar
        self.centerline = None  # the vertical center line of light bar

    def get_points(self):
        """
        Return four vertices of bounding rectangle
        :return: vertices array of shape (4,)
        """
        if self.points is None:
            self.points = np.int0(cv2.boxPoints(self.rect))
        return self.points

    def get_centerline(self):
        """
        Calculate the median points of two shortest edges of rectangle.
        Return the centerline passing the two points.
        :return: centerline of format [[x0, y0], [x1, y1], angle]
        """
        if self.points is None:
            self.get_points()

        if self.centerline is None:
            p1 = self.points[0]
            p2 = self.points[1]
            p3 = self.points[2]
            p4 = self.points[3]

            length = [[(p1, p2), np.hypot(p1[0] - p2[0], p1[1] - p2[1])],
                      [(p2, p3), np.hypot(p2[0] - p3[0], p2[1] - p3[1])],
                      [(p3, p4), np.hypot(p3[0] - p4[0], p3[1] - p4[1])],
                      [(p4, p1), np.hypot(p4[0] - p1[0], p4[1] - p1[1])]]
            length = sorted(length, key=lambda x: x[1])
            x1 = (length[0][0][0][0] + length[0][0][1][0]) // 2
            y1 = (length[0][0][0][1] + length[0][0][1][1]) // 2
            x2 = (length[1][0][0][0] + length[1][0][1][0]) // 2
            y2 = (length[1][0][0][1] + length[1][0][1][1]) // 2
            degree = np.degrees(math.atan2(y1 - y2, x1 - x2))
            if degree < 0:
                degree = 180 + degree
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1

            self.centerline = [[x1, y1], [x2, y2], 180 - degree]
        return self.centerline

    def get_center(self):
        """
        Return the center of rectangle
        :return: center point of format (x, y)
        """
        return self.center

    def check_validity(self):
        """
        Return true if the potential light bar passes all rules; return false otherwise
        :return: whether the light bar is valid
        """

        # light bar should be generally vertical
        if 15 < self.angle < 75 or -75 < self.angle < -15:
            return False

        # the ratio of width and height should remain in proper range
        if self.size[0] > self.size[1] * 10 or self.size[1] > self.size[0] * 10:
            return False

        # width < height
        if (self.size[0] > self.size[1] and -15 < self.angle < 15) \
                or (self.size[0] < self.size[1] and (self.angle < -75 or self.angle > 75)):
            return False

        # light bar should not be too small
        area = self.size[0] * self.size[1]
        if area < 16 or area > 32400:
            return False
        return True


class PotentialArmor:
    def __init__(self, light1=None, light2=None, rect=None):
        """
        Construct a PotentialArmor object to store possible armor area detected.
        It supports two forms of inputs:
            a) two light bars that bounds the armor (close armor)
            b) the return of cv2.minRectArea() that bounds the armor (far armor)
        :param light1: the first light bar
        :param light2: the second light bar
        :param rect: the rectangle bounding the armor
        """

        # the two forms of inputs should not appear together
        assert (light1 is not None and light2 is not None and rect is None) \
               or (rect is not None and light1 is None and light2 is None)

        # construct armor based on two light bars
        if None not in (light1, light2):
            self.p_11, self.p_12, self.angle1 = light1.get_centerline()
            self.center1 = light1.get_center()
            self.p_21, self.p_22, self.angle2 = light2.get_centerline()
            self.center2 = light2.get_center()
            self.category = "close"

        # construct armor based on rectangle
        elif rect is not None:
            self.rect = rect
            self.p_11, self.p_12, self.p_22, self.p_21 = cv2.boxPoints(rect)
            self.center, self.size, self.angle = rect
            self.category = "far"

        self.warped = None  # the image of rectangle after perspective correction
        self.dst = np.float32([[0., 0.], [0., 96.], [96., 96.], [96., 0.]])  # the target armor shape
        self.confidence = 0.0  # the confidence score returned by cnn
        self.label = -1  # the label predicted by cnn

    def check_validity(self, src):
        """
        Return true if the potential armor passes all rules; return false otherwise
        :param src: the original frame; used for color checking
        :return: whether the light bar is valid
        """

        # rules for close armor
        if self.category == "close":

            # difference between angle of two light bars should be small
            diff_angle = abs(self.angle1 - self.angle2)
            if diff_angle > 10 and (self.p_12[1] - self.p_11[1] > 15 or self.p_22[1] - self.p_21[1] > 15):
                return False

            # x position should not be too close
            x_diff = self.center1[0] - self.center2[0]
            if abs(x_diff) < 15:
                return False

            # difference in y / difference in x > 0.5
            y_diff = self.center1[1] - self.center2[1]
            if abs(y_diff / x_diff) > 0.5:
                return False

            # length ratio should not be large
            ratio = (self.p_12[1] - self.p_11[1]) / (self.p_22[1] - self.p_21[1])
            if ratio > 2 or ratio < 0.5:
                return False

            # height * possible ratio < width
            # possible ratio = (height / frame_height) * (-3) + 4
            avg_h = (self.p_12[1] - self.p_11[1] + self.p_22[1] - self.p_21[1]) / 2
            avg_w = (abs(self.p_11[0] - self.p_21[0]) + abs(self.p_12[0] - self.p_22[0])) / 2
            if avg_h * (avg_h / 480 * (-3) + 4) < avg_w:
                return False

            # TODO: I forgot what it is...
            if max(abs(self.p_11[1] - self.p_21[1]), abs(self.p_12[1] - self.p_22[1])) > (
                    abs(self.p_11[0] - self.p_21[0]) + abs(self.p_12[0] - self.p_22[0])) / 2:
                return False

            # color should be generally red
            if not self.check_warped(src):
                return False

        # rules for far armor
        elif self.category == "far":

            # ratio of width and height should remain in reasonable range
            if self.size[0] > self.size[1] * 5 or self.size[1] > self.size[0] * 5:
                return False

            # if rectangle is horizontal (suggesting a clear boundary), apply stronger criteria on ratio
            if (self.size[0] * 1.5 < self.size[1] and -25 < self.angle < 25) or (
                    self.size[0] > self.size[1] * 1.5 and (self.angle < -65 or self.angle > 65)):
                return False

            # armor should not be too small
            if self.size[0] * self.size[1] < 64:
                return False

            # armor should not have large ratio and large angel at the same time
            if (self.size[0] > self.size[1] * 2 or self.size[1] > self.size[0] * 2) \
                    and (25 < self.angle < 65 or -65 < self.angle < -25):
                return False

            # color should be generally red
            if not self.check_warped(src):
                return False
        return True

    def get_points(self):
        """
        Return four vertices of armor
        :return: vertices of shape (4,)
        """
        return np.int0([self.p_11, self.p_12, self.p_22, self.p_21])

    def get_floatpoints(self):
        """
        Return four vertices of armor in float type
        :return: vertices of shape (4,) in float type
        """
        return np.float32(self.get_points())

    def check_warped(self, src):
        """
        Warp the armor to a square by perspective transformation
        Check the general color of armor area
        :param src: original frame
        :return: whether the area is generally red
        """
        M = cv2.getPerspectiveTransform(self.get_floatpoints(), self.dst)
        self.warped = cv2.warpPerspective(src, M, (96, 96))
        r_sum = np.sum(self.warped[..., 2])
        g_sum = np.sum(self.warped[..., 1])
        b_sum = np.sum(self.warped[..., 0])
        if r_sum < g_sum - 230400 or r_sum < b_sum - 230400:
            return False
        return True

    def get_warped(self, src):
        """
        Return the armor image warped to a square
        :param src: original frame
        :return: the warped armor image
        """
        if self.warped is None:
            self.check_warped(src)
        return self.warped

    def register(self, label, confidence):
        """
        Register the label of potential armor (NEGATIVE, RED, BLUE) and corresponding confidence
        :param label: predicted label of current armor
        :param confidence: confidence of prediction
        """
        self.label = label
        if confidence is not None:
            self.confidence = confidence[label]


# TODO: implement blue armor detection
class ArmorLocator:
    def __init__(self, enemy):
        """
        Construct a ArmorLocator object to handle the detection task.
        It is the the main class structure of detection thread.
        :param enemy: "red" or "blue"; the color of armor we need to detect
        """
        self.red_range = DEFAULT_RED_RANGE
        self.sobel_thresh = DEFAULT_SOBEL_THRESH
        self.close_closure_size = DEFAULT_CLOSE_CLOSURE_SIZE
        self.close_dilate_size = DEFAULT_CLOSE_DILATE_SIZE
        self.far_closure_size = DEFAULT_FAR_CLOSURE_SIZE

        self.clf = CNN("MobileNet")
        self.clf.train("load")

        self.savor = SampleSavor(SAVE_PATH)
        self.enemy = enemy
        self.img = None  # original frame

        self.targets = []
        self.debug_img = None  # the mat we used to draw final debug info

    def locate(self, src):
        """
        The main pipeline for armor detection. It is called on every frame.
        :param src: original frame
        :return: None (for now)
        """

        # initialization
        self.img = src
        self.debug_img = self.img.copy()

        # (for red armor) use blue and green channel to generate grey image
        # since red channel is too glowing
        b, g, r = cv2.split(self.img)
        grey = np.uint8(np.add(b, g, dtype=np.uint16) // 2)

        # ensemble masks
        red_mask = self.create_red_mask(self.img)
        sobel_mask = self.create_sobel_mask(grey)

        far_cnts = self.find_far_contours(sobel_mask, red_mask)
        close_cnts = self.find_close_contours(sobel_mask, red_mask)

        # detect armor based on light bar
        lights = self.detect_light(close_cnts)
        armors = []
        armors += self.detect_close_armor(lights)

        # detect armor based on overall shape
        armors += self.detect_far_armor(far_cnts)

        if armors is None:
            return

        # warp armor image into square shape by perspective transformation
        warped_imgs = self.warp_armor(armors)

        # use cnn to classify armor candidate
        self.targets = self.classify_targets(warped_imgs, armors)

        # show debug info
        if SHOW_ALL:
            cv2.imshow("red_mask", red_mask)
        if SHOW_RESULT:
            debug = cv2.resize(self.debug_img, FINAL_RESOLUTION)
            cv2.imshow("final", debug)
            if FRAME_PAUSE:
                cv2.waitKey(0)
            else:
                if cv2.waitKey(33) & 0xff == ord('q'):
                    pass
        self.refresh()

    def find_far_contours(self, sobel_mask, red_mask):
        """
        Generate the contours of far object
        :param sobel_mask: original sobel mask of image
        :param red_mask: original red mask of image
        :return: the contour of far objects
        """
        far_sobel_mask = cv2.morphologyEx(sobel_mask, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, DEFAULT_FAR_SOBEL_CLOSURE_SIZE))
        far_mixed_mask = far_sobel_mask & red_mask
        far_closure = self.create_far_closure(far_mixed_mask)
        _, far_cnts, _ = cv2.findContours(far_closure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if SHOW_ALL:
            cv2.imshow("far_sobel_mask", far_sobel_mask)
            cv2.imshow("far_mixed_mask", far_mixed_mask)
            cv2.imshow("far_closure", far_closure)

        return far_cnts

    def find_close_contours(self, sobel_mask, red_mask):
        """
        Generate the contours of close object
        :param sobel_mask: original sobel mask of image
        :param red_mask: original red mask of image
        :return: the contour of close objects
        """
        close_sobel_mask = cv2.morphologyEx(sobel_mask, cv2.MORPH_CLOSE,
                                            cv2.getStructuringElement(cv2.MORPH_RECT, DEFAULT_CLOSE_SOBEL_CLOSURE_SIZE))
        close_mixed_mask = close_sobel_mask & red_mask
        close_closure = self.create_close_closure(close_mixed_mask)
        _, close_cnts, _ = cv2.findContours(close_closure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if SHOW_ALL:
            cv2.imshow("close_sobel_mask", close_sobel_mask)
            cv2.imshow("close_mixed_mask", close_mixed_mask)
            cv2.imshow("close_closure", close_closure)

        return close_cnts

    def create_red_mask(self, src):
        """
        Generate mask of red region from original image
        :param src: original image
        :return: the mask of red region
        """
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, self.red_range[0][0], self.red_range[0][1])
        red2 = cv2.inRange(hsv, self.red_range[1][0], self.red_range[1][1])
        red3 = cv2.inRange(hsv, self.red_range[2][0], self.red_range[2][1])  # include all bright area
        red_mask = red1 | red2 | red3
        return red_mask

    def create_sobel_mask(self, grey):
        """
        Calculate Sobel gradient in x-direction on grey image
        Generate mask of area with extremely high Sobel gradient
        :param grey: grey image
        :return: mask of high Sobel gradient area
        """
        grad = cv2.Sobel(grey, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3, scale=4)  # use 16 bits to prevent overflow
        abs_16 = np.abs(grad) - 1
        abs_8 = (abs_16 / 256).astype('uint8')  # transform back to 8 bits
        hist = cv2.equalizeHist(abs_8)
        _, thresh = cv2.threshold(hist, self.sobel_thresh[0], self.sobel_thresh[1], cv2.THRESH_BINARY)
        return thresh

    def create_close_closure(self, red):
        """
        Apply morphological operation to red mask for light detection
        :param red: red mask
        :return: mask after operation
        """
        close_closure_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.close_closure_size)
        close_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.close_dilate_size)
        close_closure = cv2.morphologyEx(red, cv2.MORPH_CLOSE, close_closure_kernel)
        close_dilate = cv2.dilate(close_closure, close_dilate_kernel)
        return close_dilate

    def create_far_closure(self, mixed_mask):
        """
        Apply morphological operation to mixed mask for approximate detection of armor far away
        :param mixed_mask: the intersection of red and Sobel mask
        :return: mask after operation
        """
        far_closure_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.far_closure_size)
        far_closure = cv2.morphologyEx(mixed_mask, cv2.MORPH_CLOSE, far_closure_kernel)
        return far_closure

    def detect_light(self, close_cnts):
        """
        Select potential light bars based on contours
        :param close_cnts: the contour generated from red mask
        """
        lights = []
        for close_cnt in close_cnts:
            rect = cv2.minAreaRect(close_cnt)
            light = PotentialLight(rect)
            if not light.check_validity():
                continue
            if SHOW_RESULT:
                box = light.get_points()
                self.debug_img = cv2.drawContours(self.debug_img, [box], 0, (255, 0, 0), 2)
            lights.append(light)
        return lights

    def detect_close_armor(self, lights):
        """
        Select potential armor bounded by two potential light bars.
        """
        armors = []
        for i in range(len(lights)):
            for j in range(i):
                armor = PotentialArmor(light1=lights[i], light2=lights[j])
                if not armor.check_validity(self.img):
                    continue
                armors.append(armor)
                if SHOW_RESULT:
                    box = armor.get_points()
                    self.debug_img = cv2.drawContours(self.debug_img, [box], 0, (0, 255, 0), 1)
        return armors

    def detect_far_armor(self, far_cnts):
        """
        Select potential far armor based on contours
        :param far_cnts: the contour generated from mixed mask
        """
        armors = []
        for far_cnt in far_cnts:
            rect = cv2.minAreaRect(far_cnt)
            armor = PotentialArmor(rect=rect)
            if not armor.check_validity(self.img):
                continue
            armors.append(armor)
            if SHOW_RESULT:
                box = armor.get_points()
                self.debug_img = cv2.drawContours(self.debug_img, [box], 0, (0, 255, 255), 1)
        return armors

    def warp_armor(self, armors):
        """
        Warp the sheared armor image to a square by perspective transformation
        :return: the square image after transformation
        """
        warpeds = []
        for armor in armors:
            warpeds.append(armor.get_warped(self.img))
        if len(warpeds) == 0:
            return None
        return warpeds

    def classify_targets(self, warpeds, armors):
        """
        Call classfier to predict the label of potential armor image
        :param warpeds: the list of warped armor image
        """
        red_armors = []
        blue_armors = []

        predictions, possibilities = self.clf.predict(warpeds)
        for i in range(len(armors)):
            prediction = predictions[i]
            if possibilities is None:
                possibility = None
            else:
                possibility = possibilities[i]
            armor = armors[i]
            armor.register(prediction, possibility)
            warped = warpeds[i]
            if predictions[i] == 0:
                if SHOW_RESULT:
                    self.debug_img = cv2.drawContours(self.debug_img, [armor.get_points()], 0, (0, 0, 255), 1)
                if SAVE:
                    self.savor.save_negative(warped)
            elif predictions[i] == 1:
                red_armors.append(armor)
                if SHOW_RESULT:
                    self.debug_img = cv2.drawContours(self.debug_img, [armor.get_points()], 0, (0, 255, 0), 1)
                if SAVE:
                    self.savor.save_red(warped)
            elif predictions[i] == 2:
                blue_armors.append(armor)
                if SHOW_RESULT:
                    self.debug_img = cv2.drawContours(self.debug_img, [armor.get_points()], 0, (0, 255, 0), 1)
                if SAVE:
                    self.savor.save_blue(warped)
        if self.enemy == "red":
            return red_armors
        else:
            return blue_armors

    # TODO: implement Nonmax Suppression (NMS)
    # def nonmax_suppress(self):
    #     enemy_armors.sort(key=lambda x: x.confidence)

    def refresh(self):
        """
        Reset all variables.
        """
        self.debug_img = None


if __name__ == '__main__':
    al = ArmorLocator("red")
    mode = "image"
    pr = cProfile.Profile()
    pr.enable()
    if mode == "image":
        # img = cv2.imread("../asset/vlcsnap-2019-07-28-19h56m57s214.png")
        # img = cv2.imread("../asset//vlcsnap-2019-07-28-18h53m07s609.png")
        img = cv2.imread("../asset/vlcsnap-2019-07-31-18h26m31s945.png")
        # img = cv2.imread("../asset/vlcsnap-2019-08-18-20h26m58s868.png")
        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        img = cv2.flip(img, 1)

        img = img[TOP_CROP:, LEFT_RIGHT_CROP:-LEFT_RIGHT_CROP]
        for i in range(1):
            al.locate(img)

    elif mode == "video":
        # cap = cv2.VideoCapture("../WIN_20190724_18_47_15_Pro.mp4")
        # cap = cv2.VideoCapture("../WIN_20190726_23_58_14_Pro.mp4")
        cap = cv2.VideoCapture("../asset/720ptest.mp4")
        # cap = cv2.VideoCapture(2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        while cap.isOpened():
            count += 1
            ret, frame = cap.read()

            # EDIT if you want to play longer
            if count == 1000:
                break
            if not ret:
                break
            img = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
            img = cv2.flip(img, 1)

            img = img[TOP_CROP:, LEFT_RIGHT_CROP:-LEFT_RIGHT_CROP]
            al.locate(img)
    pr.disable()
    pr.print_stats(sort='time')
