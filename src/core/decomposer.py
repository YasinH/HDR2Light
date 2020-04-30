"""
Main image processor that decomposes lights from HDR images
"""

import utils
import common.constants as constants

import cv2
import numpy as np
import math
from pathlib import Path
import sys
import logging


class Decomposer(object):

    def __init__(self, img_path=''):

        self._img_path = Path(img_path)
        self._img_in = self.__readImage(self._img_path)
        self._img_height, self._img_width = self._img_in.shape[:2]
        self._img_copy = self._img_in.copy()

        self._lights = []
        self._envs = []
        self._labels = []
        self._ret = 0
        self._stats = None
        self._centroids = None
        self._sorted_rets = []

    @property
    def lights(self):
        return self._lights

    @lights.setter
    def lights(self, lights):
        self._lights = lights

    @property
    def envs(self):
        return self._envs

    @envs.setter
    def envs(self, envs):
        self._envs = envs

    def isSimilarTo(self, l1, l2):
        '''
        Checks whether two labels are similar
        :param l1: First label
        :param l2: Second label
        :return: True if both are similar, otherwise false
        '''
        shared = set(l1) & set(l2)
        diff = float(abs(len(l1) - len(l2))) / float(len(l1)) * 100
        if diff < 10:
            return True
        return False


    def pixelsOnBorder(self, label_id):
        '''
        Identifies which label pixels are located on image borders
        :param label_id: Label ID
        :return: List of pixels are on borders as a tuple
        '''
        y_pixels = np.where(self._labels == label_id)[0]
        x_pixels = np.where(self._labels == label_id)[1]

        x_borders = {'bottom': y_pixels[np.where(self._labels == label_id)[1]==self._img_width-1].tolist(),
                    'top': y_pixels[np.where(self._labels == label_id)[1]==0].tolist()}
        y_borders = {'left': x_pixels[np.where(self._labels == label_id)[0]==self._img_height-1].tolist(),
                    'right': x_pixels[np.where(self._labels == label_id)[0]==0].tolist()}

        return x_borders, y_borders


    def findLight(self, img, light_no=1):
        '''
        Detects light sources from image
        :param img: Input image to find light on
        :param light_no: What light number to decompose
        :return: ImageLight instance
        '''
        # TO-DO: img should match class img!! This method needs rethinking overall
        if light_no > len(self._sorted_rets):
            return

        hot_labels_idx = []
        x_borders = {'bottom':[], 'top':[]}
        y_borders = {'left':[], 'right':[]}
        x_borders, y_borders = self.pixelsOnBorder(light_no)
        hot_labels_idx.append(light_no)

        # look for other continuous labels
        is_on_border = any(x_borders.values()) or any(y_borders.values())
        if is_on_border:
            for i in self._sorted_rets:
                if i in hot_labels_idx:
                    continue
                x_borders_ext = {'bottom':[], 'top':[]}
                y_borders_ext = {'left':[], 'right':[]}
                x_borders_ext, y_borders_ext = self.pixelsOnBorder(i)
                if x_borders_ext['top']:
                    if self.isSimilarTo(x_borders['bottom'], x_borders_ext['top']):
                        hot_labels_idx.append(i)
                if x_borders_ext['bottom']:
                    if self.isSimilarTo(x_borders['top'], x_borders_ext['bottom']):
                        hot_labels_idx.append(i)
                if y_borders_ext['left']:
                    if self.isSimilarTo(y_borders['right'], y_borders_ext['left']):
                        hot_labels_idx.append(i)
                if y_borders_ext['right']:
                    if self.isSimilarTo(y_borders['left'], y_borders_ext['right']):
                        hot_labels_idx.append(i)

            pos_img = self._getPosition(hot_labels_idx[:1]), self._img_copy
        else:
            pos_img = self._getPosition(hot_labels_idx), self._img_copy
        # TO-DO: fix with proper logger
        if len(hot_labels_idx) > 4:
            raise Exception ("Too many labels")

        light = ImageLight(hot_labels_idx)
        light.position = pos_img

        return light


    def sortComponents(self):
        '''
        Sorts label IDs based on their energy
        '''
        labels_energy = []
        img_hsv = cv2.cvtColor(self._img_copy, cv2.COLOR_BGR2HSV)

        for i in range(1, self._ret):
            energy = 0
            x_pixels, y_pixels = np.where(self._labels == i)
            for x, y in zip(x_pixels, y_pixels):
                energy += img_hsv[x, y][2]
            labels_energy.append((i, energy))

        labels_energy.sort(key= lambda x: x[1], reverse=1)
        self._sorted_rets = [i[0] for i in labels_energy]


    def preprocess(self):
        '''
        Preprocesses image and initializes certain parameters
        '''
        if self._img_height > 2048:
            self._img_copy = cv2.resize(self._img_copy, (0,0), fx=self._img_width/2, fy=self._img_height/2)

        height, width = self._img_copy.shape[:2]
        img_gray = cv2.cvtColor(self._img_copy, cv2.COLOR_BGR2GRAY)

        gray_blr = cv2.GaussianBlur(img_gray, (5, 5), 0)

        img_thresh = cv2.threshold(gray_blr, 1, 1, cv2.THRESH_BINARY)[1]
        img_thresh = cv2.erode(img_thresh, None, iterations=2)
        img_thresh = cv2.dilate(img_thresh, None, iterations=4)

        self._ret, self._labels, self._stats, self._centroids = cv2.connectedComponentsWithStats(np.uint8(img_thresh))

        self.sortComponents()

        visited_labels = set()
        for id in self._sorted_rets:
            if id in visited_labels:
                continue
            light = self.findLight(gray_blr, id)
            self._lights.append(light)
            visited_labels.update(light.labels)

    def borderBlackPixels(self, img):
        '''
        Black pixels on image borders
        :param img: Input image
        :return: A tuple of (whether the image has pixels on borders, pixels on X axis, pixels on Y axis)
        '''
        eps = 0.05
        height, width = img.shape[:2]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        x_borders = {'bottom':[], 'top':[]}
        y_borders = {'left':[], 'right':[]}

        for x in range(width):
            if img_gray[0][x] > eps:
                x_borders['bottom'].append(x)
            if img_gray[height-1][x] > eps:
                x_borders['top'].append(x)
        for y in range(height):
            if img_gray[y][0] > eps:
                y_borders['left'].append(y)
            if img_gray[y][width-1] > eps:
                y_borders['right'].append(y)

        has_border_pixels = any(x_borders.values()) or any(y_borders.values())

        for xb, yb in zip(x_borders, y_borders):
            x_borders[xb] = utils.removeListFromList(range(width), x_borders[xb])
            y_borders[yb] = utils.removeListFromList(range(height), y_borders[yb])

        return has_border_pixels, x_borders, y_borders


    def cropToArea(self, img):
        '''
        Crops image to non-black area while ensuring features are shifted inside frames
        :param img: Input image
        :return: Cropped image
        '''
        has_border_pixels, x_borders, y_borders = self.borderBlackPixels(img)
        img_offset = img

        if has_border_pixels:
            img_offset = utils.shiftImageInside(img_offset)

        img_cropped = img_offset[np.ix_(cv2.cvtColor(img_offset, cv2.COLOR_BGR2GRAY).any(1),
                                cv2.cvtColor(img_offset, cv2.COLOR_BGR2GRAY).any(0))]

        return img_cropped


    def decompose(self, lights_limit=1, modes=[]):
        '''
        Main method to decompose lights from HDR image
        :param lights_limit: Number of lights to extract.
                            This is limited to maximum number of connected components
        :param modes: A list of 0 or 1 for each extracted lights to set their type.
                    0 for skydome mode, 1 for area mode
        '''
        if lights_limit < 1 :
            return

        lights_count = int(min(lights_limit, len(self._lights)))
        height, width = self._img_copy.shape[:2]
        lights_mask = np.uint8(np.zeros(self._labels.shape[:2]))

        for idx in xrange(lights_count):

            img_hsv = cv2.cvtColor(self._img_copy, cv2.COLOR_BGR2HSV)

            hot_label_mask = np.uint8(np.zeros(self._labels.shape[:2]))
            for i in self._lights[idx].labels:
                hot_label_mask[np.where(self._labels == i)] = 255
                lights_mask[np.where(self._labels == i)] = 255

            hot_label_blur = cv2.GaussianBlur(hot_label_mask, (25, 25), 0)

            for i in range(width):
                for j in range(height):
                    if hot_label_blur[j, i] > 0:
                        if img_hsv[j, i][2] > 1:
                            img_hsv[j, i][2] = 1

            key = np.zeros((height,width,3), np.float32)
            hot_label_blur_32 = (hot_label_blur).astype('float32')/255
            key = self._img_copy * hot_label_blur_32[:, :, None]

            if modes:
                if modes[idx] == constants.AREA_MODE and idx < len(modes):
                    key = self.cropToArea(key)
                    self._lights[idx].mode = constants.AREA_MODE

            l_height, l_width = key.shape[:2]
            self._lights[idx].img = key
            self._lights[idx].scale_world = (l_width/float(width), l_height/float(height))

        env = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        lights_mask = cv2.dilate(lights_mask, None, iterations=10)
        env_clamped = np.clip(env, 0, 1)
        env_8 = np.uint8(env_clamped * 255).astype('uint8')
        inpaint = cv2.inpaint(env_8, lights_mask, 3, cv2.INPAINT_TELEA)
        inpaint = (inpaint).astype('float32')/255
        env[np.where(lights_mask == 255)] = inpaint[np.where(lights_mask == 255)]

        env_light = ImageLight([0])
        env_light.img = env
        self._envs.append(env_light)


    def export(self):
        '''
        Exports all decomposed lights to disk
        '''
        if not self._lights or not self._envs:
            return

        out_dir = self._img_path.parent

        lights_to_export = [il for il in self._lights if il.img is not None]
        # Key lights
        for idx, light in enumerate(lights_to_export):
            out_file = out_dir / (self._img_path.stem + '_key_' + format(idx+1, '03') + self._img_path.suffix)
            light.img_path = out_file
            cv2.imwrite(str(light.img_path), light.img)

        # Env light
        out_file = out_dir / (self._img_path.stem + '_env' + self._img_path.suffix)
        cv2.imwrite(str(out_file), self._envs[0].img)
        self._envs[0].img_path = out_file


    def _getPosition(self, label_ids, method=0):

        if method == 0:
            x = 0
            y = 0
            for id in label_ids:
                x += (self._stats[id, cv2.CC_STAT_LEFT] + self._stats[id, cv2.CC_STAT_WIDTH] // 2)
                y += (self._stats[id, cv2.CC_STAT_TOP] + self._stats[id, cv2.CC_STAT_HEIGHT] // 2)

            return x, y

        else:
            label_img = np.uint8(np.zeros(self._labels.shape[:2]))
            for id in label_ids:
                label_img[np.where(self._labels == id)] = 255

            label_img = (label_img).astype('float32')/255
            label_img = self._img_copy * label_img[:, :, None]
            # label_img = cv2.GaussianBlur(label_img, (5, 5), 0)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY))
            return max_loc[0], max_loc[1]

    def __readImage(self, img_path):
        # TO-DO: Complete this
        if not img_path.is_file():
            raise Exception('invalid input image')
        img = cv2.imread(str(img_path), -1)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
        # if max_val <= 1:
        #     raise Exception('invalid input image')

        return img

    # def isOnBorder(labels):
    #
    #     thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    #                                     .1, 255, cv2.THRESH_BINARY)[1]
    #
    #     count = np.count_nonzero(thresh, axis=0)
    #     blacks = np.where(count==0)
    #     if blacks[0][0] != 0 and blacks[0][-1] != img.shape[1]-1:
    #         return 1
    #
    #     count = np.count_nonzero(thresh, axis=1)
    #     blacks = np.where(count==0)
    #     if blacks[0][0] != 0 and blacks[-1][0] != img.shape[0]-1:
    #         return 1
    #
    #     return 0


class ImageLight(object):

    def __init__(self, labels, mode=constants.SKYDOME_MODE):

        self._labels = labels
        self._mode = mode
        self._img = None
        self._img_path = None
        self._position = (0, 0)
        self._uv = (0, 0)
        self._scale_world = (1, 1)
        self._ratio = 2
        self._isOnBorder = False

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        self._isOnBorder = True if len(labels) > 1 else False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        self._img = img
        self._ratio = img.shape[1]/float(img.shape[0])

    @property
    def img_path(self):
        return self._img_path

    @img_path.setter
    def img_path(self, img_path):
        self._img_path = img_path

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position_img):
        self._position, img_world= position_img[0], position_img[1]
        self._uv = (self._position[0]/float(img_world.shape[1]),
                    self._position[1]/float(img_world.shape[0]))

    @property
    def uv(self):
        return self._uv

    @property
    def scale_world(self):
        return self._scale_world

    @scale_world.setter
    def scale_world(self, scale_world):
        self._scale_world = scale_world

    @property
    def ratio(self):
        return self._ratio

    def __eq__(self, other):
        if isinstance(other, Light):
            return self.labels == other.labels
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
