#
#  Copyright (c) 2020, Yasin Hasanian
#  See license.txt
#

import cv2
import numpy as np

def shiftImage(img, sx, sy):

    return np.roll(img, (sx, sy), axis=(1, 0))

def shiftImageInside(img):

    height, width = img.shape[:2]
    thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                            .1, 255, cv2.THRESH_BINARY)[1]

    #count number of white pixels in columns as new array
    count = np.count_nonzero(thresh, axis=0)
    # get first and last x coordinate where black (count==0)
    first_black = np.where(count==0)[0][0]
    last_black = np.where(count==0)[0][-1]
    # compute x center
    black_center = (first_black + last_black) // 2
    # crop into two parts
    left = img[0:height, 0:black_center]
    right = img[0:height, black_center:width]
    # combine them horizontally after swapping
    shift_horiz = np.hstack([right, left])

    # repeat vertically
    count = np.count_nonzero(thresh, axis=1)
    first_black = np.where(count==0)[0][0]
    last_black = np.where(count==0)[-1][0]
    black_center = (first_black + last_black) // 2
    top = shift_horiz[0:black_center, 0:width]
    bottom = shift_horiz[black_center:height, 0:width]
    shift_vert = np.vstack([bottom, top])

    return shift_vert

def ranges(nums):

    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])

    return list(zip(edges, edges))

def maxRange(num_list):

    max_range = ()
    tmp = 0
    for l in num_list:
        if len(range(*l)) > tmp:
            max_range = l

    return max_range

def removeListFromList(a, b):

    for x in b:
        try:
            a.remove(x)
        except ValueError:
            pass
    return a
