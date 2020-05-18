#!/bin/env python
#
#  Copyright (c) 2020, Yasin Hasanian
#  See license.txt
#

import core.decomposer as decomposer
from common.logger import Logger

log = Logger()

def run(img_path, lights_limit=1, modes=[]):

    dcmp = decomposer.Decomposer(img_path)
    dcmp.preprocess()
    dcmp.decompose(lights_limit, modes)
    dcmp.export()

    return dcmp
