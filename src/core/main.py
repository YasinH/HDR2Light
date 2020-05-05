#!/bin/env python

import core.decomposer as decomposer
from common.logger import Logger

log = Logger()

def run(img_path, lights_limit=1, modes=[]):

    dcmp = decomposer.Decomposer(img_path)
    dcmp.preprocess()
    dcmp.decompose(lights_limit, modes)
    dcmp.export()

    return dcmp
