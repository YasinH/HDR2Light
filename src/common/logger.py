#
#  Copyright (c) 2020, Yasin Hasanian
#  See license.txt
#

import logging
import sys

class Logger(object):

    def __init__(self, name='[HDR2Light]', level=logging.DEBUG):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.logger.handlers=[]
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.handler_set = True

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def exception(self, msg):
        self.logger.exception(msg)
