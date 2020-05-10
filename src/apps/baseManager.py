"""
Base interface to be inherited by apps
"""

import core.decomposer as core
import common.constants as constants
import utils
from common.logger import Logger

from pathlib import Path

log = Logger()


class BaseManager(object):

    def __init__(self):

        self.decomposer = None
        self._lights = {'src_lights':set(), 'trg_lights':set()}
        self._radius = 1000

    @property
    def lights(self):
        return self._lights

    @lights.setter
    def lights(self, value):
        if isinstance(value, set):
            self._lights = value

    def isValidLight(self, node):
        return True

    def getLights(self, selection):
        return self._lights['src_lights']

    def setLightProps(self, light):
        pass

    def makeLight(self, src_light, img_light, suffix):
        pass

    def extractLights(self, lights_count, modes=[], radius=1000, blend=25):
        self._radius = radius
        pass

    def deleteLights(self):
        '''
        Deletes created lights
        '''
        for light in self.lights['trg_lights']:
            light.delete()
        count = len(self.lights['trg_lights'])
        self.lights['trg_lights'] = set()

        log.info("Deleted {} lights".format(count))


class BaseLight(object):

    def __init__(self, node, img_path=Path(), img_node=None, light_type=''):

        self._node = node
        self._img_path = img_path
        self._img_node = img_node
        self._light_type = light_type

    @property
    def node(self):
        return self._node

    @node.setter
    def labels(self, node):
        self._node = node

    @property
    def img_path(self):
        return str(self._img_path)

    @img_path.setter
    def img_path(self, img_path):
        self._img_path = Path(img_path)

    @property
    def img_node(self):
        return self._img_node

    @img_node.setter
    def img_node(self, img_node):
        self._img_node = img_node

    @property
    def light_type(self):
        return self._light_type

    @light_type.setter
    def light_type(self, light_type):
        self._light_type = light_type

    def delete(self):
        pass

    def __eq__(self, other):
        if isinstance(other, MayaLight):
            return (self.node == other.node and
                    self.img_path == other.img_path and
                    self.img_node == other.img_node and
                    self.light_type == other.light_type)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
