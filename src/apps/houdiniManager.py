#
#  Copyright (c) 2020, Yasin Hasanian
#  See license.txt
#

"""
Houdini interface for light decomposer
"""

import core.decomposer as core
import common.constants as constants
import baseManager as bm
import utils

import hou
import math

from common.logger import Logger

log = Logger()


class HoudiniManager(bm.BaseManager):

    THETA_OFFSET = +math.pi/2
    LIGHT_TYPES = {'mantra':['envlight', 'hlight']}

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
        '''
        Determines if given node is a supported light type
        :param node: Input node
        :return: 1 if light type is support, otherwise 0
        '''
        for r, t in self.LIGHT_TYPES.iteritems():
            if node.type().name() in t:
                return 1

        return 0


    def getLights(self, selection):
        '''
        Find supported lights under selections. Generally should be dome lights
        :param selection: Selection to traverse in
        '''
        selected_nodes = [i for i in selection]
        child_nodes = [i.allSubChildren(recurse_in_locked_nodes=1) for i in selected_nodes if not self.isValidLight(i)]
        selected_nodes.extend(child_nodes)

        for i in selected_nodes:
            if (self.isValidLight(i)):
                m_light = HoudiniLight(i)
                self.lights['src_lights'].add(m_light)


    def setLightProps(self, light):
        '''
        Constructs HoudiniLight instances properties
        :param light: Input light
        '''
        if light not in self.lights['src_lights']:
            return

        if light.node.type().name() == self.LIGHT_TYPES['mantra'][constants.SKYDOME_MODE]:
            light.img_path = light.node.parm('env_map').eval()
            # setting file_node to light node since it's the same node (for now!)
            light.file_node = light.node
            light.light_type = self.LIGHT_TYPES['mantra'][constants.SKYDOME_MODE]


    def makeLight(self, src_light, img_light, suffix):
        '''
        Creates Houdini lights from Decomposer ImageLights.
        :param src_light: HoudiniLight instance
        :param img_light: Decomposer ImageLight
        :param suffix: A custom suffix to prepend to the created light name
        :return: The created Houdini light
        '''
        # TO-DO: Add full support for input textures (implicit or explicit)
        if img_light.mode == constants.SKYDOME_MODE:
            new_node = src_light.node.copyTo(src_light.node.parent())
            new_node.parm('env_map').set(str(img_light.img_path))
        else:
            new_node = src_light.node.parent().createNode(self.LIGHT_TYPES['mantra'][constants.AREA_MODE])
            new_node.parm('light_type').set(2)

            scale_coeff = 1.05
            # Translate
            tx, ty, tz = utils.uvToPoint(img_light.uv[0], self._radius, self.THETA_OFFSET)
            t = hou.hmath.buildTranslate((tx, ty, tz))
            # Rotation
            mat = utils.lookAtMatrix([tx, ty, tz], [0, 0, 0], [0, 1, 0])
            rx, ry, rz = utils.matrixToRotation(mat)
            src_rot = hou.hmath.buildRotate(src_light.node.worldTransform().extractRotates())
            r = hou.hmath.buildRotate((rx, ry, rz))
            new_node.setWorldTransform(r * t * src_rot)
            # Scale
            # TO-DO: May be simpler ways to find the correct scale
            pl = utils.uvToPoint(img_light.uv[1], self._radius, self.THETA_OFFSET)
            pr = utils.uvToPoint(img_light.uv[2], self._radius, self.THETA_OFFSET)
            pt = utils.uvToPoint(img_light.uv[3], self._radius, self.THETA_OFFSET)
            pb = utils.uvToPoint(img_light.uv[4], self._radius, self.THETA_OFFSET)
            dist_x = utils.distance(pl, pr)/float(2)
            dist_y = utils.distance(pt, pb)/float(2)
            new_node.parm('areasize1').set(dist_x * scale_coeff)
            new_node.parm('areasize2').set(dist_y * scale_coeff)

            new_node.parm('normalizearea').set(0)
            new_node.parm('light_texture').set(str(img_light.img_path))

        new_node.setName(src_light.node.name() + suffix, unique_name=1)
        # setting new_file to light node since it's the same node (for now!)
        new_file = new_node
        new_light = HoudiniLight(new_node, img_light.img_path, new_file)

        self.lights['trg_lights'].add(new_light)
        return new_light


    def extractLights(self, lights_count, modes=[], radius=1000, blend=25):
        '''
        Main interface to decompose lights
        :param lights_count: Number of lights to extract.
                            This is limited to maximum number of lights decomposed by the decomposer
        :param modes: A list or int of 0 or 1 for each extracted lights to set their type.
                    0 for skydome mode, 1 for area mode
        :param radius: If the extracted light is an area,
                    radius indicates how far the light should be from the origin
        :param blend: The amount of edge blur from key lights to environment
        '''
        self._radius = radius

        for light in self.lights['src_lights']:
            self.setLightProps(light)

            self.decomposer = core.Decomposer(hou.expandString(light.img_path))
            self.decomposer.preprocess()
            self.decomposer.decompose(lights_count, modes, blend)
            self.decomposer.export()

            for idx, env_light in enumerate(self.decomposer.envs):
                self.makeLight(light, env_light, constants.ENV_SUFFIX + str(idx + 1))
            for i in xrange(self.decomposer.lights_count):
                self.makeLight(light, self.decomposer.lights[i], constants.KEY_SUFFIX + str(i + 1))

        log.info("Finished extracting {} lights".format(len(self.lights['trg_lights'])))


    @staticmethod
    def getSelection():
        return hou.selectedNodes()


class HoudiniLight(bm.BaseLight):

    def delete(self):
        hou.Node.destroy(self._node)
