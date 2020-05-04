"""
Maya interface to decompose lights
"""

import core.decomposer as core
import common.constants as constants
import baseManager as bm
import utils

import maya.cmds as cmds
import math

from common.logger import Logger

log = Logger()


class MayaManager(bm.BaseManager):

    THETA_OFFSET = -math.pi/2
    LIGHT_TYPES = {'arnold':['aiSkyDomeLight', 'aiAreaLight']}

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
            if cmds.nodeType(node) in t:
                return 1

        return 0


    def getLights(self, selection):
        '''
        Find supported lights under selections. Generally should be dome lights
        :param selection: Selection to traverse in
        '''
        for i in selection:
            for j in cmds.listRelatives(i, allDescendents=1):
                if (self.isValidLight(j)):
                    light_trans = cmds.listRelatives(j, type='transform', parent=1)[0]
                    m_light = MayaLight(light_trans)
                    self.lights['src_lights'].add(m_light)

        return self._lights['src_lights']


    def setLightProps(self, light):
        '''
        Constructs MayaLight instances properties
        :param light: Input light
        '''
        if light not in self.lights['src_lights']:
            return

        if cmds.nodeType(cmds.listRelatives(light.node, s=1)) == self.LIGHT_TYPES['arnold'][constants.SKYDOME_MODE]:
            img_node = cmds.listConnections(light.node + '.color')
            if img_node[0] and cmds.nodeType(img_node[0]) == 'file':
                light.img_path = cmds.getAttr(img_node[0] + '.fileTextureName')
                light.img_node = img_node[0]
                light.light_type = self.LIGHT_TYPES['arnold'][constants.SKYDOME_MODE]


    def makeLight(self, src_light, img_light, suffix):
        '''
        Creates Maya lights from Decomposer ImageLights
        :param src_light: MayaLight instance
        :param img_light: Decomposer ImageLight
        :param suffix: A custom suffix to prepend to the created light name
        :return: The created Maya light
        '''
        if img_light.mode == constants.SKYDOME_MODE:
            new_node = cmds.duplicate(src_light.node, name=src_light.node + suffix)[0]
        else:
            new_node = cmds.shadingNode(self.LIGHT_TYPES['arnold'][constants.AREA_MODE],
                                    name=('%sShape1' % src_light.node + suffix),
                                    asLight=1)
            scale_coeff = 1.05
            # Translate
            tx, ty, tz = utils.uvToPoint(img_light.uv[0], self._radius, self.THETA_OFFSET)
            cmds.setAttr(new_node + '.t', *[tx, ty, tz], type="float3")
            # Rotation
            mat = utils.lookAtMatrix([tx, ty, tz], [0, 0, 0], [0, 1, 0])
            rx, ry, rz = utils.matrixToRotation(mat)
            cmds.setAttr(new_node + '.r', *[rx, ry, rz], type="float3")
            # Scale
            # TO-DO: May be simpler ways to find the correct scale
            pl = utils.uvToPoint(img_light.uv[1], self._radius, self.THETA_OFFSET)
            pr = utils.uvToPoint(img_light.uv[2], self._radius, self.THETA_OFFSET)
            pt = utils.uvToPoint(img_light.uv[3], self._radius, self.THETA_OFFSET)
            pb = utils.uvToPoint(img_light.uv[4], self._radius, self.THETA_OFFSET)
            dist_x = utils.distance(pl, pr)/float(2)
            dist_y = utils.distance(pt, pb)/float(2)
            cmds.setAttr(new_node + '.sx', dist_x * scale_coeff)
            cmds.setAttr(new_node + '.sy', dist_y * scale_coeff)

            cmds.setAttr(new_node + '.normalize', 0)

        new_file = cmds.duplicate(src_light.img_node,
                                name=src_light.img_node + suffix,
                                inputConnections=1)[0]
        cmds.connectAttr(new_file+'.outColor', new_node+'.color')
        cmds.setAttr(new_file + '.fileTextureName', img_light.img_path, type='string')
        cmds.setAttr(new_file + '.colorSpace',
                    cmds.getAttr(src_light.img_node + '.colorSpace'),
                    type='string')
        new_light = MayaLight(new_node, img_light.img_path, new_file)

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

            self.decomposer = core.Decomposer(light.img_path)
            self.decomposer.preprocess()
            self.decomposer.decompose(lights_count, modes, blend)
            self.decomposer.export()

            for idx, env_light in enumerate(self.decomposer.envs):
                self.makeLight(light, env_light, constants.ENV_SUFFIX + str(idx + 1))
            for i in xrange(self.decomposer.lights_count):
                self.makeLight(light, self.decomposer.lights[i], constants.KEY_SUFFIX + str(i + 1))

        log.info("Finished extracting {} lights".format(len(self.lights['trg_lights'])))


class MayaLight(bm.BaseLight):

    pass
