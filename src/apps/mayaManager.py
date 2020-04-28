import core.decomposer as core
import utils
import common.constants as constants

import maya.cmds as cmds
from pathlib import Path
import math

class MayaManager(object):

    LIGHT_TYPES = {'arnold':['aiSkyDomeLight', 'aiAreaLight']}
    ENV_SUFFIX = '_env_'
    KEY_SUFFIX = '_key_'

    def __init__(self):

        self.decomposer = None
        self._lights = {'src_lights':set(), 'trg_lights':set()}
        self._radius = 10

    @property
    def lights(self):
        return self._lights

    @lights.setter
    def lights(self, value):
        if isinstance(value, set):
            self._lights = value


    def isValidLight(self, node):

        for r, t in self.LIGHT_TYPES.iteritems():
            if cmds.nodeType(node) in t:
                return 1

        return 0


    def getLights(self, selection):

        for i in selection:
            for j in cmds.listRelatives(i, allDescendents=1):
                if (self.isValidLight(j)):
                    light_trans = cmds.listRelatives(j, type='transform', parent=1)[0]
                    m_light = MayaLight(light_trans)
                    self.lights['src_lights'].add(m_light)


    def getLightImage(self, light):

        if light not in self.lights['src_lights']:
            return

        if cmds.nodeType(cmds.listRelatives(light.node, s=1)) == self.LIGHT_TYPES['arnold'][constants.SKYDOME_MODE]:
            file_node = cmds.listConnections(light.node + '.color')
            if file_node[0] and cmds.nodeType(file_node[0]) == 'file':
                light.img_path = cmds.getAttr(file_node[0] + '.fileTextureName')
                light.file_node = file_node[0]
                light.light_type = self.LIGHT_TYPES['arnold'][constants.SKYDOME_MODE]


    def makeLight(self, src_light, img_light, suffix):

        if img_light.mode == constants.SKYDOME_MODE:
            new_node = cmds.duplicate(src_light.node, name=src_light.node + suffix)[0]
        else:
            new_node = cmds.shadingNode('aiAreaLight',
                                    name=('%sShape1' % src_light.node + suffix),
                                    asLight=1)
            # Translate
            tx, ty, tz = utils.uvToPoint(img_light.uv, self._radius, -math.pi/10)
            cmds.setAttr(new_node + '.t', *[tx, ty, tz], type="float3")
            # Rotation
            mat = utils.lookAtMatrix([tx, ty, tz], [0, 0, 0], [0, 1, 0])
            rx, ry, rz = utils.matrixToRotation(mat)
            cmds.setAttr(new_node + '.r', *[rx, ry, rz], type="float3")
            # Scale
            cmds.setAttr(new_node + '.sx', img_light.ratio * self._radius)
            cmds.setAttr(new_node + '.sy', self._radius)

        new_file = cmds.duplicate(src_light.file_node,
                                name=src_light.file_node + suffix,
                                inputConnections=1)[0]
        cmds.connectAttr(new_file+'.outColor', new_node+'.color')
        cmds.setAttr(new_file + '.fileTextureName', img_light.img_path, type='string')
        cmds.setAttr(new_file + '.colorSpace',
                    cmds.getAttr(src_light.file_node + '.colorSpace'),
                    type='string')
        new_light = MayaLight(new_node, img_light.img_path, new_file)

        self.lights['trg_lights'].add(new_light)
        return new_light


    def extractLights(self, lights_limit, modes=[], radius=10):

        self._radius = radius

        for light in self.lights['src_lights']:
            self.getLightImage(light)

            self.decomposer = core.Decomposer(light.img_path)
            self.decomposer.preprocess()
            self.decomposer.decompose(lights_limit, modes)
            self.decomposer.export()

            for idx, env_light in enumerate(self.decomposer.envs):
                self.makeLight(light, env_light, self.ENV_SUFFIX + str(idx + 1))

            for i in xrange(lights_limit):
                self.makeLight(light, self.decomposer.lights[i], self.KEY_SUFFIX + str(idx + 1))


class MayaLight(object):

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


    def __eq__(self, other):
        if isinstance(other, MayaLight):
            return (self.node == other.node and
                    self.img_path == other.img_path and
                    self.img_node == other.img_node)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
