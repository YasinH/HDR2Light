import core.decomposer as core
import hou
from pathlib import Path

class HoudiniManager(object):

    LIGHT_TYPES = {'mantra':['envlight']}
    ENV_SUFFIX = '_env_'
    KEY_SUFFIX = '_key_'

    def __init__(self):

        self.decomposer = None
        self._lights = {'src_lights':set(), 'trg_lights':set()}

    @property
    def lights(self):
        return self._lights

    @lights.setter
    def lights(self, value):
        if isinstance(value, set):
            self._lights = value


    def isValidLight(self, node):

        for r, t in self.LIGHT_TYPES.iteritems():
            if node.type().name() in t:
                return 1

        return 0


    def getLights(self, selection):

        selected_nodes = [i for i in selection]
        child_nodes = [i.allSubChildren(recurse_in_locked_nodes=1) for i in selected_nodes if not self.isValidLight(i)]
        selected_nodes.extend(child_nodes)

        for i in selected_nodes:
            if (self.isValidLight(i)):
                m_light = HoudiniLight(i)
                self.lights['src_lights'].add(m_light)


    def getLightImage(self, light):

        if light not in self.lights['src_lights']:
            return

        if light.node.type().name() == self.LIGHT_TYPES['mantra'][0]:
            light.img_path = light.node.parm('env_map').eval()
            # setting file_node to light node since it's the same node (for now!)
            light.file_node = light.node
            light.light_type = self.LIGHT_TYPES['mantra'][0]


    def makeLight(self, src_light, img_path, suffix):

        new_node = src_light.node.copyTo(src_light.node.parent())
        new_node.setName(src_light.node.name() + suffix, unique_name=1)
        # setting new_file to light node since it's the same node (for now!)
        new_file = new_node
        new_file.parm('env_map').set(str(img_path))
        new_light = HoudiniLight(new_node, img_path, new_file)

        self.lights['trg_lights'].add(new_light)
        return new_light


    def extractLights(self, lights_limit):

        for light in self.lights['src_lights']:
            self.getLightImage(light)

            self.decomposer = core.Decomposer(hou.expandString(light.img_path))
            self.decomposer.preprocess()
            self.decomposer.decompose(lights_limit)
            self.decomposer.export()

            for idx, env_path in enumerate(self.decomposer._envs_paths):
                self.makeLight(light, env_path, self.ENV_SUFFIX + str(idx + 1))
            for idx, light_path in enumerate(self.decomposer._lights_paths):
                self.makeLight(light, light_path, self.KEY_SUFFIX + str(idx + 1))


class HoudiniLight(object):

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
        if isinstance(other, HoudiniLight):
            return (self.node == other.node and
                    self.img_path == other.img_path and
                    self.img_node == other.img_node)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
