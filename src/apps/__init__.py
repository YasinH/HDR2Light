#
#  Copyright (c) 2020, Yasin Hasanian
#  See license.txt
#

import main
import baseManager
import utils
reload(main)
reload(baseManager)
reload(utils)

try:
    import mayaManager
    reload(mayaManager)
except ImportError as e:
    # Not running Maya, pass
    if 'maya.cmds' in e.message:
        pass
    else:
        raise

try:
    import houdiniManager
    reload(houdiniManager)
except ImportError as e:
    # Not running Houdini, pass
    if 'hou' in e.message:
        pass
    else:
        raise
