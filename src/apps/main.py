#!/bin/env python

from common.logger import Logger
log = Logger()

try:
    from apps.mayaManager import MayaManager
    app = 'maya'
except ImportError:
    pass
try:
    from apps.houdiniManager import HoudiniManager
    app = 'houdini'
except ImportError:
    pass

if not app:
    msg = "Failed to load manager! Make sure the tool and its dependencies are in PYTHONPATH"
    log.exception(msg)
    raise ImportError(msg)


def run(lights_count, modes=[], radius=1000, blend=25):
    if app == 'maya':
        manager = MayaManager()
        manager.getLights(MayaManager.getSelection())
    else:
        manager = HoudiniManager()
        manager.getLights(HoudiniManager.getSelection())
    manager.extractLights(lights_count, modes, radius, blend)
    
    return manager
