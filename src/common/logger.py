import logging

class Logger(object):

    def __init__(self, name='[HDR2Light]', level=logging.DEBUG):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        # fh = logging.FileHandler('%s.log' % name, 'w')
        # self.logger.addHandler(fh)
        # if not getattr(self.logger, 'handler_set', None):
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


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
# # create a file handler
# handler = logging.FileHandler('hello.log')
# handler.setLevel(logging.INFO)
#
# # create a logging format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
#
# # add the file handler to the logger
# logger.addHandler(handler)
