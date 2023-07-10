import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(name)s] - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def logger_begin(name:str, color:bool=True, level:str=None):
    """create a logger

    Args:
        name (str):                 the name for this logger
        color (bool, optional):     whether to color the log. Defaults to True.
        level (str, optional):      the logging's level. Defaults to None.

    Returns:
        logger (logging.Logger):    the created logger
                                    [usage]:
                                        logger.debug("debug message")\n
                                        logger.info("info message")\n
                                        logger.warning("warning message")\n
                                        logger.error("error message")\n
                                        logger.critical("critical message")
    """
    # create logger with 'spam_application'
    logger:logging.Logger = logging.getLogger(name)
    logger.propagate = False

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # set the output format
    ch.setFormatter(CustomFormatter() if color else logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s'))
    
    # set the logging level
    logger.setLevel(logging.DEBUG if level is None else logging.__getattribute__(level))
    
    logger.addHandler(ch)
    
    return logger

if __name__ == '__main__':
    logger = logger_begin('debug', color=True, level='DEBUG')
    logger.debug("debug message")
    logger = logger_begin('info', color=True, level='INFO')
    logger.info("info message")
    logger = logger_begin('warning', color=True, level='WARNING')
    logger.warning("warning message")
    logger = logger_begin('error', color=True, level='ERROR')
    logger.error("error message")
    logger = logger_begin('critical', color=True, level='CRITICAL')
    logger.critical("critical message")