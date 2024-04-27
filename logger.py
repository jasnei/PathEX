import logging
import logging.handlers
import os
import time


class Logger:
    
    def __init__(self, path: str, c_verbose: int=0, f_verbose: int=0, 
                formatter: str=None, default_formatter: bool=True,
                filehandler: int=0, when=''):
        """
        Description:
            - create logger using logging, you could disable existing logging
              by calling Logger.disable_existing_loggers before app to suppress
              other packages logger
        
        Arguments:
            - path: (str) log path
            - c_verbose: (int) console verbosity, options = [0, 1, 2], 
                            0: logging.DEBUG, which will print out all the log to console
                            1: logging.INFO
                            2: logging.WARN
                            3: logging.ERROR
                            4: logging.CRITIAL
                            5: logging.FATAL
            - f_verbose: (int) log file verbosity, options = [0, 1, 2], 
                            0: logging.DEBUG, which will save out all the log to log file
                            1: logging.INFO
                            2: logging.WARN
                            3: logging.ERROR
                            4: logging.CRITIAL
                            5: logging.FATAL
            - formatter: (str) logging formatter
            - default: (boolean) default is True, then will use built-int formatter, if False and formatter is None,
                                there is not formatter (None)
        """
        assert c_verbose in (0, 1, 2, 3, 4, 5), f'c_verbose must be integer number in (0, 1, 2, 3, 4, 5), but got {c_verbose}'
        assert f_verbose in (0, 1, 2, 3, 4, 5), f'f_verbose must be integer number in (0, 1, 2, 3, 4, 5), but got {f_verbose}'
        self.level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN, 
                      3: logging.ERROR, 4: logging.CRITICAL, 5: logging.FATAL}
        c_level = self.level_dict[c_verbose]
        f_level = self.level_dict[f_verbose]

        # Formatter
        self.builtin_formatter = '%(asctime)s - %(levelname)s - %(message)s'
        if formatter is None and default_formatter:
            formatter = self.builtin_formatter

        self.logger = logging.getLogger(path)
        self.logger.setLevel(level=logging.DEBUG)

        # set this false to disable propagate
        self.logger.propagate = False
        
        # Console
        self.console = logging.StreamHandler()
        self.console.setLevel(c_level)
        
        # Log file
        if filehandler==0:
            # 50*1024**2 = 50MB
            self.fh_handler = logging.handlers.RotatingFileHandler(path, maxBytes=50*1024**2, backupCount=1000)
        elif filehandler==1:
            # when：日志文件按什么维度切分。'S'-秒；'M'-分钟；'H'-小时；'D'-天；'W'-周
            #       这里需要注意，如果选择 D-天，那么这个不是严格意义上的'天'，而是从你
            #       项目启动开始，过了24小时，才会从新创建一个新的日志文件，
            #       如果项目重启，这个时间就会重置。所以这里选择'MIDNIGHT'-是指过了午夜
            #       12点，就会创建新的日志。
            # interval：是指等待多少个单位 when 的时间后，Logger会自动重建文件。
            # backupCount：是保留日志个数。默认的0是不会自动删除掉日志。
            if when is not None:
                when = when
            else: when = 'D'
            self.fh_handler = logging.handlers.TimedRotatingFileHandler(path, when=when)
        elif filehandler==2:
            self.fh_handler = logging.handlers.BaseRotatingHandler(path, mode='a')
        self.fh_handler.setLevel(f_level)
        
        self.logger.addHandler(self.console)
        self.logger.addHandler(self.fh_handler)

        # set formatter
        self.set_formatter(formatter)
    
    def set_level(self, c_verbose=1, f_verbose=0):
        c_level = self.level_dict[c_verbose]
        f_level = self.level_dict[f_verbose]
        # Console
        self.console.setLevel(c_level)
        # Log file
        self.fh_handler.setLevel(f_level)

    def set_formatter(self, formatter):
        formatter = logging.Formatter(formatter)
        self.console.setFormatter(formatter)
        self.fh_handler.setFormatter(formatter)
        
    def debug(self, message):
        self.logger.debug(message)
        
    def info(self, message):
        self.logger.info(message)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def error(self, message):
        self.logger.error(message)
        
    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)

    def reset_default_formatter(self,):
        self.set_formatter(self.builtin_formatter)

    @staticmethod
    def disable_existing_loggers():
        """Call before you create any instance to disable existing logger from other packages"""
        import logging.config
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': True,
            })


def get_localtime():
    return time.strftime('%Y-%m-%d %H:%M:%S')
    

#=================================== Log config ===================================
save_path = "./data/log"
if not os.path.exists(save_path):
    os.makedirs(save_path)
logger = Logger(os.path.join(save_path, 'process.log'), c_verbose=1, f_verbose=0, default_formatter=True, filehandler=1, when='W0')
# time_message = get_localtime()
# logger.info(time_message.center(120, '*'))



if __name__ == '__main__':

    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    # DATE_FORMAT = '%Y/%m/%d %H:%M:%S %p'
    # logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler('trace.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info('Start print log')
    logger.debug('Do something')
    logger.warning('Something maybe fail')
    try:
        open('sklearn.txt', 'rb')
    except (SystemExit, KeyboardInterrupt):
        raise 
    except Exception:
        # logger.error('Fail to open sklearn.txt from logger.erro', exc_info=True)
        logger.exception('Fail to open sklearn.txt from logger.exception')

    logger.info('Finish')
