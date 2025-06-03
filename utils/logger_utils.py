import logging
import os
import time

def get_logger(log_dir=None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        time_str=time.strftime('%Y%m%d%H%M%S',time.localtime())
        log_file= f'{time_str}.log'
        file_handler=logging.FileHandler(os.path.join(log_dir,log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

if __name__ == '__main__':
    logger=get_logger('./log')
    logger.info('this is a test msg')