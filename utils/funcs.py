import logging 
import os
from pathlib import Path

""" INIT LOGGER
the LCN_model method 'train' should output a log every 50 training iterations.
I am setting it up here once in the main parent .py file
logger = logging.getLogger('name_of_log') can be used to get a pre-existing logger within any object/class method
then call logger.info(your message lala) to send some info to that log """
    
def initLogger(logger_name,output_file_name):
    logger = logging.getLogger(logger_name)
    if len(logger.handlers)>0:
        logger.handlers.clear() # avoids duplicate log messages caused by adding handlers
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(output_file_name) # logs will be saved to this file
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(f_format)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    return logger

def path_exists(pathname: str) -> bool:
    p = Path(pathname)
    return p.is_dir()
    
 