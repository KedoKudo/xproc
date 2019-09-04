#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event and exception handeling with logger
"""

import functools
import logging


def create_logger(logfile=r"/tmp/tomoproc.log"):
    """Default logger for exception tracking"""
    logger = logging.getLogger("tomoproc_logger")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler(logfile)
    fh.setFormatter(
         logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
         )
    )
 
    # add handler to logger object
    logger.addHandler(fh)
    return logger

logger_default = create_logger()


def log_exception(logger):
    """decorator for logging exception"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                args_str = ",".join(map(str, args))
                kwargs_str = ",".join([f"{k}={v}" for k,v in kwargs.items()])
                logger.exception(
                    f'Exception in calling {func.__name__}()\n\targs: {args_str}\n\tkwargs:{kwargs_str}'
                )
                # re-raise the exception
                raise
        return wrapper
    return decorator


def log_event(logger):
    """decorator for verbose event logging"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_str = ",".join(map(str, args))
            kwargs_str = ",".join([f"{k}={v}" for k,v in kwargs.items()])
            logger.info(f'Executing {func.__name__}()\n\targs: {args_str}\n\tkwargs:{kwargs_str}')
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log_exception(logger_default)
# @log_event(logger_default)
def _test_logger(a, b=1, c=1):
    """testing the logger for exception"""
    return a/b


if __name__ == "__main__":
    print(_test_logger.__name__)
    print(_test_logger.__doc__)
    print(f"no exception test:\n\t{_test_logger(1, b=1)}")
    print(f"exception test:\n\t{_test_logger(1, b=0)}")
