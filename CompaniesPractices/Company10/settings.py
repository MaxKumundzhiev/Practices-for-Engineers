# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import logging.config

LOGGING = dict(
        version=1,
        disable_existing_loggers=False,
        formatters={
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        handlers={
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'DEBUG',
                'stream': 'ext://sys.stdout'
            },
            'error': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'WARNING',
                'stream': 'ext://sys.stderr'
            }
        },
        loggers={
            '': {
                'handlers': ['console', 'error'],
                'level': 'INFO'
            }
        })

logging.config.dictConfig(LOGGING)
LOGGER = logging.getLogger('KHEIRON_TEST-TASK')







