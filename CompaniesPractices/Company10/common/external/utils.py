# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import base64
import random
from datetime import datetime, timedelta

SALT = '_KHEIRON_TEST_TASK'


def salty_encode(string) -> str:
    text_to_encode = str(string) + SALT
    message_bytes = text_to_encode.encode('ascii')
    base64_bytes = base64.b64encode(message_bytes)
    base64_message = base64_bytes.decode('ascii')
    return base64_message


def salty_decode(string: str) -> str:
    base64_bytes = string.encode('ascii')
    message_bytes = base64.b64decode(base64_bytes)
    message = message_bytes.decode('ascii')
    return message


def jitter_date(date: str):
    n_days = random.randint(1, 10)
    date = datetime.strptime(date, '%Y%m%d')
    shifted_date = date + timedelta(days=n_days)
    return f'{shifted_date.year}.{shifted_date.month}.{shifted_date.day}'
