from __future__ import division
from datetime import datetime

__author__ = 'hmourit'


def _strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


class Timer():

    def __init__(self):
        self.start = datetime.now()

    def elapsed(self):
        # return datetime.now() - self.start
        return _strfdelta(datetime.now() - self.start, '{hours:02d}:{minutes:02d}:{seconds:02d}')