from __future__ import division
from datetime import datetime

__author__ = 'hmourit'


class Timer():

    def __init__(self):
        self.start = datetime.now()

    def elapsed(self):
        return datetime.now() - self.start