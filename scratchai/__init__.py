from scratchai import learners
from scratchai import DataLoader
from scratchai import utils
from scratchai import nets
from scratchai import init
from scratchai import attacks
from scratchai import pretrained
from scratchai import imgutils
from scratchai import one_call
from scratchai._config import *

import os

# TODO Confirm if this is the most efficient way to do so.
if not os.path.exists(home):
  os.makedirs(home)
