from .data_loader import *
from .train_model import *
from .test_model import *
from .grad_cam import *
from .plotting import *
from .lr_finder import LRFinder

# Future print function
from __future__ import print_function

# Installing latest Albumentation library
!pip install -U git+https://github.com/albu/albumentations -q --quiet
#pip install apex -q

import matplotlib.pyplot as plt

# For inline matplotlib plotting
%matplotlib inline