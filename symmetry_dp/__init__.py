"""SymmetryDP package.
"""
# see https://github.com/ContinuumIO/anaconda-issues/issues/905
import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from .linear_policy_LQ import *