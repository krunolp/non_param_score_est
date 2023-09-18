"""non_param_score_est is a package for non-parametric estimation of the score function.

More information is available at https://github.com/krunolp/non_param_score_est
"""

import sys

#: The release version
version = '0.0.1'
__version__ = version

MIN_PYTHON_VERSION = 3, 8
MIN_PYTHON_VERSION_STR = '.'.join([str(v) for v in MIN_PYTHON_VERSION])

if sys.version_info < MIN_PYTHON_VERSION:
    raise Exception(f"non_param_score_est {version} requires Python {MIN_PYTHON_VERSION_STR} or newer.")
