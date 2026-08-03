"""Pytest configuration: make the package importable as ``python -m evals.*``.

The eval CLIs live under ``cartesia-pytorch/`` and are designed to be run as
``python -m evals.<name>`` with that directory on ``sys.path``. Put it there
for the test suite so ``evals`` and ``cartesia_pytorch`` resolve regardless of
where pytest is invoked from.
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
