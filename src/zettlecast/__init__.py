"""Zettlecast - Digital Zettelkasten Middleware"""

import sys

if sys.platform == "win32":
    # On Windows, torch must load its DLLs before lancedb/pyarrow, otherwise
    # torch's c10.dll fails to initialise (WinError 1114).
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

__version__ = "0.1.0"
