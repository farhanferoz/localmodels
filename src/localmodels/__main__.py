"""``python -m localmodels``: the command line, for a job that must run this checkout's interpreter."""

from __future__ import annotations

import sys

from .cli import main

sys.exit(main())
