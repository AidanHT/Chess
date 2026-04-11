#!/usr/bin/env python3
"""
Play chess against your AlphaZero engine in the browser.

Usage:
    python play.py                    # defaults (port 8000, 800 sims)
    python play.py --num-sims 1600    # stronger engine
    python play.py --no-engine        # UI only (no model)
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    server = Path(__file__).resolve().parent / "gui" / "server.py"
    sys.exit(subprocess.call([sys.executable, str(server)] + sys.argv[1:]))
