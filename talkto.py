#!/usr/bin/env python3
# talkto.py — talk to the haze
#
# A simple bridge to the interactive REPL.
# Because sometimes you need to have a conversation.

import sys
from pathlib import Path

# Add haze directory to path
sys.path.insert(0, str(Path(__file__).parent / "haze"))

# Import and run
from haze import run

if __name__ == "__main__":
    run.main()
