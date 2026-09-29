"""FC-Comparator entry point.

    python main.py                      # start the station UI
    python main.py inspect --image samples/boards/board_P001_mixed.jpg --part P001
    python main.py --help               # all commands (report, export, harvest, set-secret, check)
"""

import sys

from fc_comparator.cli import main

if __name__ == "__main__":
    sys.exit(main())
