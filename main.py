"""FC-Comparator entry point.

    python main.py                      # start the station UI (login: nice / nice1234)
    python main.py inspect --image board.jpg --part P001
    python main.py --help               # all commands (add-part, train, evaluate, report, export, check ...)
"""

import sys

from fc_comparator.cli import main

if __name__ == "__main__":
    sys.exit(main())
