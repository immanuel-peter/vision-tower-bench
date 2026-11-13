#!/usr/bin/env python3
"""Pair two runs' correspondence or perturbation results and bootstrap the difference.

correspondence_bootstrap.py compares Stages inside one run. This script pairs
two runs by pair or image identity and keeps the paired measurements.
"""

from vtb.cross_run_bootstrap import main

if __name__ == "__main__":
    main()
