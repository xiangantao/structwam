#!/usr/bin/env python3
"""Create the full-LIBERO 33-observed+16-future teacher manifest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vace_latent_distill.manifest import main

if __name__ == "__main__":
    main()
