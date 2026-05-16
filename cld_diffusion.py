"""Compatibility wrapper for running the toy CLD script from the repo root."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from hamiltonian_guidance.cld_diffusion import main


if __name__ == "__main__":
    main()
