"""Compatibility wrapper; optional arguments override the original fixed paths."""
import sys
from ltx25_cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["decode", "--input", "outputs/fox_5s.latents.pt",
                          "--output", "outputs/fox_5s.mp4", *sys.argv[1:]]))
