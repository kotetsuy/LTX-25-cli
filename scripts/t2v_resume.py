#!/usr/bin/env python
"""Compatibility entry point for generation; VAE tiling is enabled by default."""
import sys
from ltx25_cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["generate", "--output", "outputs/t2v_smoke.mp4", *sys.argv[1:]]))
