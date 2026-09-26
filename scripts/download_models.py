#!/usr/bin/env python
"""Selectively download the LTX-2.5 diffusers checkpoint.

The repo is 153 GiB in full, but a distilled-inference setup needs ~67 GiB of it.
Two traps this script exists to avoid:

  * ``transformer/`` ships the *same* weights sharded two ways (a 4-shard and an
    8-shard set, 35.4 GiB each). Only the one named by the shard index is real;
    a naive ``transformer/*`` glob downloads both.
  * ``transformer_full/`` is the non-distilled model (another 35.4 GiB) and is
    never needed for distilled inference.

Same story for ``connectors/``: a single file and a 2-shard set of equal size.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError

REPO = "Lightricks/LTX-2.5-Diffusers"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "models" / "ltx-2.5-diffusers"

# Everything small enough that picking through it is not worth the complexity.
BASE_PATTERNS = [
    "*.json",
    "tokenizer/*",
    "scheduler/*",
    "vae/*",
    "audio_vae/*",
    "vocoder/*",
    "duration_head/*",
    "diffusion_decoder/*",
    "latent_upsampler/*",
    "text_encoder/*",
]

# Directories whose shard layout is ambiguous: resolve via their index file.
SHARDED_DIRS = ["transformer", "connectors"]


def resolve_shards(repo: str, subdir: str) -> list[str]:
    """Return the allow-patterns for `subdir`, following its shard index.

    Falls back to the unsharded single file when no index is published.
    """
    index_name = f"{subdir}/diffusion_pytorch_model.safetensors.index.json"
    try:
        index_path = hf_hub_download(repo, index_name)
    except EntryNotFoundError:
        # Genuinely unsharded. Anything else (gating, network) must propagate:
        # falling back here would silently produce a wrong file list.
        return [f"{subdir}/config.json", f"{subdir}/diffusion_pytorch_model.safetensors"]

    index = json.loads(Path(index_path).read_text())
    shards = sorted(set(index["weight_map"].values()))
    print(f"  {subdir}: index names {len(shards)} shard(s)")
    return [f"{subdir}/config.json", index_name, *(f"{subdir}/{s}" for s in shards)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"download target (default: {DEFAULT_DEST})")
    ap.add_argument("--full-transformer", action="store_true", help="also fetch transformer_full/ (+35 GiB, non-distilled)")
    ap.add_argument("--lora", action="store_true", help="also fetch the 22b distilled LoRA (+9 GiB)")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved file list and exit")
    args = ap.parse_args()

    patterns = list(BASE_PATTERNS)
    try:
        for subdir in SHARDED_DIRS:
            patterns += resolve_shards(REPO, subdir)
    except GatedRepoError:
        print(
            f"error: {REPO} is gated and this account is not approved yet.\n"
            f"       Visit https://huggingface.co/{REPO} and accept the licence,\n"
            f"       then re-run. (`hf auth whoami` must show the approved account.)",
            file=sys.stderr,
        )
        return 1

    if args.full_transformer:
        patterns.append("transformer_full/*")
    if args.lora:
        patterns.append("ltx-2.5-22b-distilled-lora-450-bf16.safetensors")

    # `*.json` is matched with fnmatch, whose `*` crosses `/` — so it also picks up
    # transformer_full's config and shard index, leaving a stub directory behind.
    ignore = [] if args.full_transformer else ["transformer_full/*"]

    print(f"\nallow_patterns ({len(patterns)}):")
    for p in patterns:
        print("  ", p)
    if ignore:
        print(f"ignore_patterns: {ignore}")
    if args.dry_run:
        return 0

    free_gib = shutil.disk_usage(args.dest.parent.parent).free / 1024**3
    print(f"\nfree space at {args.dest}: {free_gib:.0f} GiB (need ~67 GiB for the default set)")

    snapshot_download(
        REPO,
        local_dir=args.dest,
        allow_patterns=patterns,
        ignore_patterns=ignore,
        max_workers=8,
    )
    print(f"\ndone -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
