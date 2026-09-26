#!/usr/bin/env python
"""Phase 2 smoke test: one short text-to-video generation, with timing and memory.

Distilled inference is driven by an explicit sigma schedule (8 steps), not by
num_inference_steps, and runs unguided. Passing num_inference_steps instead would
hand the model a generic linear schedule and quietly cost quality.

Runs in two phases, because the checkpoint does not fit anywhere whole: 67 GiB of
bf16 weights against 45 GiB of system RAM and 48 GiB of VRAM.

  encode    tokenizer + text_encoder only (23 GiB) -> prompt embeddings on disk
  generate  everything except text_encoder (43 GiB) -> mp4

Neither phase needs the other's weights, and 43 GiB fits in VRAM outright, so
offload is no longer required. Within generate, the transformer is dropped once
denoising is done, which is what leaves the VAE room to decode. Holding all of
it at once is what does not fit: the
earlier single-phase run under enable_model_cpu_offload() peaked at 43.3 GiB RSS
plus 6.3 GiB swap, flooded the kernel log with `amdgpu: SVM mapping failed,
exceeds resident system memory limit`, and took the terminal down with it.

`both` (the default) runs encode as a subprocess, so its 23 GiB goes back to the
OS on process exit rather than on the allocator's good behaviour, and a failure
in generate does not cost a re-encode.

Long runs should still be detached from the terminal -- the crash above killed
gnome-terminal-server, and every child went with it:

    nohup .venv/bin/ltx25 generate > run.log 2>&1 &
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Both are read while torch initialises, so they have to be set before the import.
# Override either from the environment if a run misbehaves.
#
# Without AOTRITON_ENABLE_EXPERIMENTAL, ROCm reports no memory-efficient SDPA kernel and
# torch falls back to the math backend, which materialises the full [B, heads, S, S]
# score matrix -- 4.5 GiB for one attention call at 768x512x121 (6144 tokens, 32 heads).
# That does not fit in what is left after the transformer, and torch prints the hint
# itself: "Mem Efficient attention on Current AMD GPU is still experimental."
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
# The weights land in 43 GiB of a 48 GiB card, so the ~1.5 GiB the caching allocator
# strands between segments is the difference between fitting and not.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402,F401  (import before anything that pulls in a second ROCm runtime)

DEFAULT_MODEL = Path.cwd() / "models" / "ltx-2.5-diffusers"

# Components `from_pretrained` skips entirely when passed as None: pipeline_utils drops
# None-valued components from init_dict and re-inserts them as None, and
# LTX2Pipeline.__init__ guards every attribute it derives from one. `tokenizer` and
# `scheduler` are always loaded -- together they are under 32 MiB. `diffusion_decoder`
# is listed in model_index.json but absent from __init__, so diffusers ignores it.
_DROPPABLE = ("text_encoder", "transformer", "connectors", "vae", "audio_vae", "vocoder", "duration_head")

_ENCODE_ONLY = {"text_encoder"}
_GENERATE_ONLY = {"transformer", "connectors", "vae", "audio_vae", "vocoder", "duration_head"}


class Stage:
    """Times a block and records peak host/device memory."""

    log: list[dict] = []

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        import psutil

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.proc = psutil.Process()
        self.t = time.perf_counter()
        print(f"  [{self.name}] ...", flush=True)
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        entry = {
            "stage": self.name,
            "seconds": round(time.perf_counter() - self.t, 1),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "rss_gib": round(self.proc.memory_info().rss / 1024**3, 2),
        }
        Stage.log.append(entry)
        print(
            f"  [{self.name}] {entry['seconds']}s  "
            f"peak VRAM {entry['peak_vram_gib']} GiB  RSS {entry['rss_gib']} GiB",
            flush=True,
        )
        return False


def load_pipeline(model: Path, keep: set[str], *, image_to_video=False):
    """Build an LTX2Pipeline holding only `keep`; the rest is never read off disk."""
    from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline

    pipeline_class = LTX2ImageToVideoPipeline if image_to_video else LTX2Pipeline
    return pipeline_class.from_pretrained(
        model, dtype=torch.bfloat16, **{name: None for name in _DROPPABLE if name not in keep}
    )


def phase_encode(args) -> None:
    with Stage("encode:load"):
        pipe = load_pipeline(args.model, keep=_ENCODE_ONLY)
        pipe.text_encoder.to("cuda")

    # no_grad is load-bearing: only __call__ and enhance_prompt carry @torch.no_grad(),
    # so encode_prompt called on its own builds an autograd graph across all 48 Gemma
    # layers -- ~24 GiB of saved activations on top of the 22 GiB of weights, which OOMs
    # a 48 GiB card on a phase that should need barely half of it.
    with Stage("encode:embed"), torch.no_grad():
        # do_classifier_free_guidance=True unconditionally: the negative pass costs one
        # extra forward and keeps the cache valid if a later run turns guidance on. The
        # distilled schedule runs unguided, so generate normally leaves them unused.
        embeds, mask, neg_embeds, neg_mask = pipe.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=args.max_sequence_length,
            device=torch.device("cuda"),
        )

    args.embeds.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "max_sequence_length": args.max_sequence_length,
            # _get_gemma_prompt_embeds forces left padding before encoding; generate has
            # to hand the connectors the same value or the embeddings are misread.
            "padding_side": "left",
            "prompt_embeds": embeds.cpu(),
            "prompt_attention_mask": mask.cpu(),
            "negative_prompt_embeds": neg_embeds.cpu(),
            "negative_prompt_attention_mask": neg_mask.cpu(),
            "stages": Stage.log,
        },
        args.embeds,
    )
    mib = args.embeds.stat().st_size / 1024**2
    print(f"  embeds -> {args.embeds}  {tuple(embeds.shape)} ({mib:.0f} MiB)")


def load_embeds(path: Path, args) -> dict | None:
    """Return the cached embeddings, or None if they are absent or for another prompt."""
    if not path.exists():
        return None
    cache = torch.load(path, map_location="cpu", weights_only=True)
    stale = (
        cache["prompt"] != args.prompt
        or cache["negative_prompt"] != args.negative_prompt
        or cache["max_sequence_length"] != args.max_sequence_length
    )
    return None if stale else cache


def phase_generate(args) -> int:
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    from diffusers.utils import encode_video

    image_path = getattr(args, "image", None)
    image_kwargs = {}
    if image_path is not None:
        from .images import load_start_image
        image_kwargs["image"] = load_start_image(image_path, args.width, args.height, args.image_fit)
    input_metadata = {
        "image": str(image_path) if image_path is not None else None,
        "image_fit": args.image_fit if image_path is not None else None,
    }

    cache = load_embeds(args.embeds, args)
    if cache is None:
        raise SystemExit(
            f"no embeddings for this prompt at {args.embeds}. "
            f"Run --phase encode first, or use --phase both."
        )
    Stage.log.extend(cache.get("stages", []))

    with Stage("generate:load"):
        pipe = load_pipeline(args.model, keep=_GENERATE_ONLY, image_to_video=image_path is not None)
        # The tokenizer is loaded but unused here; __call__ reads padding_side off it to
        # drive the connectors, so pin it to whatever encode actually used.
        pipe.tokenizer.padding_side = cache["padding_side"]

    with Stage(f"generate:offload:{args.offload}"):
        if args.offload == "model":
            pipe.enable_model_cpu_offload()
        elif args.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")

    if args.vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.use_framewise_decoding = True

    device = torch.device("cuda")
    with Stage("generate:denoise"):
        # prompt stays None: check_inputs rejects prompt and prompt_embeds together.
        # output_type="latent" stops before the VAE: decoding in the same breath would run
        # with the transformer still resident, and denoise alone already peaks at 45.4 GiB
        # of the 48 -- an untiled conv3d over 121x512x768 then asks for another 9.65 GiB.
        latents, audio_latents = pipe(
            prompt=None,
            prompt_embeds=cache["prompt_embeds"].to(device),
            prompt_attention_mask=cache["prompt_attention_mask"].to(device),
            negative_prompt_embeds=cache["negative_prompt_embeds"].to(device),
            negative_prompt_attention_mask=cache["negative_prompt_attention_mask"].to(device),
            max_sequence_length=cache["max_sequence_length"],
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            frame_rate=args.fps,
            sigmas=DISTILLED_SIGMA_VALUES,
            # Distilled runs unguided; these override the pipeline's SFT defaults.
            guidance_scale=1.0,
            audio_guidance_scale=1.0,
            stg_scale=0.0,
            audio_stg_scale=0.0,
            modality_scale=1.0,
            audio_modality_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(args.seed),
            output_type="latent",
            return_dict=False,
            **image_kwargs,
        )

    with Stage("generate:save_latents"):
        checkpoint = args.output.with_suffix(".latents.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as f:
            torch.save({
                "latents": latents.detach().cpu(),
                "audio_latents": audio_latents.detach().cpu(),
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "fps": args.fps,
                "seed": args.seed,
                "prompt": args.prompt,
                **input_metadata,
            }, f)
            f.flush()
            os.fsync(f.fileno())
        print(f"  checkpoint -> {checkpoint}", flush=True)

    with Stage("generate:release"):
        # The transformer and connectors are finished, and dropping them hands 42 GiB of
        # VRAM back to the decode. Moving them to the host instead is what `--offload
        # model` does, and 42 GiB into 45 GiB of system RAM is the configuration that
        # crashed; __init__ guards every component it reads, so None is a valid state.
        pipe.transformer = None
        pipe.connectors = None
        gc.collect()
        torch.cuda.empty_cache()

    # Mirrors the non-latent tail of LTX2Pipeline.__call__. Lifting it out is only safe
    # because this VAE has timestep_conditioning=False -- no decode-time noise mixing --
    # and the latent branch has already denormalized both sets of latents. no_grad because
    # we are outside the @torch.no_grad() on __call__.
    with Stage("generate:decode"), torch.no_grad():
        video = pipe.vae.decode(latents.to(pipe.vae.dtype), None, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="np")
        mel = pipe.audio_vae.decode(audio_latents.to(pipe.audio_vae.dtype), return_dict=False)[0]
        audio = pipe.vocoder(mel)

    with Stage("generate:encode"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = pipe.vocoder.config.output_sampling_rate
        # .float().cpu() is load-bearing: the audio tensor comes back on the GPU in
        # bf16, and handing that to the muxer is the ComfyUI-LTXVideo #361 crash.
        encode_video(
            (video[0].clip(0, 1) * 255).round().astype("uint8"),
            fps=int(args.fps),
            output_path=str(args.output),
            audio=audio[0].float().cpu(),
            audio_sample_rate=sample_rate,
        )

    size_mib = args.output.stat().st_size / 1024**2
    total = sum(e["seconds"] for e in Stage.log)
    print(f"\nwrote {args.output} ({size_mib:.1f} MiB, audio @ {sample_rate} Hz) in {total:.0f}s total")

    metrics = args.output.with_suffix(".metrics.json")
    metrics.write_text(json.dumps({
        "prompt": args.prompt,
        "width": args.width, "height": args.height, "frames": args.frames, "fps": args.fps,
        "seed": args.seed, "offload": args.offload, "steps": len(DISTILLED_SIGMA_VALUES),
        "output_mib": round(size_mib, 1), "audio_sample_rate": sample_rate,
        "stages": Stage.log,
        **input_metadata,
    }, indent=2))
    print(f"metrics -> {metrics}")
    return 0


def run_encode_subprocess(args) -> None:
    """Re-invoke this script for the encode phase, so the 23 GiB is freed by exit()."""
    cmd = [
        sys.executable, "-m", "ltx25_cli", "generate",
        "--phase", "encode",
        "--model", str(args.model),
        "--prompt", args.prompt,
        "--negative-prompt", args.negative_prompt,
        "--max-sequence-length", str(args.max_sequence_length),
        "--embeds", str(args.embeds),
    ]
    subprocess.run(cmd, check=True)


def decode_saved(args):
    from diffusers.utils import encode_video
    checkpoint, output = args.input, args.output
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)

    with Stage("decode:load"):
        pipe = load_pipeline(
            args.model,
            keep={"vae", "audio_vae", "vocoder"},
        )
        pipe.to("cuda")
        if args.vae_tiling:
            pipe.vae.enable_tiling()
            pipe.vae.use_framewise_decoding = True

    with Stage("decode:video"), torch.inference_mode():
        latent = data["latents"].to("cuda", dtype=pipe.vae.dtype)
        video = pipe.vae.decode(latent, None, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="np")[0]
        frames = (video.clip(0, 1) * 255).round().astype("uint8")
        del latent, video
        torch.cuda.empty_cache()

    with Stage("decode:audio"), torch.inference_mode():
        latent = data["audio_latents"].to("cuda", dtype=pipe.audio_vae.dtype)
        mel = pipe.audio_vae.decode(latent, return_dict=False)[0]
        audio = pipe.vocoder(mel)[0].float().cpu()

    output.parent.mkdir(parents=True, exist_ok=True)
    with Stage("write:mp4"):
        encode_video(
            frames,
            fps=int(data["fps"]),
            output_path=str(output),
            audio=audio,
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        )

    print(f"完成: {output.resolve()} ({output.stat().st_size / 1024**2:.1f} MiB)")

    output.with_suffix(".metrics.json").write_text(json.dumps({
        "input": str(checkpoint), "output": str(output), "fps": data["fps"],
        "audio_sample_rate": pipe.vocoder.config.output_sampling_rate,
        "stages": Stage.log,
    }, indent=2))
    return 0
