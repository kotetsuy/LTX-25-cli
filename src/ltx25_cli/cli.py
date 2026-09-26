"""Command-line interface; help and argument validation do not load torch."""
import argparse
from pathlib import Path


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(prog="ltx25", description="LTX-2.5 video generation on ROCm")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate a video from text and an optional start image")
    generate.add_argument("--image", type=Path, help="Start-frame image (compose separate photos first)")
    generate.add_argument("--image-fit", choices=["contain", "cover"], default="contain",
                          help="contain: preserve full image with black padding; cover: center crop")
    generate.add_argument("--model", type=Path, default=Path("models/ltx-2.5-diffusers"))
    generate.add_argument("--phase", choices=["both", "encode", "generate"], default="both")
    generate.add_argument("--prompt", default="A cinematic shot of a red fox walking through a snowy forest at dawn, "
                          "the camera tracking alongside, snow crunching underfoot.")
    generate.add_argument("--negative-prompt")
    generate.add_argument("--max-sequence-length", type=positive, default=1024)
    generate.add_argument("--embeds", type=Path)
    generate.add_argument("--width", type=positive, default=768)
    generate.add_argument("--height", type=positive, default=512)
    generate.add_argument("--frames", type=positive, default=121)
    generate.add_argument("--fps", type=positive, default=24)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--offload", choices=["none", "model", "sequential"], default="none")
    generate.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True,
                          help="Spatial and temporal VAE tiling (default: enabled)")
    generate.add_argument("--output", type=Path, default=Path("outputs/generated.mp4"))
    decode = sub.add_parser("decode", help="Decode saved video/audio latents without denoising")
    decode.add_argument("--input", required=True, type=Path, help="Saved .latents.pt checkpoint")
    decode.add_argument("--output", required=True, type=Path)
    decode.add_argument("--model", type=Path, default=Path("models/ltx-2.5-diffusers"))
    decode.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.model / "model_index.json").is_file():
        parser.error(f"model_index.json not found in {args.model}; download the model or use --model")
    args.model = args.model.resolve()
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must have the .mp4 extension")
    if args.command == "generate":
        if args.width % 32 or args.height % 32:
            parser.error("--width and --height must be multiples of 32")
        if (args.frames - 1) % 8:
            parser.error("--frames must be 8n+1 (e.g. 49 or 121)")
        args.embeds = args.embeds or args.output.with_suffix(".embeds.pt")
        reserved = [args.output, args.output.with_suffix(".latents.pt"),
                    args.output.with_suffix(".metrics.json")]
        if args.embeds.resolve() in [path.resolve() for path in reserved]:
            parser.error("--embeds must differ from the video, latent and metrics output paths")
        if args.image is not None:
            if not args.image.is_file():
                parser.error(f"image not found: {args.image}")
            args.image = args.image.resolve()
            if args.image in [path.resolve() for path in [*reserved, args.embeds]]:
                parser.error("--image must differ from all output and embedding paths")
            from .images import load_start_image
            try:
                load_start_image(args.image, args.width, args.height, args.image_fit)
            except (OSError, ValueError) as error:
                parser.error(f"cannot read --image: {error}")
    else:
        if not args.input.is_file():
            parser.error(f"checkpoint not found: {args.input}")
        if args.input.resolve() == args.output.resolve():
            parser.error("--input and --output must be different paths")
        if args.input.resolve() == args.output.with_suffix(".metrics.json").resolve():
            parser.error("--input must differ from the metrics output path")
    from . import runner
    if not runner.torch.cuda.is_available():
        parser.error("GPU is unavailable; check the ROCm driver and gfx1151 torch environment")
    runner.Stage.log.clear()
    if args.command == "decode":
        return runner.decode_saved(args)
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEGATIVE_PROMPT
    if args.phase == "encode":
        runner.phase_encode(args)
        return 0
    if args.phase == "both":
        if runner.load_embeds(args.embeds, args) is None:
            runner.run_encode_subprocess(args)
        else:
            print(f"[encode] reusing {args.embeds}", flush=True)
    return runner.phase_generate(args)
