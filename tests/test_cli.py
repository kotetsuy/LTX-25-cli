import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ltx25_cli.cli import main, build_parser


class CliTests(unittest.TestCase):
    def test_image_validation_and_dispatch(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model_index.json').write_text('{}')
            photo = root / 'start.png'
            argv = ['generate', '--model', directory, '--image', str(photo),
                    '--phase', 'generate', '--image-fit', 'cover']
            for content in (None, b'not an image'):
                if content is not None:
                    photo.write_bytes(content)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(argv)
                self.assertEqual(error.exception.code, 2)
            Image.new('RGB', (64, 32), 'red').save(photo)
            fake = MagicMock()
            fake.phase_generate.return_value = 0
            with patch('ltx25_cli.runner', fake, create=True), patch.dict(
                'sys.modules', {'diffusers.pipelines.ltx2.utils': MagicMock()}
            ):
                self.assertEqual(main(argv), 0)
                args = fake.phase_generate.call_args.args[0]
                self.assertEqual(args.image, photo.resolve())
                self.assertEqual(args.image_fit, 'cover')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([*argv, '--embeds', str(photo)])

    def test_start_image_fit_and_orientation(self):
        from PIL import Image
        from ltx25_cli.images import load_start_image
        with tempfile.TemporaryDirectory() as directory:
            photo = Path(directory) / 'start.png'
            Image.new('RGB', (80, 40), 'red').save(photo)
            contained = load_start_image(photo, 64, 64)
            self.assertEqual(contained.size, (64, 64))
            self.assertEqual(contained.getpixel((32, 0)), (0, 0, 0))
            self.assertEqual(contained.getpixel((32, 32)), (255, 0, 0))
            covered = load_start_image(photo, 64, 64, 'cover')
            self.assertEqual(covered.getpixel((32, 0)), (255, 0, 0))
            source = Image.new('RGB', (80, 40), 'red')
            exif = source.getexif()
            exif[274] = 6
            source.save(photo, exif=exif)
            oriented = load_start_image(photo, 64, 64)
            self.assertEqual(oriented.getpixel((0, 32)), (0, 0, 0))
            self.assertEqual(oriented.getpixel((32, 0)), (255, 0, 0))

    def test_help_without_loading_torch(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-c',
            'import sys; from ltx25_cli.cli import main; '
            '\ntry: main(["--help"])\nexcept SystemExit: pass\n'
            'assert "torch" not in sys.modules'], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tiling_default_and_override(self):
        parser = build_parser()
        self.assertTrue(parser.parse_args(['generate']).vae_tiling)
        self.assertFalse(parser.parse_args(['generate', '--no-vae-tiling']).vae_tiling)

    def test_invalid_dimensions_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'model_index.json').write_text('{}')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(['generate', '--model', directory, '--width', '123'])
            self.assertEqual(error.exception.code, 2)

    def test_decode_dispatch_and_missing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / 'model_index.json').write_text('{}')
            checkpoint = model / 'saved.pt'
            argv = ['decode', '--model', directory, '--input', str(checkpoint),
                    '--output', str(model / 'result.mp4')]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            checkpoint.touch()
            fake = MagicMock()
            fake.decode_saved.return_value = 0
            with patch('ltx25_cli.runner', fake, create=True):
                self.assertEqual(main(argv), 0)
                args = fake.decode_saved.call_args.args[0]
                self.assertEqual(args.input, checkpoint)
                self.assertTrue(args.vae_tiling)


if __name__ == '__main__':
    unittest.main()
