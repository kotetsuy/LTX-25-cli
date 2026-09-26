"""Exercise pipeline selection and conditioning without loading GPU weights."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image
import ltx25_cli


class ImageRunnerTests(unittest.TestCase):
    def load_runner(self):
        spec = importlib.util.spec_from_file_location(
            'ltx25_cli._test_runner', Path(ltx25_cli.__file__).with_name('runner.py'))
        module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'torch': MagicMock()}):
            spec.loader.exec_module(module)
        return module

    def test_select_pipeline_class(self):
        runner = self.load_runner()
        diffusers = MagicMock()
        with patch.dict('sys.modules', {'diffusers': diffusers}):
            runner.load_pipeline(Path('model'), {'vae'}, image_to_video=True)
            diffusers.LTX2ImageToVideoPipeline.from_pretrained.assert_called_once()
            diffusers.LTX2Pipeline.from_pretrained.assert_not_called()
            runner.load_pipeline(Path('model'), {'vae'})
            diffusers.LTX2Pipeline.from_pretrained.assert_called_once()

    def test_conditioning_reaches_denoise_only_for_image_mode(self):
        runner = self.load_runner()
        with tempfile.TemporaryDirectory() as directory:
            photo = Path(directory) / 'start.png'
            Image.new('RGB', (80, 40), 'red').save(photo)
            for image_path in (None, photo):
                with self.subTest(image=image_path):
                    args = SimpleNamespace(image=image_path, image_fit='contain',
                        width=64, height=64, frames=9, fps=24, seed=42,
                        embeds=Path('cache.pt'), model=Path('model'),
                        vae_tiling=True, offload='none')
                    cache = {key: MagicMock() for key in (
                        'prompt_embeds', 'prompt_attention_mask',
                        'negative_prompt_embeds', 'negative_prompt_attention_mask')}
                    cache.update(padding_side='left', max_sequence_length=1024)
                    pipe = MagicMock(side_effect=RuntimeError('denoise reached'))
                    with patch.dict('sys.modules', {
                        'diffusers.pipelines.ltx2.utils': MagicMock(),
                        'diffusers.utils': MagicMock(),
                    }), patch.object(runner, 'Stage'), patch.object(
                        runner, 'load_embeds', return_value=cache
                    ), patch.object(runner, 'load_pipeline', return_value=pipe) as loader:
                        with self.assertRaisesRegex(RuntimeError, 'denoise reached'):
                            runner.phase_generate(args)
                    self.assertEqual(loader.call_args.kwargs['image_to_video'], image_path is not None)
                    kwargs = pipe.call_args.kwargs
                    self.assertEqual(kwargs['output_type'], 'latent')
                    if image_path is None:
                        self.assertNotIn('image', kwargs)
                    else:
                        self.assertEqual(kwargs['image'].size, (64, 64))
                        self.assertEqual(kwargs['image'].getpixel((32, 32)), (255, 0, 0))
