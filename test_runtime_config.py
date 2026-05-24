import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

if "einops" not in sys.modules:
    import types

    einops_module = types.ModuleType("einops")
    einops_module.rearrange = lambda x, *args, **kwargs: x
    einops_layers_module = types.ModuleType("einops.layers")
    einops_layers_torch_module = types.ModuleType("einops.layers.torch")

    class Rearrange(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    einops_layers_torch_module.Rearrange = Rearrange
    sys.modules["einops"] = einops_module
    sys.modules["einops.layers"] = einops_layers_module
    sys.modules["einops.layers.torch"] = einops_layers_torch_module

from ip_adapter.attention_processor import get_phase_mix_table
from ip_adapter.ip_adapter import IPAdapterPlusXL


class DummyClipImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        if not isinstance(images, list):
            images = [images]
        batch = []
        for image in images:
            arr = np.array(image.convert("RGB"), dtype=np.float32)
            mean_value = float(arr.mean() / 255.0)
            batch.append(torch.full((3, 4, 4), mean_value, dtype=torch.float32))
        return SimpleNamespace(pixel_values=torch.stack(batch))


class DummyVisionEncoder:
    def __call__(self, pixel_values, output_hidden_states=True):
        batch_size = pixel_values.shape[0]
        base = pixel_values.mean(dim=(1, 2, 3)).view(batch_size, 1, 1)
        token_offsets = torch.arange(4, dtype=pixel_values.dtype).view(1, 4, 1) / 10.0
        feature_offsets = torch.arange(8, dtype=pixel_values.dtype).view(1, 1, 8) / 100.0
        embeds = base + token_offsets + feature_offsets
        return SimpleNamespace(hidden_states=[embeds, embeds])


def make_stub_adapter():
    adapter = IPAdapterPlusXL.__new__(IPAdapterPlusXL)
    adapter.device = "cpu"
    adapter.image_proj_model = torch.nn.Identity()
    adapter.clip_image_processor = DummyClipImageProcessor()
    adapter.image_encoder = DummyVisionEncoder()
    return adapter


def make_test_image(color):
    return Image.new("RGB", (8, 8), color=color)


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_files_do_not_import_mode_profiles(self):
        runtime_files = [
            "infer_style_plus_color_texture.py",
            os.path.join("ip_adapter", "attention_processor.py"),
            os.path.join("ip_adapter", "ip_adapter.py"),
        ]
        root = os.path.dirname(__file__)
        for rel_path in runtime_files:
            abs_path = os.path.join(root, rel_path)
            with open(abs_path, "r", encoding="utf-8") as handle:
                contents = handle.read()
            self.assertNotIn("mode_profiles", contents, msg=rel_path)

    def test_fixed_phase_mix_table(self):
        phase_mix = get_phase_mix_table("down_blocks.0.attentions.1")
        self.assertEqual(phase_mix["mid"], (0.10, 0.80, 0.10))

        phase_mix = get_phase_mix_table("mid_block.attentions.0")
        self.assertEqual(phase_mix["back"], (0.18, 0.54, 0.28))

        phase_mix = get_phase_mix_table("up_blocks.1.attentions.1")
        self.assertEqual(phase_mix["back"], (0.24, 0.30, 0.46))

    def test_fixed_layer_stream_scales(self):
        adapter = make_stub_adapter()
        self.assertEqual(adapter.get_layer_stream_scales("down_blocks.0.attentions.1"), [0.80, 1.50, 0.18])
        self.assertEqual(adapter.get_layer_stream_scales("mid_block.attentions.0"), [0.95, 1.05, 0.95])
        self.assertEqual(adapter.get_layer_stream_scales("up_blocks.1.attentions.1"), [1.10, 0.65, 1.00])

    def test_get_image_embeds_smoke_all_and_partial_refs(self):
        adapter = make_stub_adapter()
        color_img = make_test_image((255, 0, 0))
        texture_img = make_test_image((0, 255, 0)).convert("L")
        geometry_img = make_test_image((0, 0, 255))
        color_gray = color_img.convert("L")

        all_embeds, all_uncond = adapter.get_image_embeds(
            clr_ref_img=color_img,
            clr_texture_ref_img=color_gray,
            texture_ref_img=texture_img,
            geometry_ref_img=geometry_img,
            color_scale=1.0,
            texture_scale=1.0,
            geometry_scale=1.0,
            punish_weight=0.0,
        )
        self.assertEqual(all_embeds.shape, (1, 12, 8))
        self.assertEqual(all_uncond.shape, (1, 12, 8))

        no_geometry_embeds, _ = adapter.get_image_embeds(
            clr_ref_img=color_img,
            clr_texture_ref_img=color_gray,
            texture_ref_img=texture_img,
            geometry_ref_img=None,
            color_scale=1.0,
            texture_scale=1.0,
            geometry_scale=1.0,
            punish_weight=0.0,
        )
        self.assertEqual(no_geometry_embeds.shape, (1, 12, 8))

        no_texture_embeds, _ = adapter.get_image_embeds(
            clr_ref_img=color_img,
            clr_texture_ref_img=color_gray,
            texture_ref_img=None,
            geometry_ref_img=geometry_img,
            color_scale=1.0,
            texture_scale=1.0,
            geometry_scale=1.0,
            punish_weight=0.0,
        )
        self.assertEqual(no_texture_embeds.shape, (1, 12, 8))

        geometry_only_embeds, _ = adapter.get_image_embeds(
            clr_ref_img=None,
            clr_texture_ref_img=None,
            texture_ref_img=None,
            geometry_ref_img=geometry_img,
            color_scale=1.0,
            texture_scale=1.0,
            geometry_scale=1.0,
            punish_weight=0.0,
        )
        self.assertEqual(geometry_only_embeds.shape, (1, 12, 8))

    def test_get_image_embeds_requires_at_least_one_reference(self):
        adapter = make_stub_adapter()
        with self.assertRaises(ValueError):
            adapter.get_image_embeds(
                clr_ref_img=None,
                clr_texture_ref_img=None,
                texture_ref_img=None,
                geometry_ref_img=None,
                texture_ref_img2=None,
                punish_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
